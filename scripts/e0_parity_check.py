#!/usr/bin/env python
"""E0 — Parity gate: TransformerLens vs HuggingFace on Gemma-2-2B.

The first gate of the whole project (plan §3, E0). Gemma-2 applies attention
and final-logit softcapping; legacy TransformerLens weight processing
(LayerNorm folding / center_unembed) breaks the tanh-softcap invariance, so we
load with raw-preserving flags and compare final-token logits in fp32 against
a plain HuggingFace forward pass.

Gate: mean KL(HF ‖ TL) < 1e-3 AND argmax agreement >= 99%  -> PASS
On FAIL: fall back to pyvene (works on raw HF weights directly).

Usage:
    python scripts/e0_parity_check.py --n 100
    python scripts/e0_parity_check.py --hf-id google/gemma-2-2b --tl-name gemma-2-2b
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Fail fast with a clear message when run outside the project venv (the DLVM
# system Python has torch preinstalled but not our deps — a classic trap).
if ".venv" not in sys.prefix:
    print(
        "[warn] ดูเหมือนยังไม่ได้ activate venv (sys.prefix = %s)\n"
        "       รัน:  source .venv/bin/activate   แล้วรันสคริปต์ใหม่\n"
        "       (ทุกหน้าต่าง tmux ใหม่ต้อง activate ใหม่ทุกครั้ง)" % sys.prefix
    )

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval.metrics import bootstrap_ci  # noqa: E402  (reuse project metrics)

DEFAULT_PROMPTS = ROOT / "configs" / "parity_prompts.txt"
DEFAULT_OUT = ROOT / "results" / "e0_parity" / "report.json"
KL_THRESHOLD = 1e-3
ARGMAX_THRESHOLD = 0.99


def load_prompts(path: Path, n: int) -> list[str]:
    prompts = [ln.strip() for ln in path.read_text().splitlines()]
    prompts = [p for p in prompts if p and not p.startswith("#")]
    if n > len(prompts):
        print(f"[warn] only {len(prompts)} prompts available (asked {n})")
    return prompts[:n]


@torch.no_grad()
def hf_last_token_logits(hf_id: str, prompts: list[str], device: str, dtype=torch.bfloat16) -> torch.Tensor:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=dtype
    ).to(device).eval()
    out = []
    for i, p in enumerate(prompts, 1):
        enc = tok(p, return_tensors="pt").to(device)
        logits = model(**enc).logits[0, -1].float().cpu()
        out.append(logits)
        print(f"\r[HF] {i}/{len(prompts)}", end="")
    print()
    del model
    torch.cuda.empty_cache()
    return torch.stack(out)


@torch.no_grad()
def tl_last_token_logits(tl_name: str, prompts: list[str], device: str, dtype=torch.bfloat16) -> torch.Tensor:
    from importlib.metadata import version as pkg_version

    major = pkg_version("transformer-lens").split(".")[0]
    if major != "2":
        raise SystemExit(
            f"[abort] TransformerLens {major}.x detected — this project targets TL 2.x "
            f"(HookedTransformer API; TL 4.x replaced it with TransformerBridge).\n"
            f"        Fix:  pip install 'transformer_lens>=2.4,<3.0'"
        )

    from transformer_lens import HookedTransformer

    def _load():
        # Raw-weight-preserving flags — the Gemma-2 softcap-safe way to load
        # (see references.md [T5] and GCP_SETUP.md §4). Retry without flags if
        # the installed TransformerLens version rejects them.
        try:
            return HookedTransformer.from_pretrained(
                tl_name,
                dtype=dtype,
                device=device,
                center_unembed=False,
                center_writing_weights=False,
                fold_ln=False,
            )
        except TypeError:
            print("[warn] TransformerLens rejected raw-weight flags; loading with defaults")
            return HookedTransformer.from_pretrained(tl_name, dtype=dtype, device=device)

    model = _load().eval()
    out = []
    for i, p in enumerate(prompts, 1):
        logits = model(p)[0, -1].float().cpu()
        out.append(logits)
        print(f"\r[TL] {i}/{len(prompts)}", end="")
    print()
    del model
    torch.cuda.empty_cache()
    return torch.stack(out)


def compare(hf_logits: torch.Tensor, tl_logits: torch.Tensor) -> dict:
    log_p_hf = F.log_softmax(hf_logits, dim=-1)
    log_p_tl = F.log_softmax(tl_logits, dim=-1)
    # KL(HF ‖ TL): reference = HuggingFace
    kl = F.kl_div(log_p_tl, log_p_hf.exp(), reduction="none").sum(-1)
    argmax_agree = (hf_logits.argmax(-1) == tl_logits.argmax(-1)).float()
    return {
        "per_prompt_kl": kl.tolist(),
        "kl_reverse_mean": float(F.kl_div(log_p_hf, log_p_tl.exp(), reduction="none").sum(-1).mean()),
        "argmax_agree": argmax_agree.tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="google/gemma-2-2b")
    ap.add_argument("--tl-name", default="gemma-2-2b")
    ap.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--dtype",
        choices=["bfloat16", "float32"],
        default="bfloat16",
        help="load dtype for both frameworks; float32 diagnoses whether a marginal bf16 KL is rounding noise",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    if args.dtype == "float32":
        args.out = args.out.with_name(args.out.name.replace("report", "report_fp32"))

    prompts = load_prompts(args.prompts, args.n)
    print(f"[E0] parity check — {len(prompts)} prompts, device={args.device}")

    hf_logits = hf_last_token_logits(args.hf_id, prompts, args.device, dtype)
    tl_logits = tl_last_token_logits(args.tl_name, prompts, args.device, dtype)

    cmp = compare(hf_logits, tl_logits)
    kl = torch.tensor(cmp["per_prompt_kl"])
    agree = torch.tensor(cmp["argmax_agree"])
    mean_kl, kl_lo, kl_hi = bootstrap_ci(kl)
    agree_mean = float(agree.mean())

    passed = (mean_kl < KL_THRESHOLD) and (agree_mean >= ARGMAX_THRESHOLD)

    report = {
        "experiment": "E0_parity",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hf_id": args.hf_id,
        "tl_name": args.tl_name,
        "n_prompts": len(prompts),
        "dtype": args.dtype,
        "mean_kl_hf_tl": mean_kl,
        "median_kl": float(kl.median()),
        "p90_kl": float(kl.quantile(0.9)),
        "kl_ci95": [kl_lo, kl_hi],
        "max_kl": float(kl.max()),
        "kl_reverse_mean": cmp["kl_reverse_mean"],
        "argmax_agreement": agree_mean,
        "gate": {"mean_kl_below": KL_THRESHOLD, "argmax_agreement_at_least": ARGMAX_THRESHOLD},
        "pass": passed,
        "versions": _versions(),
        "note_on_fail": "If FAIL: fall back to pyvene (declarative interventions on raw HF weights) — plan §3 E0 / GCP_SETUP.md §4.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print(f"mean KL(HF‖TL) : {mean_kl:.2e}   (gate < {KL_THRESHOLD:.0e})")
    print(f"median / p90   : {report['median_kl']:.2e} / {report['p90_kl']:.2e}")
    print(f"max  KL        : {report['max_kl']:.2e}")
    print(f"argmax agree   : {agree_mean:.1%}  (gate ≥ {ARGMAX_THRESHOLD:.0%})")
    print(f"RESULT         : {'PASS ✅' if passed else 'FAIL ❌ — use pyvene fallback'}")
    print(f"report         : {args.out}")
    print("=" * 60)
    return 0 if passed else 2


def _git_rev() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _versions() -> dict:
    import transformers

    v = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "git_rev": _git_rev(),
    }
    try:
        import transformer_lens

        v["transformer_lens"] = transformer_lens.__version__
    except Exception:
        v["transformer_lens"] = "unknown"
    return v


if __name__ == "__main__":
    raise SystemExit(main())
