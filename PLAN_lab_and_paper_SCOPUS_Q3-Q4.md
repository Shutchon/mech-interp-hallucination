# แผนการทำ Lab และเขียน Paper ฉบับปรับปรุงสำหรับ SCOPUS Q3–Q4

**หัวข้อวิจัย:** *Locating and Mitigating Factual Hallucination Circuits in Small Language Models via Activation Patching*
(การระบุตำแหน่งและระงับวงจรสร้างข้อมูลเท็จในโมเดลภาษาขนาดเล็กด้วยเทคนิค Activation Patching)

**เอกสารนี้คือ:** แผนงานรวม (Master Plan) แปลงเค้าโครงเดิมที่มุ่งส่ง Workshop ให้เป็นงานวิจัยเชิงวารสาร SCOPUS Q3–Q4 พร้อม experiment protocol, สถิติ, งบ compute, ตารางเวลา และแผนเขียน paper รายส่วน

---

## 0. สรุปการเปลี่ยนแปลงสำคัญจากเค้าโครงเดิม (ทำไมต้องปรับ)

| มิติ | เค้าโครงเดิม (Workshop) | แผนปรับปรุง (SCOPUS Q3–Q4) | เหตุผล |
|---|---|---|---|
| ประเภทผลงาน | Extended abstract / short paper | Full paper 8–15 หน้า + 30–50 อ้างอิง | วารสารต้องการ methodology ครบถ้วน ไม่ใช่ผลเบื้องต้น |
| โมเดล | Gemma-2 2B + 9B | Gemma-2-2B (หลัก) + **Llama-3.2-1B, Qwen2.5-1.5B** (cross-model) + Gemma-2-9B (เฉพาะ coarse sweep) | ผลข้ามโมเดล = ข้อเสนอ "generalizable finding" ซึ่งเป็นสิ่ง reviewers วารสารถามเสมอ และโมเดลเล็กอื่นถูกกว่าการรัน 9B เต็มรูปแบบ |
| การหลอกล่อ (corruption) | เปลี่ยนชื่อ entity | **2 แนวทาง:** (a) ROME-style embedding noise (sanity check มาตรฐาน) + (b) **misleading-context prompts** (แข่งขันระหว่างความจำกับบริบท — หัวใจของ RQ2) | เพิ่มความหลากหลายของ evidence และเชื่อมกับ hallucination จริง ไม่ใช่แค่ causal tracing เชิงเทคนิค |
| การเลือกวงจร | ดูจาก heatmap แล้วเลือก | เลือกบน **development split ของ relations** แล้วรายงานผลบน **held-out relations** | กันข้อครหา "circular analysis / cherry-picking" ซึ่งเป็นจุดอ่อนที่ reviewers จับได้ง่ายที่สุด |
| สถิติ | ไม่มีระบุ | McNemar test, paired bootstrap 95% CI, BH-FDR correction, dose–response curve | Q3–Q4 ยอมรับ scope เล็กได้ แต่ "วิธีวิทยาแน่น" เป็นเงื่อนไขขั้นต่ำ |
| Deliverable สาย venue | BlackboxNLP / MI Workshop | วารสาร Scopus Q3–Q4 (ดู §8) + GitHub artifact | ตรงเป้าหมายใหม่ |

**หลักคิดของการปรับ:** งบ 5,000 บาท + 1 คน ทำได้ scope แบบ "small but rigorous" — Q3–Q4 ไม่ได้ต้องการ novelty ระดับ NeurIPS แต่ต้องการ **ความซ้ำได้ (reproducibility), การวัดผลที่โปร่งใส, และการรายงานที่ไม่ overclaim** แผนนี้จึงออกแบบให้ทุก claim มีตัวเลข + ช่วงความเชื่อมั่น + การทดสอบบนข้อมูลที่ไม่ได้ใช้เลือกวงจรมาก่อน

---

## 1. ขอบเขตวิจัย (Scope) — ฉบับ final

- **โมเดลหลัก:** Gemma-2-2B-it / Gemma-2-2B (26 layers, 8 heads, hidden 2304)
- **โมเดลเทียบข้ามสถาปัตยกรรม:** Llama-3.2-1B (16 layers, 32 heads), Qwen2.5-1.5B (28 layers, 12 heads)
- **โมเดลขยายผล (แค่ coarse sweep):** Gemma-2-9B (42 layers, 16 heads) — ใช้ subset เล็ก เพราะ 9B กินเวลา ~5 เท่า
- **ภาษา:** English เท่านั้น (จะระบุเป็น limitation)
- **นิยาม Hallucination ในงานนี้:** การตอบข้อเท็จจริงผิดโดยวัดได้เชิงโปรแกรม (greedy argmax ไม่ตรง ground truth หรือตรงตาม distractor) บน prompt ที่มีบริบทหลอกล่อ — *ไม่ใช่* hallucination แบบ open-ended ที่วัดยาก
- **สิ่งที่จะไม่ทำ (กัน scope creep):** model editing (ROME/MEMIT), fine-tuning, multilingual, LLM-as-judge, โมเดล > 9B
- **ทางเลือกโมเดลรุ่นใหม่กว่า (ตัดสินใจที่ E0):** `gemma-3-1b-it` (2025 — ตระกูล 3 ขนาด 1B มีบน HF เฉพาะแบบ instruction-tuned ไม่มีตัว base; ถ้าใช้ต้อง apply chat template สม่ำเสมอและระบุใน paper) หรือ Qwen3-0.6B/1.7B (ต้องปิด thinking mode ก่อนทำ patching) — ข้อแลกเปลี่ยน: ใหม่กว่าแต่ TransformerLens รองรับ Gemma-3 ผ่าน issue #898 ต้องเช็คสถานะจริง · **Gemma 4 (เม.ย. 2026, Apache 2.0)** ยังไม่เหมาะ: รุ่นเล็กสุดเป็นสถาปัตยกรรมแนว edge (E2B) และไม่มี tooling mech interp รองรับ — ใช้เป็น future work + เหตุผลตอบ reviewer ว่าทำไมเลือก Gemma-2 (tooling maturity + Gemma Scope + literature comparability) · **คำแนะนำ: หลักยังเป็น Gemma-2** โดยเพิ่มตัวรุ่นใหม่ 1 ตัวเป็นการทดสอบ "ความเป็นปัจจุบัน" ถ้า E0 ผ่าน

---

## 2. คำถามวิจัย (ปรับปรุงให้วัดผลได้ชัดเจน)

1. **RQ1 (Localization):** Heads/MLPs ตำแหน่งใดที่ restoration effect สูงสุดเมื่อ patch จาก clean run ลง corrupted run — และตำแหน่งเหล่านี้ **คงเส้นคงวาข้ามโมเดลและข้าม relation types หรือไม่** (วัดด้วย rank correlation ระหว่างโมเดล)
2. **RQ2 (Competition):** เมื่อมีบริบทหลอกล่อ โมเดลเปลี่ยนคำตอบเป็นตามบริบท (context-following) หรือคงความจำเดิม (memory-following) — component กลุ่มไหนที่ patch แล้ว "ดึงกลับ" สู่ความจริงได้มากที่สุด และสัดส่วนการชนะของบริบทขึ้นกับตำแหน่ง layer อย่างไร
3. **RQ3 (Intervention & Trade-off):** Mean-ablation / knockout บน top-k heads ที่เลือกจาก development split ลด hallucination rate บน **held-out relations** ได้เท่าไร เทียบกับ "ต้นทุน" ที่เกิดกับ MMLU / WikiText-2 PPL — โดยรายงานเป็น **dose–response curve** (k = 1…20)

---

## 3. แผนการทดลอง (Lab Plan)

### โครงสร้างรวมของ pipeline

```
E0 Setup+Parity → E1 Data → E2 Baselines → E3 Patching sweeps → E4 Circuit analysis
                                                          ↓
                                   E5 Interventions + full evals → E6 Generalization → E7 (stretch) Path patching
```

### E0 — Environment & Parity Check (สัปดาห์ 1) ⚠️ **Gate แรกที่ห้ามข้าม**

| รายการ | รายละเอียด |
|---|---|
| สิ่งที่ทำ | ติดตั้ง TransformerLens + circuitsvis + lm-eval-harness บน VM L4; ดาวน์โหลดโมเดล (Gemma เป็น gated model — ต้อง accept license ที่ HF และตั้ง `HF_TOKEN`); รันสคริปต์ patching 1 head แบบ smoke test |
| **Parity check** | เทียบ logit จาก TransformerLens กับ HuggingFace forward pass ตรงๆ บน 100 prompts (รายงาน max abs diff / KL) — **ยืนยันแล้วว่า Gemma-2 softcapping ทำให้ค่าเพี้ยนได้จริง** (มี bug report ว่า HF กับ HookedTransformer ให้ output ต่างกันเล็กน้อย) สาเหตุหลัก: LayerNorm folding/`center_unembed` ทำลาย invariance ของ tanh softcap + dtype ผสม float32/bfloat16 · **แนวปฏิบัติ:** ใช้ TransformerLens รุ่นใหม่ (default อนุรักษ์ raw HF weights), ตั้ง `center_unembed=False`, คุม dtype คงที่, เทียบ post-softcap logits · **Fallback อันดับแรกถ้าไม่ผ่าน: `pyvene`** (Stanford, ICLR 2025 — ทำ interchange intervention บน HF model ตรงๆ แบบ declarative ไม่ต้องแปลง weight) แล้วจึงเป็น `nnsight` หรือ PyTorch hooks เขียนเอง |
| เกณฑ์ผ่าน | ค่า KL(TL ‖ HF) บน 100 prompt ต่ำกว่า 1e-3 และ argmax ตรงกัน ≥ 99% |
| **ผลลัพธ์ (ผ่านแล้ว — ต.ค. 2026)** | **PASS (มีเงื่อนไข):** fp32 → mean KL **1.32e-05**, median 6.1e-06, argmax **100%** · bf16 → 1.21e-3 (เกินเกณฑ์เล็กน้อย = rounding noise) · **คำตัดสิน: ใช้ TransformerLens 2.x + bf16 สำหรับทุก sweep**, ระบุ noise floor ~1e-3 ใน Methodology, ตรวจซ้ำ top-k heads ด้วย fp32 ตอนจบ E3 · report: `results/e0_parity/report{,_fp32}.json` |
| ต้นทุน | ~2 GPU-h |

### E1 — Dataset Preparation (สัปดาห์ 1–2)

| ชุดข้อมูล | บทบาท | ขนาดที่ใช้ |
|---|---|---|
| **CounterFact** | หัวใจของ patching (มี true target + counterfactual target ต่อ fact) | กรองเหลือ **2,000 triples** ที่ผ่านเกณฑ์ "known fact": โมเดลตอบ true target เป็น argmax และ logit gap (true − รองชั้น) ≥ 0.5 |
| **Misleading-prompt suite (สร้างเอง)** | สำหรับ RQ2/RQ3 — template 3 แบบ เช่น *"Earlier you learned that the Eiffel Tower is in Rome. The Eiffel Tower is located in the city of"* | 3 templates × 2,000 facts = 6,000 prompts |
| **TruthfulQA** (MC1/MC2) | downstream วัดความเชื่อผิด | 818 ข้อ ครบชุด (ระวัง: ใช้เพื่อประเมินเท่านั้น ห้ามใช้เลือกวงจร) |
| **MMLU + WikiText-2** | เกณฑ์ว่า "โมเดลไม่พัง" หลัง intervention | MMLU 0-shot (ถ้าช้าใช้ `--limit`), WikiText-2 PPL |
| **PopQA (entity popularity)** | covariate สำหรับ E4 — stratify facts ตามความนิยมของ entity | ดึงเฉพาะ popularity score มา map กับ facts ของเรา (ไม่ต้องใช้ทั้งชุด) |
| **HaluEval (ทางเลือก)** | secondary eval เพิ่มถ้ามีเวลา | subset ของ QA/knowledge task |

**การแบ่ง split (สำคัญกับความน่าเชื่อถือ):** แบ่ง relations เป็น **Dev (50%) / Held-out (50%)** — ใช้ Dev เลือก heads และหา threshold, รายงานผล RQ3 ทั้งหมดบน Held-out เท่านั้น

### E2 — Baselines (สัปดาห์ 2)

- วัด: clean accuracy, corrupted/misleading accuracy (context-following rate), TruthfulQA, MMLU, PPL ของทุกโมเดล
- เก็บเป็นตาราง baseline สำหรับ paper ตั้งแต่ต้น (Table 2 ของ paper)
- ต้นทุน: ~3 GPU-h

### E3 — Activation Patching Sweeps (สัปดาห์ 3–5) — *หัวใจของงาน*

**การออกแบบ coarse-to-fine เพื่อคุมงบ:**

1. **Coarse (block-level):** patch residual stream output ทุก layer × ทุก token position (จำกัด positions: subject tokens + final 3 tokens) → ได้ heatmap ระดับ layer
2. **Mid (component-level):** แยก attention block vs MLP block บน top-5 layers จาก coarse
3. **Fine (head-level):** patch output (z) ของแต่ละ head เฉพาะ layers ที่สนใจ

**สอง corruption regimes ต่อ sweeps ข้างต้น:**
- **(a) Noising→Denoising แบบ ROME:** รบกวน subject embedding ด้วย Gaussian noise เป็น corrupted run (มาตรฐาน เทียบกับ literature ได้)
- **(b) Misleading-context:** corrupted = prompt ที่มีบริบทหลอก / clean = บริบทเป็นกลาง (ตอบ RQ2 โดยตรง)

**ขนาดการทดลอง:**
- Gemma-2-2B: N = 1,000 facts (สุ่มจาก 2,000) — ~8–12 GPU-h
- Llama-3.2-1B / Qwen2.5-1.5B: coarse + head-level บน top layers เท่านั้น, N = 500 — ~6 GPU-h
- Gemma-2-9B: coarse เท่านั้น, N = 500 — ~10–15 GPU-h

**ตัวเลือกประหยัดเวลา:** ถ้า sweep เต็มรูปแบบช้าเกินไป ใช้ **Attribution Patching** (Syed, Rager & Conmy, BlackboxNLP 2024 — ประมาณการ effect ด้วย gradient แทนการรันจริงทุกจุด) เป็นตัวคัดกรองรอบแรก แล้วยืนยัน top candidates ด้วย patching จริง — ต้องรายงานใน paper ว่าใช้เป็นเพียง screening เท่านั้น

**Output:** heatmap (layer × position) และตาราง top-k heads พร้อม restoration effect + 95% CI (bootstrap over facts)

### E4 — Circuit Characterization (สัปดาห์ 5–6)

- Attention pattern ของ top heads ด้วย circuitsvis (เก็บเป็น figure) หรือ HeadVis (2026, รองรับ Gemma 3)
- จัดกลุ่ม heads ที่พบด้วย **taxonomy 4 ระยะของ Zheng et al. 2025 (Patterns survey): Knowledge Recalling / In-Context Identification / Latent Reasoning / Expression Preparation** — ระยะแรกกับระยะสองตรงกับการแข่งขัน memory-vs-context ของเราพอดี ทำให้เชื่อมผลกับสายหลักได้
- วิเคราะห์พฤติกรรม: head ตรวจจับ subject? head ย้ายข้อมูลไป last token? MLP บนช่วงกลาง?
- วัด **ความคงเส้นคงวาข้ามโมเดล:** Spearman rank correlation ของ layer-profile ระหว่าง 4 โมเดล
- **ทดสอบข้ออ้าง "superposition" ของ JuICE (ICML 2025):** ตรวจว่า heads ที่ทรงอิทธิพลใน data เราเอนไปทาง memory เดียว/context เดียว หรือมีบทบาทสองฝั่งพร้อมกัน (วัดจาก patching effect สองทิศ: clean→corrupt และ corrupt→clean) — เป็นจุดเชื่อมโยงกับ literature ล่าสุดโดยตรง
- **ควบคุม confound แบบ copy-suppression (Campregher 2025):** ทดสอบว่า top heads เป็น "selective fact recall" หรือแค่ generic copier/suppressor ด้วย control run ที่แทน distractor ด้วย token กลางๆ ที่ไม่ใช่คำตอบ และการสลับตำแหน่ง true/false target
- **วิเคราะห์ตาม entity popularity (จาก PopQA scores):** circuit/gullibility เปลี่ยนตามความนิยมของ entity หรือไม่ (เชื่อมกับ susceptibility score ของ Du et al. 2024 แบบ mechanistic)
- เกือบทั้งหมดเป็นงาน CPU/วิเคราะห์ ต้นทุน GPU ~2 h

### E5 — Interventions & Full Evaluation (สัปดาห์ 7–9)

- **วิธีแทรกแซง 2 แบบ:** (1) mean-ablation (แทน activation ด้วยค่าเฉลี่ยจาก 64–128 prompts สุ่มของ Dev split), (2) zero-knockout
- **Dose–response:** ปิด k = 1, 2, 5, 10, 20 heads (เรียงตาม restoration effect บน Dev) วัด hallucination rate บน **Held-out** + MMLU + PPL → วาด trade-off curve (figure เด่นของ paper)
- **Baselines ที่ต้องมีเพื่อความเป็นธรรม:** (1) ปิด heads สุ่ม k ตัว (10 ครั้ง รายงาน mean±SD), (2) ปิด heads ที่ activation norm สูงสุด — พิสูจน์ว่า "ตำแหน่งที่ patch ชี้" สำคัญกว่า "ปิด head ใดก็ได้"
- ต้นทุน: ~15 GPU-h

### E6 — Generalization & Robustness (สัปดาห์ 9–10)

- รัน config ที่ดีที่สุดของ E5 บน Llama/Qwen (ยืนยันข้ามสถาปัตยกรรม)
- ทดสอบ 3 prompt templates (ต่างจากที่ใช้เลือกวงจร)
- ต้นทุน: ~5 GPU-h

### E7 — (Stretch, ทำถ้าเวลาเหลือ) Path Patching

- ตรวจ direct effect ของ top heads ผ่าน frozen attention เพื่อยืนยันความเป็น "วงจร" แบบเข้มข้นขึ้น — ถ้าทำได้จะยกระดับ discussion มาก แต่**ไม่จำเป็น**ต่อการส่ง Q3–Q4

---

## 4. Metrics และแผนสถิติ (สิ่งที่เค้าโครงเดิมไม่มี — จำเป็นมากสำหรับวารสาร)

| Metric | นิยาม | สถิติที่รายงาน |
|---|---|---|
| Factual Accuracy | greedy argmax == true target | % + 95% CI (bootstrap over facts, 10k resamples) |
| Context-Following (Hallucination) Rate | greedy argmax == distractor target บน misleading suite | เหมือนกัน + **McNemar test** เทียบก่อน/หลัง intervention (paired บน item เดียวกัน) |
| Restoration Effect (E3) | Δ logit-diff หลัง patch | per-head effect + CI; คัดเลือกบน Dev split |
| TruthfulQA MC1/MC2 | ผ่าน lm-eval-harness | % + CI |
| MMLU 0-shot | lm-eval-harness | % + **Capability Retention Ratio** (post/pre) |
| WikiText-2 PPL | lm-eval-harness | ค่าเดียว (deterministic) |

- **Multiple comparisons:** ใช้ Benjamini–Hochberg FDR ตอนทดสอบหลาย heads พร้อมกัน
- **การเลือกตัวเลขทุกตัว (threshold, k, mean-ablation set) ทำบน Dev เท่านั้น** — เขียนกำกับไว้ชัดเจนใน paper
- Decoding เป็น greedy (deterministic) ทั้งงาน จึงไม่ต้องทำซ้ำหลาย seed ยกเว้น random-head baseline ที่สุ่ม 10 ครั้ง

---

## 5. แผน Compute และงบ 5,000 บาท บน GCP (ปรับปรุง)

| ทรัพยากร | สเปก | ราคา | ชั่วโมง/วงเงิน | หมายเหตุ |
|---|---|---|---|---|
| VM หลัก | `g2-standard-4` (L4 24GB) **Spot** | ~$0.35/h (~13 บาท/h) | ~180 h ที่ใช้จริง (จองไว้ 300+) | ทุกโมเดล ≤ 9B bf16 ลง L4 ได้ |
| Boot disk | 100GB Balanced PD | ~350 บาท/เดือน | — | เก็บ code + data; activation ใหญ่เขียนทับได้ |
| สำรอง 9B | `a2-highgpu-1g` (A100) Spot | ~$1.20/h | ~15–20 h | เผื่อ sweep 9B ช้ากว่าคาด |
| **รวมประมาณ** | | | **~3,300–4,200 บาท** | เหลือ buffer ~800–1,700 บาท |

**แนวปฏิบัติกับ Spot instance (สำคัญ):**
- ทุก experiment เขียน result เป็น JSONL ทีละ batch ลง disk (idempotent — รันซ้ำข้ามส่วนที่เสร็จแล้วได้)
- checkpoint ค่า mean สำหรับ ablation และ prompt cache ไว้บน persistent disk
- ใช้ startup script + `gcloud compute instances start` กลับมาได้อัตโนมัติหลังถูก preemption
- ติด tracking ด้วย Weights & Biases (free tier) หรือ local MLflow — วารสารชอบ reproducibility artifact

**ประมาณการใช้ GPU รวม:** ~50–60 ชม. (มี buffer 5 เท่าจากวงเงิน — สบาย)

---

## 6. ความเสี่ยงและแผนสำรอง (Risk Register)

| ความเสี่ยง | โอกาส | แผนสำรอง |
|---|---|---|
| TransformerLens ค่าเพี้ยนเพราะ Gemma-2 softcapping | กลาง | Parity check E0; fallback เป็น raw PyTorch hooks บน HF หรือ `nnsight` · **→ ปิดจบแล้ว (ต.ค. 2026): fp32 parity KL 1.3e-5, argmax 100% — ความเพี้ยนของ bf16 เป็น noise ล้วนๆ** |
| ผล head-level diffuse ไม่คม | กลาง | รายงานที่ระดับ layer/MLP (ยังตอบ RQ1 ได้) + ย้ำ contribution เชิง empirical |
| Intervention ไม่ลด hallucination อย่างมีนัยสำคัญ | กลาง | **Reframe เป็นบทความ characterization:** "When does context beat memory? Competition dynamics in SLMs" — ยังส่ง Q4 ได้สบาย (กำหนด success gate ไว้ล่วงหน้าที่สัปดาห์ 9) |
| TruthfulQA ติด contamination ในโมเดลใหม่ | สูง | ใช้เป็น secondary metric เท่านั้น; ตัวชี้วัดหลักคือ misleading suite ที่สร้างเอง (ตรวจซ้ำไม่ได้จาก pretrain) |
| Spot preemption ทำงานสะเด็ด | สูง | §5 (idempotent scripts) |
| เวลาไม่พอ | กลาง | ลด Qwen/Gemma-9B ออก (cross-model เหลือ Llama 1 ตัวก็ยังตั้ง claim ได้) |
| งาน CM-conflict ระดับ head มีมากขึ้นเร็ว (โดยเฉพาะ JuICE, ICML 2025) | กลาง | Positioning เป็น "systematic study + quantified trade-off บน sub-2B" ไม่ใช่ first; อ่านฉบับเต็ม JuICE ในสัปดาห์ 1 เพื่อ finalize gap wording; ใช้การทดสอบ superposition เป็นสะพานเชื่อมแทนการแข่งหัวกัน |

**Success gates (จุดตัดสิน go/no-go):**
- สิ้นสัปดาห์ 2: parity ผ่าน + baselines ครบ → มิฉะนั้นเปลี่ยน framework/โมเดลหลัก
- สิ้นสัปดาห์ 5: heatmap มีโครงสร้างชัดเจนบนอย่างน้อย 1 corruption regime
- สิ้นสัปดาห์ 9: ผล intervention บน held-out มีทิศทางชัด (ไม่จำเป็นต้อง positive ขนาดใหญ่ — ถ้า null ก็เปลี่ยน narrative เป็น characterization study)

---

## 7. ตารางเวลา 16 สัปดาห์

| สัปดาห์ | งาน Lab | งาน Paper (ทำขนาน) |
|---|---|---|
| 1 | E0 setup + parity | ตั้ง repo + เตรียม LaTeX/Word template ของวารสารเป้าหมาย |
| 2 | E1 data + E2 baselines | เขียน draft ส่วน Methodology ได้เลย (ยังไม่ต้องรอผล) |
| 3–5 | E3 patching sweeps | ต่อ Methodology + เริ่ม Related Work (สืบค้นใหม่ปี 2024–2026) |
| 5–6 | E4 circuit analysis | ทำ Figure 2–3 (heatmap, attention patterns) |
| 7–9 | E5 interventions + evals | ทำ Table 2–3 + Figure 4 (trade-off curve) |
| 9–10 | E6 generalization | เขียน Results |
| 11 | (ถ้ามีเวลา) E7 path patching | เขียน Introduction + Discussion + Limitations |
| 12 | ปิดการทดลองทั้งหมด | เขียน Abstract + Conclusion + ขัดเกลาทั้งฉบับ |
| 13 | — | ตรวจเลขทุกตัวเทียบ log; เพื่อน/อาจารย์อ่านครั้งแรก |
| 14 | — | แก้ตาม feedback; เตรียม cover letter + reproducibility statement |
| 15–16 | — | **ส่ง** + ยก repository ขึ้น GitHub (ล้าง credential, ใส่ README + instructions) |

---

## 8. แผนเขียน Paper

### 8.1 วารสารเป้าหมาย (เรียงตามลำดับการส่ง)

> ⚠️ **Quartile เปลี่ยนทุกปี และบางวารสารถูกถอดจาก Scopus** — ก่อนส่งต้องเช็ค SJR/CiteScore ล่าสุดที่ scimagojr.com และสถานะ indexing ที่วารสาร

| ลำดับ | วารสาร | เหตุผล | ข้อควรระวัง |
|---|---|---|---|
| 1 (เร็วสุด) | **ECTI Transactions on Computer and Information Technology** (ไทย) | Scopus indexed, รอบรีวิวไว, ค่าใช้จ่ายต่ำ, ธีรเมตต้องกับคนไทย | ประวัติมักอยู่ **Q3–Q4** ตาม SJR (ยังไม่ได้ verify สด — เช็ค scimagojr.com ก่อนส่ง) |
| 2 | **Journal of Advances in Information Technology (JAIT)** | Q4, สาย AI รับกว้าง, APC ปานกลาง | เช็ค APC |
| 3 | **International Journal of Computers and Applications** (Taylor & Francis) | Q3–Q4, no-APC แบบ hybrid ถ้ามีสิทธิ์ผ่านสถาบัน | เวลารอรีวิวนานกว่า |
| 4 (ยกระดับขึ้นถ้าผลแข็งแรง) | **Natural Language Engineering** (Cambridge) หรือ **Journal of Experimental & Theoretical Artificial Intelligence** (T&F) | ~Q3, น่าเชื่อถือดีในวงการ | แข่งสูงกว่า; ผล cross-model ต้องครบ |
| ทางเลือก | IEEE Access | เปิดกว้าง รีวิวไว | **APC ~$1,900 เกินงบ 5,000 บาท** (งบนี้เป็นคนละก้อนกับ compute) — หลีกเลี่ยงถ้าไม่มีทุน APC |

**กลยุทธ์:** ส่ง ECTI ก่อนเพื่อการันตีผลงานเร็ว ถ้า reviewer บอก under-scope ค่อยยกระดับขึ้นลำดับ 3–4; **ห้ามส่งวารสารที่ email ชวนเสนอให้ตีพิมพ์โดยตรง (เสี่ยง predatory)**

### 8.2 โครงสร้าง paper พร้อมงบหน้า (ประมาณ 10–14 หน้า สองคอลัมน์)

| ส่วน | ปริมาณ | สาระสำคัญที่ต้องมี |
|---|---|---|
| Abstract | 150–250 คำ | ปัญหา → วิธี (patching + ablation) → ตัวเลขเด่น 2–3 ตัว → artifact |
| 1. Introduction | 1–1.5 หน้า | SLM hallucination เป็นปัญหาจริงในการใช้งาน; **gap (ตรวจสอบ 2 รอบ ณ 30 ก.ย. 2026 — ดู references.md §0 และ §0.5):** มีงาน head-level ของ CM-conflict แล้ว (JuICE ICML 2025 ฯลฯ) **แต่**ยังไม่มี (1) systematic activation-patching localization แบบครบตาราง + heatmap + dev/held-out protocol บน SLMs < 2B หลายสถาปัตยกรรม และ (2) quantified dose–response trade-off ระหว่างการลด hallucination กับความสามารถทั่วไป — เขียนแบบ "เติมช่องว่างเชิงประจักษ์" ไม่ใช่ "เปิดพื้นที่ใหม่"; **contributions เป็น bullet 3 ข้อ** (ดู §8.3) |
| 2. Related Work | 1–1.5 หน้า | (a) Hallucination in LLMs (surveys + การวัด), (b) mechanistic interpretability / activation patching, (c) factual recall circuits (ROME และงานต่อยอด), (d) ความต่างของเรา |
| 3. Methodology | 2.5–3 หน้า | นิยาม notation ของ patching อย่างเป็นทางการ (สำหรับ readers ที่ไม่ใช่สาย interp — วารสาร Q3–Q4 reviewers อาจไม่คุ้น TransformerLens ต้องเขียนให้สอนได้เอง), datasets, dev/held-out protocol, metrics, สถิติ |
| 4. Results | 3–4 หน้า | 4 subsections ตาม RQ1–RQ3 + cross-model |
| 5. Discussion | 0.5–1 หน้า | ตีความ: วงจรความจำ vs บริบทต่างกันอย่างไร, นัยต่อการทำ SLM ให้น่าเชื่อถือ |
| 6. Limitations | ครึ่งหน้า | English เท่านั้น; 4 โมเดล; hallucination แบบวัดได้เท่านั้น; ผลเป็น empirical ไม่ใช่หลักฐานเชิงวงจรสมบูรณ์ (ถ้าไม่ได้ทำ E7) |
| 7. Conclusion | ครึ่งหน้า | — |
| Declarations | — | Code availability (GitHub), ทุน, conflict of interest, ไม่มี human subjects |

### 8.3 ตั้ง contributions ให้ตรงระดับ Q3–Q4 (อย่า overclaim)

1. **Empirical map** ของ head/MLP ที่เกี่ยวข้องเชิงเหตุผล (causally involved) กับการตอบข้อเท็จจริง**ภายใต้บริบทหลอกล่อ (context–memory conflict)** บน SLMs 4 ตัว — พร้อมรายงานความสอดคล้องข้ามโมเดล (rank correlation)
2. **Quantified trade-off** ระหว่างการลด hallucination rate กับความสามารถทั่วไป (MMLU/PPL) ในรูปแบบ dose–response curve ที่ยังไม่มีรายงานแบบเป็นระบบใน SLMs
3. **Methodological protocol** ที่กัน circular analysis (dev/held-out relation split) + artifact ที่ทำซ้ำได้บน GPU สินค้าทั่วไป (L4) ภายในหลักสิบชั่วโมง

หลีกเลี่ยงคำอย่าง "we discovered the hallucination circuit" — ใช้ "we identify components causally involved in…" แทน

**ตารางแยกแยะจากงานที่ใกล้ที่สุด (ต้องอ้างและแยกให้ชัดใน Related Work — รายละเอียดเต็มใน references.md §1):**

| งาน | สิ่งที่เขาทำ | สิ่งที่เราต่าง |
|---|---|---|
| Yu et al. (Findings EMNLP 2024) | โมเดลเล็ก (Llama-2, Pythia, GPT-J); hallucination จาก **ความรู้บกพร่อง** (knowledge-deficit); แก้ด้วยการ "restoration" ของ fact-recall pipeline | เราศึกษา hallucination ที่**เกิดจากบริบทหลอกล่อ** (knowledge มีอยู่แต่ถูก override) — corruption regime ต่างกัน + เพิ่ม trade-off curve + cross-model |
| Li et al. ITI (NeurIPS 2023) | เลือก heads ด้วย **linear probe** (correlational) แล้ว shift activation ตาม "truthful direction" | เราเลือก heads ด้วย **causal patching** และรายงานต้นทุนต่อ capability อย่างเป็นระบบ |
| Du et al. (ACL 2024) | context vs prior knowledge แบบ **behavioral** (persuasion/susceptibility scores) | เราอธิบายกลไก **ภายใน** (ที่ตำแหน่ง head/layer ใดที่การแข่งขันเกิดขึ้น) |
| Chuang et al. Lookback Lens (Findings EMNLP 2024) | ตรวจจับ/ลด contextual hallucination ด้วย **attention-map classifier** + guided decoding | เราเป็น causal analysis ระดับ head + intervention แบบ ablation; ใช้เป็น baseline เชิงพฤติกรรมใน discussion |
| Jiang et al. (arXiv 2026) | head-level patching บน conflict runs แต่เป็น **multimodal** (modality conflict ใน LVLM) | เราเป็น **text-only SLMs** + วัด trade-off |
| Choi et al. (arXiv 2026) | contextual truthfulness ผ่าน "inherited heads" ใน **model lineages** | เราโฟกัสกลไกในโมเดลเดี่ยว + cross-architecture แบบสุ่ม ไม่ใช่สายวงศ์ตระกูล |
| Lamba et al. (arXiv 2025) | ศึกษา hallucination ใน **Gemma** แล้วระบุ activation patching เป็น **future work** | งานของเราคือช่องว่างที่งานนี้เปิดทิ้งไว้ — อ้างเป็นหลักฐานว่า gap ยังมีชีวิต |
| Li, Chen & Tong — JuICE (ICML 2025 Spotlight) | head-level analysis + test-time intervention ต่อ CM conflict; พบ "superposition" (heads มีบทบาททั้ง memory และ context) | **คู่แข่งใกล้ที่สุดในกลุ่ม CM-conflict** — เราต่างที่: systematic patching localization + dev/held-out, dose–response trade-off กับ MMLU/PPL (เขาไม่รายงาน), sub-2B + counterfactual distractor; และ E4 ของเรา**ทดสอบข้ออ้าง superposition** ต่อ |
| Pham et al. (arXiv 2026) | mechanistic ของ **intra-memory** conflict (ในความจำเอง ไม่ใช่บริบท) | ต่างชนิด conflict; ใช้เทียบผล "final layers" และเป็น precedent ของ head vs layer granularity |
| Zhou et al. (IEEE TASLP 2026) | แก้ CM conflict ที่ระดับ **decoding** (attention-map routing) | เราแทรกแซงที่ representation ซึ่งอธิบายกลไกได้ ไม่ใช่แก้ที่ output distribution |
| Sun, Bai & Dredze (ACL 2026) | behavioral — CM conflict ขึ้นกับ task; LLM-as-judge เบ้ภายใต้ conflict | ใช้ใน Discussion: intervention ที่ดีต้อง selective ตาม task ไม่ใช่บังคับฝ่ายเดียว |

### 8.4 Figures & Tables (กำหนดล่วงหน้าให้ทำตามเก็บข้อมูล)

| # | เนื้อหา | มาจาก experiment |
|---|---|---|
| Fig 1 | Pipeline diagram (corruption → patch → ablate → eval) | วาดเอง (W3) |
| Fig 2 | Heatmap layer×position (2 corruption regimes × 4 โมเดล) | E3 |
| Fig 3 | Attention patterns ของ top heads | E4 |
| Fig 4 | Dose–response / trade-off curve (hallucination vs MMLU retention) | E5 |
| Fig 5 | การแข่งขัน logit trajectory (true vs distractor) ตาม layer | E3/E4 |
| Table 1 | Dataset stats + filtering | E1 |
| Table 2 | Baselines ทุกโมเดล | E2 |
| Table 3 | ผล intervention หลัก + CI + McNemar p | E5 |
| Table 4 | Cross-model + template robustness | E6 |

### 8.5 Seed references

➡️ **ย้ายไปรวมไว้ที่ `references.md` แล้ว (ตรวจสอบ citation จากแหล่งจริงแล้วปี 2026)** — ไฟล์นั้นมีรายการ ~40 รายการจัดกลุ่มตามธีม พร้อม models/methods/บทบาทต่องานเรา และรายชื่อที่ยังต้อง verify เพิ่ม (§8 ของไฟล์นั้น)

หมายเหตุสำคัญ: ก่อนเขียน Related Work จริง ให้สืบค้นซ้ำด้วยคำค้นเหล่านี้ เพราะพื้นที่นี้เคลื่อนเร็วมากในปี 2025–2026:
- `"activation patching" hallucination "language model"`
- `"context-memory conflict" attention heads causal`
- `factual recall circuit small language model`
- `attention head ablation truthfulness`

### 8.6 Checklist ก่อนส่ง

- [ ] ตัวเลขทุกตัวใน paper ตรวจสอบย้อนกลับถึง result JSONL ได้ (ทำสคริปต์ `make_tables.py` สร้างตารางจากไฟล์ผลโดยตรง)
- [ ] สถิติครบ: CI, McNemar, BH-FDR ในที่ที่ควรมี
- [ ] Limitations ตรงไปตรงมา (reviewers ชอบ และลดโอกาส reject เรื่อง overclaim)
- [ ] GitHub repo: README + คำสั่งรันซ้ำ + ตาราง hardware runtime; ไม่มี credential; ใส่ license
- [ ] Cover letter ระบุ fit กับ scope วารสาร + ยืนยันไม่เคยตีพิมพ์/ไม่อยู่ระหว่างพิจารณาที่อื่น
- [ ] เช็ค format (template, จำนวนหน้า, reference style) ตาม author guidelines ของวารสารนั้นๆ
- [ ] ตรวจ plagiarism ผ่านเครื่องมือของสถาบัน (Turnitin) ก่อนส่ง

---

## 9. โครงสร้าง Repository ที่วางไว้ตั้งแต่วันแรก

```
mech-interp-hallucination/
├── README.md                  # คำอธิบาย + วิธีรันซ้ำ
├── configs/                   # per-experiment YAML (โมเดล, dataset, พารามิเตอร์)
├── src/
│   ├── data/                  # E1: loader + filtering + misleading templates
│   ├── patching/              # E3: sweep loops (block/head/mlp), corruption regimes
│   ├── analysis/              # E4: heatmap, head ranking, cross-model correlation
│   ├── interventions/         # E5: mean-ablation, knockout, dose-response
│   └── eval/                  # lm-eval wrappers, metric + stats (CI, McNemar, BH)
├── scripts/                   # คำสั่งรันบน GCP + startup script + resume logic
├── results/                   # JSONL ผลรัน (append-only, idempotent)
├── figures/                   # ผลลัพธ์เป็นภาพ
└── paper/                     # LaTeX/Word + make_tables.py
```

---

## 10. Minimum Publishable Unit (MPU) — นิยาม "เสร็จ" ที่ปรับลดได้ตามสถานการณ์

- **ระดับเต็ม (เป้าหมาย):** RQ1–RQ3 ครบ + cross-model 4 ตัว → ส่ง Q3
- **ระดับกลาง:** RQ1–RQ2 + intervention เฉพาะ Gemma-2-2B + Llama 1 ตัว → ส่ง Q4 (ECTI)
- **ระดับต่ำสุด (ถ้า intervention เป็น null):** บทความ characterization ของ competition dynamics (RQ1–RQ2 อย่างเดียว) + heatmap + การวิเคราะห์ → ยังส่ง Q4 ได้
- **ตัดทิ้งได้ก่อนและทีหลัง:** E7 (path patching), Gemma-2-9B, Qwen — ตามลำดับนี้
