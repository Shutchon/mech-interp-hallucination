# GCP Setup Runbook — งบ 5,000 บาท (~$145)

> สำหรับโปรเจกต์ *Locating and Mitigating Factual Hallucination Circuits in SLMs*
> วัตถุประสงค์: ตั้ง L4 Spot VM ให้พร้อมรัน E0–E6 ภายใน 1 วันทำงาน พร้อมวินัยการใช้เงินที่ทำให้ไม่ทะลุงบ

---

## 0. ภาพรวมสถาปัตยกรรมและตัวเลขเงิน

```
[Mac ของคุณ] --SSH/IAP--> [Spot VM: g2-standard-4 + L4 24GB, us-central1 (อเมริกา)]
                                |-- boot disk 100GB pd-balanced (code + HF cache + results)
                                |-- (สำรอง) snapshot รายสัปดาห์ + rsync ผลขึ้น GCS
                                └-- (ฉุกเฉินเท่านั้น) A100 Spot แยกต่างหาก
```

**ประมาณการใช้จริงตลอดโปรเจกต์ (16 สัปดาห์):**

| รายการ | ประมาณการ | หมายเหตุ |
|---|---|---|
| L4 Spot ~70 GPU-ชม. ที่ใช้จริง | ~$21 (ราคา spot ที่ us-central1 แกว่ง $0.22–0.35/ชม.) | E0–E6 รวมกันตามแผน §5 |
| Boot disk 100GB × 2 เดือน | ~$20 | จ่ายต่อเนื่องแม้ stop VM |
| External IPv4 (ตัดได้ด้วย IAP) | $0–8 | ~$0.005/ชม. |
| Snapshot + GCS | ~$3 | incremental, ราคาถูกมาก |
| **รวม** | **~$45–55** | **เหลือกันชน ~3 เท่าจากงบ $145** — กันชนนี้พอรองรับ A100 สำรอง 15–20 ชม. ได้สบาย |

> หมายเหตุ: free trial $300/90 วันของบัญชีใหม่ **ปกติไม่ครอบคลุม GPU** — อย่านับมันเข้างบ จนกว่าจะอ่านเงื่อนไขล่าสุดยืนยันได้

---

## 1. วันที่ 0 — Project, Billing, Quota (ใช้เวลา ~30 นาที + รอ quota 1–2 วัน)

1. **สร้าง project ใหม่แยก** เพื่อ cleanup ง่ายตอนจบ: `mech-interp-paper`
   ```bash
   gcloud projects create mech-interp-paper
   gcloud config set project mech-interp-paper
   gcloud services enable compute.googleapis.com
   ```
2. **ผูกบัญชี billing** แล้วตั้ง **Budget + Alerts**: Console → Billing → Budgets & alerts → วงเงิน **$145**, alert ที่ 50 / 80 / 100%
   > ⚠️ Budget alert เป็นแค่อีเมลแจ้ง — **ไม่หยุดทรัพยากร** ต้องพึ่งวินัยใน §5
3. **ขอ GPU quota ให้ครบสองตัว (สิ่งที่คนข้ามบ่อยที่สุด — project ใหม่เริ่มที่ 0 ทั้งคู่ และต้องผ่านทั้งคู่):**
   - **ตัวที่ 1: `GPUS_ALL_REGIONS`** — โควตารวมทุก region ทุกชนิดการ์ด · ไม่ผ่านตัวนี้จะเจอ error `Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally` ตอนสร้าง VM (และคำแนะนำ "try another zone" ใน error ช่วยไม่ได้ เพราะเป็น global)
   - **ตัวที่ 2: `NVIDIA_L4_GPUS`** — โควตารายการ์ดราย region → ต้องเลือก `us-central1`
   - วิธียื่น (ทั้งสองตัวเหมือนกัน): Console → IAM & Admin → Quotas → กรองชื่อ quota → ติ๊ก → **Edit Quotas** → ตั้ง new limit = **1** → กรอกเหตุผล
   - เหตุผลตัวอย่าง: "Academic research on model interpretability; single L4 for batch experiments, a few hours per day"
   - ปกติอนุมัติ 1 การ์ดเร็ว (ไม่กี่นาทีถึง 1–2 วัน) ไม่ต้องมีเอกสาร
   - เช็คสถานะยืนยัน:
     ```bash
     gcloud compute project-info describe \
       --format="table(quotas.metric,quotas.limit,quotas.usage)" | grep -E "GPUS_ALL_REGIONS|L4"
     ```
4. **ตั้งค่า default zone ที่ใช้บ่อย:**
   ```bash
   gcloud config set compute/zone us-central1-a
   ```

**เลือก region:** ตั้งค่านี้ใช้ **`us-central1` (อเมริกา)** เพราะราคา spot ถูกที่สุดและ spot availability ของ L4 ดีที่สุด — ทางเลือกถ้าอยากใกล้บ้านมีสองอัน:

- **`asia-southeast1` (สิงคโปร์)** — ยืนยันแล้วว่ามีทั้ง G2 (L4) และ A2 (A100) ใน zone `-a`, latency จากไทย ~20–40ms (เทียบ ~200ms จากอเมริกา), timezone UTC+7 เดียวกับไทย — แพงกว่าราว $0.05–0.10/ชม. รวมทั้งโปรเจกต์**ต่างกันแค่ ~$5–8** ถ้าวันหน้ารู้สึกว่า SSH/VS Code สะดุดเกินไป ย้ายกลับมาได้
- **region กรุงเทพฯ** (เปิดแล้วปี 2026 ต่อจากแผนลงทุน $1B ปี 2024) — ยังไม่แนะนำ: region ใหม่มักยังไม่มี machine family ติด GPU (G2/L4) และ spot capacity บางกว่า ตรวจสถานะจริงได้ที่ [GPU regions & zones](https://docs.cloud.google.com/compute/docs/regions-zones/gpu-regions-zones)

> การย้าย region ภายหลังง่าย เพราะ code อยู่บน GitHub และผลทดลอง sync ไว้บน GCS แล้ว (§6) — แค่สร้าง VM ใหม่ + rsync กลับ · ข้อควรระวังเดียวของอเมริกา: timezone ต่างกัน 12 ชม. ให้ตั้ง log/cron เป็น UTC เสมอเพื่อไม่สับสนตอนอ่านผล

---

## 2. สร้าง Spot VM (ครั้งเดียว ใช้ทั้งโปรเจกต์)

หา image ล่าสุดของ Deep Learning VM (มี CUDA driver + PyTorch ติดมาเลย ประหยัดเวลา setup ที่สุด) — ชื่อ image เป็นตัวพิมพ์เล็ก ต้อง `grep -i`:

```bash
gcloud compute images list --project=ml-images --no-standard-images \
  --format="value(family)" | grep -i pytorch | grep -v release | sort -uV
```

ได้รายการ family เรียงตามเวอร์ชัน — **เลือกแถวล่างสุด** · รูปแบบชื่อปี 2026 ที่เจอจริง: `pytorch-2-9-cu129-ubuntu-2404-nvidia-580` = PyTorch 2.9 + CUDA 12.9 + Ubuntu 24.04 + NVIDIA driver 580 (รุ่นเก่ากว่าอาจเป็น `-debian-12` — ใช้ได้เหมือนกัน) · เวลาสร้าง VM ใช้ชื่อ **family** แล้ว GCP จะเลือก build ล่าสุดใน family ให้เอง · ตรวจเวอร์ชันจริงหลังสร้าง VM ด้วย `python3 -c "import torch; print(torch.__version__, torch.version.cuda)"` และ `nvidia-smi` · ดูตารางเต็ม (พร้อมคอลัมน์ DEPRECATED ให้เลี่ยง) ด้วย `gcloud compute images list --project=ml-images --no-standard-images`

สร้าง VM:

```bash
gcloud compute instances create mech-interp-l4 \
  --zone=us-central1-a \
  --machine-type=g2-standard-4 \
  --accelerator=count=1,type=nvidia-l4 \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --image-family=pytorch-2-9-cu129-ubuntu-2404-nvidia-580 \
  --image-project=ml-images
```

> ⚠️ บรรทัด `--image-family=...` ให้**แทนที่ด้วยชื่อ family ล่าสุดจากคำสั่ง list ด้านบน** — ใส่ชื่อตรงๆ อย่าใส่วงเล็บ `< >` ติดมาด้วย (shell จะตีความเป็นการ redirect ไฟล์แล้วคำสั่งพังทันที) และห้ามเติม comment ท้ายบรรทัดที่มี `\` คั่น

หมายเหตุ:
- `--instance-termination-action=STOP` = เมื่อถูก preempt หรือครบอายุ 24 ชม. VM จะหยุดแบบ**เก็บ disk ไว้** → ตื่นมาแค่ `start` ใหม่ ข้อมูลไม่หาย
- ถ้า error ว่า accelerator ซ้ำกับ machine type (นโยบาย image/API เปลี่ยนได้) ให้ตัด `--accelerator` ออก — g2-standard-4 มี L4 1 การ์ดมากับตัวเครื่องอยู่แล้ว
- **อายุสูงสุดของ Spot = 24 ชม.ต่อรอบ** และถูก preempt ได้โดยเตือนล่วงหน้า 30 วินาที — ทุก sweep ต้องเขียนผลเป็น JSONL ทีละ batch (ตามแผน §5)

---

## 3. เชื่อมต่อ + เครื่องมือประจำวัน

```bash
gcloud compute ssh mech-interp-l4 --zone=us-central1-a
```

**ทางเลือกประหยัด (ตัดค่า external IPv4 ~$7–8/เดือน และปลอดภัยกว่า):** สร้าง VM ด้วย `--no-address` แล้ว SSH ผ่าน IAP tunnel:

```bash
# ครั้งเดียว: เปิด firewall ให้ IAP
gcloud compute firewall-rules create allow-iap-ssh \
  --network=default --direction=INGRESS --action=ALLOW \
  --rules=tcp:22 --source-ranges=35.235.240.0/20
gcloud services enable iap.googleapis.com

# ต่อทุกครั้ง (ใส่ใน ~/.zshrc ก็ได้)
gcloud compute ssh mech-interp-l4 --zone=us-central1-a --tunnel-through-iap
```

บนเครื่องคุณ: ใช้ **VS Code Remote-SSH** ชี้ที่ host เดียวกัน (ใน `~/.ssh/config` ให้ใส่ `ProxyCommand` สำหรับ IAP ถ้าใช้วิธีนั้น) + รันงานยาวใน **tmux** เสมอ เพราะ SSH ตัดแล้ว process ต้องรันต่อ

---

## 4. Setup ครั้งแรกบน VM (~20 นาที)

```bash
# 1) เช็ค GPU + driver
nvidia-smi

# 2) ตั้ง HF cache ให้อยู่บน persistent disk (โหลดโมเดลครั้งเดียว ใช้ได้ทุก restart)
echo 'export HF_HOME=$HOME/hf_cache' >> ~/.bashrc
source ~/.bashrc

# 3) Token สำหรับ Gemma (gated model — ต้องกด accept license ที่หน้า HF ก่อน!)
echo 'export HF_TOKEN=hf_xxx' >> ~/.bashrc   # ห้าม commit ไฟล์นี้เด็ดขาด

# 4) Environment
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install transformer_lens circuitsvis plotly pandas lm-eval datasets accelerate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

4. **Environment**
   ```bash
   source .venv/bin/activate   # ทุกหน้าต่าง shell/tmux ใหม่ต้องเปิดใหม่ทุกครั้ง
   pip install -r requirements.txt
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
5. **เพิ่ม swap 24GB (สำคัญ — RAM ของ g2-standard-4 มีแค่ 16GB):** การโหลดโมเดล fp32 (เช่น `--dtype float32` ของ E0) ต้องถือ weight สองชุดพร้อมกันช่วงแปลง (~21GB สำหรับ 2B) → ถ้าไม่มี swap จะโดน OOM "Killed" แบบไม่มี traceback
   ```bash
   sudo fallocate -l 24G /swapfile && sudo chmod 600 /swapfile
   sudo mkswap /swapfile && sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab   # คงที่หลัง restart
   free -h   # ต้องเห็น Swap 24Gi
   ```

จากนั้น clone repo ของโปรเจกต์แล้วรัน **E0 parity check ทันที** ก่อนทำอย่างอื่น (gate แรกของทั้งโปรเจกต์ — เกณฑ์ KL < 1e-3 และ argmax ตรง ≥ 99% ตามแผนหลัก)

---

## 5. วินัยการใช้เงิน (สิ่งที่คนจ่ายเกินงบพลาดจริง)

1. **หยุด VM ทุกครั้งที่เลิกงาน** — ค่าใช้จ่าย compute หยุดทันที เหลือเสียแค่ disk (~$10/เดือน):
   ```bash
   gcloud compute instances stop mech-interp-l4   # เลิกงาน
   gcloud compute instances start mech-interp-l4  # เช้ามารันต่อ ข้อมูลอยู่ครบ
   ```
2. **อย่าทิ้ง VM ค้างคืนเพื่อ "คิด"** — ถ้าไม่ได้รัน sweep ให้ stop
3. **(ทางเลือก) Watchdog ปิดเครื่องเองเมื่อ GPU ว่างเกิน 90 นาที** — กันเผลอลืม:
   ```bash
   # บันทึกเป็น /usr/local/bin/gpu-idle-watchdog.sh แล้วเพิ่มใน crontab -e: @reboot
   # ⚠️ เฉพาะช่วง sweep เท่านั้น — ช่วง data prep (CPU-only) ให้ปิด ไม่งั้นโดนไล่ตาย
   #!/bin/bash
   LIMIT=90; IDLE=0
   while true; do
     UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
     if [ "${UTIL:-100}" -lt 5 ]; then IDLE=$((IDLE+2)); else IDLE=0; fi
     [ "$IDLE" -ge "$LIMIT" ] && sudo shutdown -h now
     sleep 120
   done
   ```
4. **ตรวจยอดทุกสัปดาห์:** Billing → Reports — ถ้า Spot ราคาพุ่งผิดปกติ พิจารณาย้าย zone

---

## 6. สำรองข้อมูล (กันเสียผลการทดลองจาก spot/preemption)

**ชั้นที่ 1 — Code:** push ขึ้น GitHub ทุกวันที่แก้ (ถือเป็น source of truth) — ตั้งค่าครั้งเดียวบน VM ด้วย SSH key:

```bash
# สร้าง key (Enter ผ่านได้ทั้งหมด)
ssh-keygen -t ed25519 -C "mech-interp-vm" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub        # คัดลอกไปใส่ที่ GitHub → Settings → SSH and GPG keys → New SSH key
ssh -T git@github.com            # ทดสอบ: ต้องได้ "Hi <username>! ..."
git clone git@github.com:<user>/mech-interp-hallucination.git   # URL แบบ git@ ไม่ใช่ https
```

- วงจรประจำวัน: `git add -A && git commit -m "..." && git push` — **ทำทุกครั้งที่จบ experiment** เพราะ spot ถูกดับได้เตือนล่วงหน้าแค่ 30 วินาที
- ⚠️ ถ้า `git clone` เจอ `Repository not found` = ยังไม่ได้สร้าง repo บน GitHub (เช็ค username ที่ถูกต้องด้วย `ssh -T git@github.com` แล้วดูว่าทักว่า "Hi ใคร") — สร้างได้ที่ [github.com/new](https://github.com/new) ชื่อ `mech-interp-hallucination` แบบ Private · ถ้าเจอ `Permission denied (publickey)` = เครื่องนั้นยังไม่มี key ที่ลงทะเบียนกับ GitHub — **key ต้องทำแยกต่อเครื่อง** (VM หนึ่งอัน, Mac หนึ่งอัน, เพิ่มได้หลายอันในบัญชีเดียว) · ทางลัดบน Mac: `brew install gh && gh auth login` แล้วเลือก SSH — สร้างและอัป key ให้อัตโนมัติ
- ห้าม push: `HF_TOKEN` ทุกรูปแบบ, `hf_cache/` (โมเดล 5–18GB), ผล raw ใหญ่ (`*.pt`, `results/raw/` — อันนั้นไป GCS ชั้นที่ 3), `.venv/`, `__pycache__/`
- `.gitignore` ขั้นต่ำ: `.venv/`, `__pycache__/`, `*.pyc`, `hf_cache/`, `.env`, `results/raw/`, `*.pt`, `*.bin`
- ทางเลือก: ติดตั้ง GitHub CLI (`gh auth login` — เปิด browser บน Mac ใส่โค้ด) แล้ว push ผ่าน HTTPS ได้ ไม่ต้องจัดการ key เอง
- **push ผลสรุปจาก VM ด้วย (ทุกครั้งที่จบ experiment):** ผลสรุปขนาดเล็ก (`results/**/*.json`/`.jsonl` ยกเว้น `results/raw/` ที่ถูก .gitignore กันไว้) ออกแบบให้เข้า git — VM push ขึ้นได้เองเพราะมี SSH key ของตัวเอง:
  ```bash
  git add results/
  git commit -m "<ชื่อ experiment + ผลสรุปหนึ่งบรรทัด>"
  git push
  ```
- **กฎสองเครื่อง:** Mac (แก้โค้ด/เอกสาร) กับ VM (รันการทดลอง) push ลง repo เดียวกัน — เปิดงานที่เครื่องไหนให้ `git pull` ก่อนทุกครั้ง และ push ฝั่งโค้ดก่อนฝั่งผลลัพธ์เสมอ เพื่อให้ทุก report เกิดจากโค้ดรุ่นล่าสุด

**ชั้นที่ 2 — Snapshot รายสัปดาห์** (รวม environment ทั้งเครื่อง, incremental ถูกมาก):
```bash
gcloud compute disks snapshot mech-interp-l4 --zone=us-central1-a \
  --snapshot-names=weekly-$(date +%Y%m%d)
```

**ชั้นที่ 3 — Results ขึ้น GCS** (กัน disk พัง/ลบผิด):
```bash
gcloud storage buckets create gs://mech-interp-results --location=us-central1
gcloud storage rsync -r ~/mech-interp/results/ gs://mech-interp-results/results/
```

---

## 7. แผนสำรอง A100 (สร้างเฉพาะเมื่อจำเป็น — อย่าสร้างไว้ก่อน)

ใช้เมื่อ: sweep Gemma-2-9B ช้าเกินแผนจน L4 ไม่ทัน milestone

```bash
gcloud compute instances create mech-interp-a100 \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --accelerator=count=1,type=nvidia-tesla-a100 \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --boot-disk-size=100GB --boot-disk-type=pd-balanced \
  --image-family=<PYTORCH_FAMILY> --image-project=ml-images
```

- ราคา spot ~$0.80–1.20/ชม. → วางแผนไม่เกิน 15–20 ชม. (ใช้กันชนของงบ)
- ใช้เสร็จ **ลบทันที** พร้อม disk — อย่าค้างคืน
- ควรย้ายผลด้วย rsync กลับ bucket ก่อนลบ

---

## 8. Checklist ตอนจบโปรเจกต์ (กันถูกเก็บเงินหลังส่ง paper)

```bash
# ดึงข้อมูลสำคัญออกก่อน (results อยู่บน GCS แล้ว + code บน GitHub)
gcloud compute instances delete mech-interp-l4     # ⚠️ จะลบ disk ด้วย — ตรวจ snapshot ก่อน!
# เก็บ snapshot สุดท้ายไว้ 6 เดือน (กัน reviewer ขอ artifact) แล้วค่อยลบ
# gcloud compute snapshots delete weekly-xxxx
# gcloud storage rm -r gs://mech-interp-results    # เมื่อไม่จำเป็น
```

> ค่า disk+bucket หลังจบงาน ~$3–5/เดือน — ถ้าอยากอุดหนุน artifact ให้ reviewer ใช้ได้ แต่ให้รู้ว่ามีเงินออกต่อ

---

## 9. ภาคผนวก — Accept Gemma license + สร้าง HF_TOKEN (ทำบนเครื่องตัวเองก่อนขึ้น VM)

> โมเดล gated จะโหลดไม่ได้เลยถ้าไม่ทำ step นี้ — ใช้เวลารวม ~10 นาที และทำก่อนสร้าง VM ได้

**ขั้นที่ 1 — Accept license (ทำทีละ repo เพราะเป็นสิทธิ์แบบ per-repo):**

| หน้าโมเดล | สถานะ | ต้องทำไหม |
|---|---|---|
| [google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b) | gated | ✅ ตัวหลัก |
| [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) | gated | ✅ เผื่อใช้ instruct เทียบใน E2 |
| [google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b) | gated | ✅ coarse sweep |
| [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) | gated | เผื่อ E0 ทดสอบรุ่นใหม่ — ตระกูล 3 ขนาด 1B มีบน HF เฉพาะแบบ instruction-tuned (ไม่มีตัว base) |
| [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | gated (ฟอร์มของ Meta แยกต่างหาก) | ✅ |
| Qwen/Qwen2.5-1.5B | เปิด | ไม่ต้อง |

> หมายเหตุ: **Gemma 4** (เปิดตัว เม.ย. 2026) ใช้ license **Apache 2.0** — ไม่ gated ไม่ต้อง accept ก็โหลดได้ แต่ยังไม่อยู่ในลิสต์โมเดลของโปรเจกต์ (รุ่นเล็กสุดเป็นสถาปัตยกรรมแนว edge ที่ tooling mech interp ยังไม่รองรับ — ดู references.md [T1c])

1. Login บัญชี HF (บัญชีใหม่ต้อง**ยืนยันอีเมลก่อน** ไม่งั้นกด accept ไม่ได้; และถ้าไม่ login ฟอร์มจะไม่โผล่ให้เห็นเลย)
2. เข้าหน้าโมเดลแต่ละตัว → แถบ gated อยู่บนสุดของเนื้อหา (เหนือ model card): เปิดลิงก์ Terms of Use → ติ๊ก checkbox ทุกอัน (ยินยอมแบ่งปันข้อมูลติดต่อ + ยอมรับเงื่อนไข) → กด **Accept** · ฝั่ง Meta จะมีฟอร์มเพิ่ม: เลือก Country + ประเภทผู้ใช้ + checkbox license
3. Gemma และ Llama **อนุมัติอัตโนมัติทันที** — รีเฟรชแล้วเข้าแท็บ "Files and versions" ได้ = ผ่าน · ⚠️ ต้อง accept ด้วย**บัญชีเดียวกับที่ใช้สร้าง token**

**ขั้นที่ 2 — สร้าง token:**

1. [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → **Create new token** → แบบ **Fine-grained**
2. ตั้งชื่อ (เช่น `mech-interp-vm`) + สิทธิ์แค่ **Read** — พอสำหรับโหลด gated repo ที่ accept แล้ว
3. คัดลอก token (`hf_...`) ทันที — หน้านี้โชว์ครั้งเดียว
4. บน VM: `echo 'export HF_TOKEN=hf_xxx' >> ~/.bashrc && source ~/.bashrc`
5. ⚠️ ห้าม commit token ลง repo — ถ้ารั่วให้กด Revoke ที่หน้าเดิม

**ตรวจว่าใช้ได้ (รันบน Mac ก่อนได้):**

```bash
python3 -m pip install -U "huggingface_hub[cli]"
export HF_TOKEN=hf_xxxxxxxxxxxx

# ไล่ตรวจทุก repo ที่โปรเจกต์ใช้ — ตัวไหน FAIL = ยังไม่ได้ accept หน้านั้น
for repo in google/gemma-2-2b google/gemma-2-2b-it google/gemma-2-9b google/gemma-3-1b-it meta-llama/Llama-3.2-1B; do
  python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('$repo','config.json')" \
    && echo "OK: $repo" || echo "ยังไม่ผ่าน: $repo"
done
```

---

## 10. ภาคผนวก — เหตุการณ์จริงที่เจอและวิธีแก้ (Incident Log)

### 10.1 Spot preemption + kernel อัปเดตเอง ทำให้ GPU ใช้ไม่ได้ (10 ต.ค. 2026)

**อาการ:** VM ถูก preemption (สถานะ TERMINATED) → start ใหม่แล้ว `nvidia-smi` ขึ้น "couldn't communicate with the NVIDIA driver" + `modprobe nvidia` บอก `Module nvidia not found in directory /lib/modules/7.0.0-1014-gcp`

**สาเหตุ:** unattended-upgrades ติดตั้ง kernel ใหม่ (7.0.0-1014) ตอนเครื่องรันอยู่ แต่โมดูล driver ของ image นี้เป็นแบบ precompile ผูกกับ kernel เดิม (7.0.0-1011) พอ reboot เข้า kernel ใหม่จึงไม่มี driver

**วิธีแก้ (ใช้ได้จริง):**
1. เช็คว่า kernel เก่ายังอยู่: `ls /lib/modules/` (ต้องเห็น 7.0.0-1011-gcp ที่มีโมดูล nvidia ข้างใน)
2. `sudo sed -i 's|^   set default="0"|   set default="1>2"|' /boot/grub/grub.cfg` — ฝังตำแหน่ง kernel เก่าตรงๆ (ตำแหน่ง = entry หลัก 1 คือ Advanced options, ตัวที่ 2 ข้างใน) ⚠️ ใช้วิธีนี้เพราะทั้ง `GRUB_DEFAULT=saved` + `grub-set-default` และ literal path ล้วนถูก grub-mkconfig ของ image นี้เมิน
3. `sudo reboot` แล้วตรวจ: `uname -r` (ต้องเป็น kernel เก่า) + `nvidia-smi`
4. **กันซ้ำ:** `sudo apt-mark hold linux-image-gcp linux-headers-gcp linux-gcp linux-image-7.0.0-1014-gcp linux-headers-7.0.0-1014-gcp linux-modules-7.0.0-1014-gcp`
5. ห้ามรัน `sudo update-grub` หลังจากนี้ (มันจะล้างการแก้ในข้อ 2 — ถ้าจำเป็นต้องรัน ให้แก้ default กลับด้วย sed อีกครั้ง)

**บทเรียน:** ทุกสคริปต์ experiment ต้องมี resume ระดับ config/record (E3, E5 ทำแล้ว) — preemption ครั้งนี้ E5 ตายตั้งแต่ config แรกเพราะเพิ่งจะเติม resume ไม่ทัน

### 10.2 SSH key มี passphrase (30 ก.ย. 2026)

อาการ `Server accepts key` แล้ว `Permission denied` — key ถูก passphrase ป้องและ agent ไม่มี identity · แก้ด้วย `ssh-add --apple-use-keychain ~/.ssh/google_compute_engine` (ครั้งเดียว) · ถ้า Mac รีสตาร์ตแล้ว `ssh-add -l` ว่าง — รันคำสั่งเดิมซ้ำได้เลย ไม่ต้องพิมพ์รหัส (Keychain จำให้)

**ถ้าเจอ error ตอนโหลดโมเดล:**
- `401 Unauthorized` → token ผิด/หมดอายุ หรือ env var ไม่ถูก set (`echo $HF_TOKEN` เช็ค)
- `403` / ข้อความเรื่อง gated repo → **license ของ repo นั้นยังไม่ได้ accept** (เป็นราย repo!) หรือ accept ด้วยบัญชีอื่นกับเจ้าของ token
- `404 Repository not found` ทั้งที่ repo มีจริง → อาการของ "ไม่ได้ login" สำหรับ gated repo — เช็ค token ก่อนสรุปว่าพิมพ์ชื่อผิด
- ตรวจ token ด้วย `hf auth whoami` (เวอร์ชันเก่า: `huggingface-cli whoami`)
