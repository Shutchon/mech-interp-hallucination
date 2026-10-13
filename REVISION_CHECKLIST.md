# Revision Checklist — IEICE Major Revision
ตามลำดับความสำคัญที่ reviewer แนะนำ

## Phase 1: ตรวจโค้ดและนิยาม A,B,N (ต้องเปิด VM เพื่อ verify)
- [ ] 1.1 ตรวจสอบว่าโค้ด e4_superposition.py คำนวณ A, B, N ตรงกับสมการที่เขียนใน paper หรือไม่
- [ ] 1.2 ยืนยัน denominator ของ N (clean-neutral หรือ clean-misleading)
- [ ] 1.3 ตรวจว่า N/A เป็น mean of ratios หรือ ratio of means

## Phase 2: แก้ความไม่สอดคล้องของตัวเลขและข้อความ
- [ ] 2.1 "sub-2B" → "small language models" (Summary + Intro)
- [ ] 2.2 ระบุจำนวน facts แยกตามโมเดลตั้งแต่ Summary
- [ ] 2.3 แยก "specialist" เฉพาะ Gemma จาก "top-1" ของ Llama/Qwen
- [ ] 2.4 อธิบาย MMLU 49.4% (lm-eval full) vs 50.0% (subset) — protocol
- [ ] 2.5 แก้ "weaker and stronger" → ทั้งคู่ต่ำกว่า Gemma แต่ต่างกัน
- [ ] 2.6 แก้ "top-5" layers → ระบุจำนวนจริง (6, 3, 6) หรือใช้ "influential layers"
- [ ] 2.7 แก้ Table 4 caption (ลบ "Δacc." หรือเพิ่มคอลัมน์)
- [ ] 2.8 แทน "E4" → "Stage 4" ให้สอดคล้องกับ Methods
- [ ] 2.9 "reproduces every headline finding" → ระบุเฉพาะผลที่ทำซ้ำจริง

## Phase 3: ทำสมการ A, B, N ให้ชัดเจน reproducible
- [ ] 3.1 เพิ่มสมการแยกสำหรับ A, B, N พร้อม baseline ที่ชัดเจน
- [ ] 3.2 ระบุ denominator ของ N
- [ ] 3.3 ระบุวิธีจัดการ denominator ใกล้ศูนย์
- [ ] 3.4 นิยาม superposition เชิงปฏิบัติการ + กรณี A=B=0
- [ ] 3.5 อธิบายว่า N/A เป็น mean of per-fact ratios หรือ ratio of means

## Phase 4: จำกัดข้ออ้าง attention-sink
- [ ] 4.1 ลด "value pathway" เป็นสมมติฐาน
- [ ] 4.2 แก้ "causally inert" → "minimal effect under the tested neutral-patching control"
- [ ] 4.3 แก้ "does nothing" → จำกัดขอบเขต
- [ ] 4.4 เพิ่ม BOS attention % จากหลายตัวอย่าง (ต้องรู้จาก data) หรือ soft-claim
- [ ] 4.5 Softening "nearly free for deployment QA settings"

## Phase 5: ความแม่นยำของผล hallucination reduction
- [ ] 5.1 ระบุว่า 30-50% เฉพาะ T0/T2; T1 ลดได้แค่ ~9% relative
- [ ] 5.2 เพิ่ม before→after ครบสำหรับ T1
- [ ] 5.3 เพิ่มบทอธิบายว่า "other responses" = คำตอบที่ไม่ใช่ true หรือ distractor
- [ ] 5.4 ใช้ "misleading-context distractor-following" แทน "hallucination" ในจุดที่เหมาะ

## Phase 6: เพิ่มหลักฐานสถิติ
- [ ] 6.1 เพิ่ม CI ใน Table 4 (dose-response)
- [ ] 6.2 แก้ p≈0 → p < 10⁻²³ (McNemar exact)
- [ ] 6.3 ระบุ n จริงในแต่ละการทดลอง
- [ ] 6.4 ระบุ MMLU เป็น "subset score changed by X" ไม่ใช่ "preserved"
- [ ] 6.5 Random baseline: รายงาน range ของทั้ง 5 seeds

## Phase 7: Reproducibility details
- [ ] 7.1 เพิ่ม templates T0/T1/T2 + neutral แบบเต็ม
- [ ] 7.2 ระบุ checkpoint IDs + software versions
- [ ] 7.3 อธิบายสูตร mean-ablation vector
- [ ] 7.4 เพิ่ม artifact link หรือแก้ "We release" ให้ตรงสถานะ
- [ ] 7.5 แก้ parity claim: ระบุว่าเป็น fp32 check; bf16 คือ noise floor

## Phase 8: แยก novelty จาก JuICE
- [ ] 8.1 เพิ่ม comparison table/passage: สิ่งที่เพิ่มจาก JuICE
- [ ] 8.2 ลด "nearly free" → "low measured cost on knowledge tasks"

## Phase 9: แก้ references
- [ ] 9.1 [8] Lookback Lens: EMNLP 2024 main (ไม่ใช่ Findings)
- [ ] 9.2 [12] Copy Suppression: BlackboxNLP 2024 (ไม่ใช่ NeurIPS 2023)
- [ ] 9.3 [16] Editing: แก้รายชื่อผู้แต่งให้ถูก
- [ ] 9.4 [19] Campregher: แก้ชื่อเรื่อง/ผู้แต่ง/TMLR 2025
- [ ] 9.5 [17] Nature: เพิ่ม vol/DOI
- [ ] 9.6 เพิ่ม citation สำหรับ [20]-[24] ที่ไม่ได้อ้าง หรือลบ
- [ ] 9.7 เพิ่ม MMLU และ WikiText-2 sources
- [ ] 9.8 เพิ่ม page/volume ที่ขาด

## Phase 10: รูปแบบ
- [ ] 10.1 ขยายตัวอักษรในกราฟให้ ≥7pt (ต้อง regenerate ทุกรูป)
- [ ] 10.2 Fig 3: เพิ่ม colorbar + ชื่อแกน
- [ ] 10.3 Fig 4: แก้ color scale ให้ครอบ N/A ติดลบ
- [ ] 10.4 Fig 5(b): เพิ่ม spec1
- [ ] 10.5 Keywords: ลดเหลือ 4-5 คำ
- [ ] 10.6 Summary: ขยายเป็น ~300 คำ
- [ ] 10.7 แก้ received/revised dates
- [ ] 10.8 ลบ/แก้ "??"
