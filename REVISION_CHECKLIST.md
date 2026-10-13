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

---

# Round 2 (13 ต.ค. 2026) — ตามรีวิวครั้งที่ 2 (Review2.md)

## แก้เรียบร้อย (local)
| # | รายการ | วิธีแก้ |
|---|--------|---------|
| R1 | JuICE mischaracterized | เขียนใหม่: เน้น neutral-control, specialist เดี่ยว, MMLU/PPL trade-off; เพิ่ม PH3 (Jin et al. Findings ACL 2024) bibitem + ความเชื่อมกับ top-k knockout |
| R2 | McNemar p ผิด + b,c ซ้ำ | exact binomial ทุก config (spec1=3.4e-23), ตัดซ้ำ, Summary ระบุ outcome; เพิ่ม accuracy-side test (b=81,c=18,p=1.0e-10) และ neutral-side (p=9.1e-4) |
| R3 | 35% ≈ 34.7% | ยืนยันถูกแล้ว ไม่แก้ |
| R4 | N/A ตีความผิด | วิเคราะห์ raw ΔLD จริง: L25.H4 rawN/rawA ≈ 2% (ค่า 76% เกิดจากตัวหารต่างกัน), denominator health (median 0.25, 35% <0.5, 5.3% zeroed), sensitivity ทุก threshold — specialist N≈0 คงทน; เขียน passage ใหม่ทั้งหมด |
| R5 | superposition สับสน | เปลี่ยนชื่อเป็น directional asymmetry index (low S = strong superposition), "confirming"→"consistent with", แก่ "strongly superposed heads" → ตามทิศ index |
| R10 | same held-out split | ระบุ relation-disjoint แต่ไม่ item-identical (456/416 vs 1,193), baseline ต่างจาก composition ด้วย, "distributed circuit" อ่อนลง |
| R12 | Fig.4 dominates + ค่าหาย | caption ใหม่ (highest accuracy, no Pareto dominance), เพิ่ม top-10/mean2/meanspec3 ลง Table 4 |
| R13 | ถ้อยคำ 6 จุด | sub-2B→"of this size class", collateral damage→limited collateral cost, zero neutral role→minimal..., PPL +5% ตรง ๆ, 30-50% +template qualifier, Table1/4 note ใน caption |
| R14 | references | campregher title จริง, arXiv:2408.00118/2407.21783/2412.15115, DOI 3 IEICE, ji2023survey ผู้แต่งเต็ม (verified 10.1145/3571730), llama32card เพิ่ม |
| R16a | "??" หลังชื่อ | \authorentry {Member}→{m} (รหัสสถานะต้องเป็นตัวอักษร) |
| R8(บางส่วน) | uncertainty | fact-clustered CI ใน Table 4 + ในเนื้อหา, random spread 5 seeds, CI 46.0-48.5 ระบุ metric, Methods ระบุ clustered+exact binomial, MMLU "indistinguishable"→descriptive |
| R15(บางส่วน) | ฟอนต์รูป | fig2/4/5 สร้างใหม่ที่ขนาดพิมพ์ (3.39in) ฟอนต์ 7.5-8pt, fig5 colorbar + spec1 star + สเกลสีแบบ diverging |
| R9 | reproducibility | เพิ่ม: split 16/15 rel seed0, first-token argmax matching, patch positions (last-3 aligned), noise 3×std, topnorm/mean-abl 64 prompts, specialist criterion (\|N\|<0.01), Spearman interpolation, MMLU seed0 subset |

## รอ VM (สคริปต์พร้อม push แล้ว)
| สคริปต์ | รายการรีวิว | งาน |
|---------|------------|-----|
| e0b_metric_parity.py | R7 | LD/head-effect/rank fp32 vs bf16 บนตัวอย่างเดียวกัน |
| e5b_mmlu_perq.py | R8 | MMLU per-question none/spec1 → paired McNemar |
| e6b_peritem.py | R8 | Llama/Qwen per-item → CI |
| fig3_extract.py | R6 | answer-position attention 32 prompts + BOS/distractor breakdown |
| e5c_correct_context.py | R11 | correct-context control (spec1 ทำให้ใช้ context ถูกแย่ลงไหม) |
- fig3 render ใหม่ (colorbar/axes/8pt) หลัง data ถึง — R6+R15
- ใส่ exact TransformerLens version — R9

## หมายเหตุ
- VM us-central1-a STOCKOUT ตั้งแต่ 13 ต.ค. — retry ทุก 10 นาที (9 ครั้ง) ถ้าไม่ได้จะสร้าง VM ใหม่ zone อื่น
- ตอนนี้ paper = 8 หน้า (เต็มงบ PAPER) — เผื่อการเพิ่มเนื้อหาจาก VM อาจต้อง trim
