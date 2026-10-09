# References — ตรวจสอบ Gap และรายการอ้างอิงสำหรับ Paper

> **อัปเดตล่าสุด:** 30 กันยายน 2026 (สำรวจครั้งที่ 3 — ยืนยันว่า trade-off dose–response ยังว่าง, ยืนยันปัญหา Gemma-2 softcapping กับ TransformerLens พร้อมวิธีแก้, เพิ่ม pyvene/nnsight, head-taxonomy survey จาก Patterns)
> **วิธีใช้:** §0 คือคำตอบเรื่อง validity ของ gap; §1–§7 คือรายการอ้างอิงจัดกลุ่มตามธีมของ Related Work แต่ละรายการมี citation, ลิงก์, โมเดลที่ใช้, สรุป และ "บทบาทต่องานเรา" (จะอ้างตรงไหน / แยกยังไง); §8 คือรายการที่ต้อง verify เพิ่มก่อนใส่ bibliography
> **สัญลักษณ์:** ✅ = ตรวจสอบจากแหล่งจริงแล้ว (arXiv/ACL Anthology/OpenReview) · ⚠️ = ข้อมูลหลักน่าเชื่อถือ แต่มีบางฟิลด์ (arXiv ID/venue/ผู้แต่งบางคน) ที่ควรเช็คซ้ำตอนทำ bibliography

---

## 0. ผลการตรวจสอบ Gap (Verdict)

**คำถาม:** gap ที่อ้างในแผน ("งาน circuit ส่วนใหญ่ทำบนโมเดลใหญ่ ไม่ค่อยทำ SLMs และไม่ quantified trade-off") ใช้ได้จริงไหม?

**คำตอบสั้น: ใช้ได้แค่บางส่วน — ต้องปรับถ้อยคำก่อนเขียน Introduction จริง**

### 0.1 สิ่งที่ "ใช้ไม่ได้แล้ว" (ห้ามอ้างใน paper)

- ❌ อ้างว่า *"ยังไม่มีใครศึกษา mechanism ของ hallucination ในโมเดลขนาดเล็ก"* — **ไม่จริง** เพราะมีงานแล้วอย่างน้อย 3 ชิ้น:
  - Yu et al. (Findings of EMNLP 2024): ใช้ interpretability methods ระบุสาเหตุ hallucination ใน Llama-2, Pythia, GPT-J และเสนอการแก้แบบ "restoration"
  - Li et al. ITI (NeurIPS 2023): intervention ระดับ attention head เพื่อเพิ่มความจริงบน TruthfulQA (Llama หลายขนาด)
  - Karim (arXiv 2026): head-level causal analysis ของ procedural hallucination บน Gemma โดยตรง
- ❌ อ้างว่า *"ไม่มีงาน intervention ลด hallucination เลย"* — ไม่จริง (ITI, Lookback Lens, Yu et al. ฯลฯ)

### 0.2 ช่องว่างที่ "ยังยืนได้" หลังตรวจแล้ว (ใช้เป็น gap จริงใน paper)

1. **ระดับความละเอียด + corruption regime:** ยังไม่พบงานที่ทำ **head-level causal activation patching** ของ **context–memory conflict แบบ text-only** (prompt มีบริบทหลอกล่อขัดกับความจำพารามิเตอร์) บน **SLMs ขนาด < 2B หลายสถาปัตยกรรม** แล้ววัดผลต่อ hallucination rate
   - งานที่ใกล้สุดทำในมุมอื่น: Yu et al. 2024 ศึกษา hallucination แบบ **knowledge-deficit** (โมเดลไม่รู้คำตอบ — ต่างจากของเราที่รู้แต่ถูกบริบทกด), Jiang et al. 2026 ทำ conflict แบบ **multimodal**, Du et al. 2024 ทำ conflict แบบ **behavioral** (ไม่แตะภายในโมเดล)
2. **Trade-off แบบ quantified:** ไม่พบงานที่รายงาน **dose–response curve** (ปิด k = 1…20 heads) ระหว่าง hallucination rate กับ MMLU/PPL อย่างเป็นระบบใน SLMs — ITI รายงาน benchmark ก่อน/หลังเพียงบางตัว
3. **Protocol กัน circularity:** dev/held-out relation split + การยืนยันข้ามสถาปัตยกรรมยังไม่เป็นมาตรฐานของงานกลุ่มนี้

> ⚠️ **อัปเดตจากรอบสำรวจที่สอง (§0.5):** มีงาน CM-conflict ระดับ attention head เพิ่มเข้ามาในปี 2025–2026 (โดยเฉพาะ JuICE, ICML 2025) — ช่องว่างข้อ 1 ต้องเขียนให้แคบและแม่นยำขึ้น ดู §0.5

### 0.3 หลักฐานเชิงบวกว่า gap ยังมีชีวิต (อ้างได้ใน paper)

- **Lamba et al. (2025)** ศึกษา hallucination ใน Gemma (รวม Gemma-2-2B) แล้ว**ระบุ activation patching / causal tracing เป็น future work อย่างชัดเจน** — งานเรากรอกช่องที่งานนี้เปิดทิ้งไว้
- งานตามมา (พ.ย. 2025) ของกลุ่มเดียวกันยังอยู่ระดับ symbolic/linguistic ไม่ได้ลงระดับ circuit
- งานปี 2026 (Jiang; Choi) ที่เริ่มเข้าใกล้ก็ยังอยู่คนละมุม (multimodal / model lineages)

### 0.4 ความเสี่ยงและข้อควรทำ

- พื้นที่นี้**เคลื่อนเร็วมาก** — ก่อน submit ต้องสืบค้นซ้ำ (คำค้นแนะนำอยู่ท้ายแผนหลัก §8.5) และถ้าเจองานที่ทำเหมือนเกือบทุกอย่าง ให้เน้นความต่างที่ trade-off quantification + protocol + cross-model
- **วิธี position ตัวเอง:** เขียนว่าเป็น *"a systematic empirical study with quantified intervention trade-offs in sub-2B models"* **ไม่ใช่** *"the first mechanistic account of hallucination"*
- คำศัพท์ที่ควรใช้ให้ตรง literature: เรียกสถานการณ์ prompt หลอกล่อว่า **context–memory (CM) conflict** ตาม taxonomy ของ Xu et al. (EMNLP 2024)

### 0.5 ผลจากรอบสำรวจที่สอง (30 ก.ย. 2026) — สำคัญต่อการเขียน gap

รอบสำรวจที่สอง (โฟกัสคำค้น "context-memory conflict" + attention heads) พบ **คลัสเตอร์งาน 2025–2026 ที่ทำ CM-conflict ระดับ head โดยตรง** ซึ่งรอบแรกไม่เจอ:

1. **JuICE / Taming Knowledge Conflicts (Li, Chen & Tong — ICML 2025 Spotlight):** วิเคราะห์ attention heads ใน knowledge conflicts พบ **"superposition"** — heads ที่ทรงอิทธิพลมีบทบาททั้ง memory และ context พร้อมกัน (ฝ่าฝืนสมมติฐาน memory-heads vs context-heads แยกขั้ว) แล้วเสนอ intervention แบบ test-time (dual-run) — ต้อง**อ่านฉบับเต็มก่อน finalize gap** ว่าเขาเลือก heads ด้วยวิธีใด (ดูเหมือนเป็น influence-based ไม่ใช่ systematic patching sweep)
2. **Where Knowledge Collides (Pham, Borkakoty & Hou, 2026):** mechanistic study ของ **intra-memory conflict** (ขัดแย้งในความจำภายใน ไม่ใช่ CM) — พบ conflict เกิดและถูกแก้ที่ final layers, head-intervention ชนะ layer-intervention, ไม่มี universal circuit — ใช้เทียบผลและเป็น methodological precedent
3. **DCRD (Zhou et al., 2026, ตีพิมพ์ที่ IEEE TASLP):** แก้ CM conflict ที่ระดับ **decoding** (attention-map ทำนาย conflict แล้ว route การ decode) — คนละระดับกับ intervention ภายในของเรา
4. **Task Matters (Sun, Bai & Dredze — ACL 2026):** แบบ behavioral — CM conflict ขึ้นกับ task เชิงความรู้ที่โจทย์ต้องการ + พบว่า conflict ทำให้ LLM-as-judge เบ้

**ผลกระทบต่อ gap (ฉบับที่ควรใช้เขียน Introduction):**

- ❌ เพิ่มข้อห้าม: อ้างว่า *"ไม่มีงานแตะ attention heads ใน CM conflict"* — **ไม่จริงแล้ว** (JuICE ทำแล้ว และเป็น ICML Spotlight)
- ✅ ช่องว่างที่ยังเหลือจริง (หลังรอบ 2):
  1. **Systematic localization evidence**: activation-patching sweep แบบครบตาราง (block/head × position) + heatmap + dev/held-out protocol บน **sub-2B SLMs หลายสถาปัตยกรรม** — JuICE ทำ influence analysis เพื่อ intervention ไม่ได้ตีพิมพ์ localization map เปรียบเทียบ (ต้องยืนยันจากฉบับเต็ม)
  2. **Dose–response trade-off** (hallucination rate vs MMLU/PPL เมื่อปิด k=1…20 heads) — ยังไม่มีใครรายงาน รวมถึงใน JuICE
  3. **Counterfactual distractor (CounterFact-style) + copy-suppression controls** (Campregher 2025) — งาน CM-conflict กลุ่มนี้ใช้ retrieval-style conflicts เป็นหลัก
- ✅ **มุมวิเคราะห์ใหม่ที่ได้มาฟรีจาก JuICE:** เราสามารถทดสอบข้ออ้าง *superposition* ใน data ของเราโดยตรง (heads ของเราเอนไปข้างเดียวหรือสองข้าง?) — เพิ่มเป็นคำถามย่อยของ E4 ทำให้ paper มีบทสนทนากับ ICML 2025 ไม่ใช่ทำงานลอย
- **ตำแหน่งยืนสุดท้าย:** *"a systematic, multi-model patching study of CM-conflict localization in sub-2B SLMs with quantified intervention trade-offs"* — เป็นการเติมช่องว่างเชิงประจักษ์ ไม่ใช่การเปิดพื้นที่ใหม่

### 0.6 ผลจากรอบสำรวจที่สาม (30 ก.ย. 2026)

1. **Contribution #2 (dose–response trade-off) ยังปลอดภัย:** ค้นด้วยหลายมุม (head ablation × MMLU × degradation/trade-off) แล้ว **ไม่พบงานใดรายงาน curve แบบเรา** — งาน ablation ที่มีอยู่เป็นสาย efficiency/pruning หรือ hybrid-architecture ไม่ใช่สาย hallucination
2. **ปัญหา Gemma-2 softcapping ใน TransformerLens ยืนยันว่ามีจริง:** มี bug report ว่า HF กับ HookedSAETransformer ให้ output "slightly different" — สาเหตุหลักคือ legacy weight processing (LayerNorm folding / `center_unembed`) ที่ทำลาย invariance ของ tanh softcap + ความไม่สอดคล้องของ dtype (float32 vs bfloat16) — **วิธีแก้ที่พบ:** ใช้ TransformerLens รุ่นใหม่ (ค่า default อนุรักษ์ raw HF weights ให้ logits/activations ตรงกับ HF), ตั้ง `center_unembed=False`, คุม dtype ให้คงที่ และเทียบ logits แบบ post-softcap ให้ถูกจุด — รายละเอียดถูกใส่ไว้ใน E0 ของแผนหลักแล้ว
3. **pyvene เป็นทางออกที่ดีกว่าที่คิด:** เป็น library จาก Stanford ที่ทำ interchange intervention (activation patching) **บน HF model ตรงๆ** แบบ declarative — ข้ามปัญหา "การแปลงโมเดล" ของ TransformerLens ทั้งหมด จึงเหมาะเป็น fallback อันดับแรกของ E0 (ดีกว่าเขียน PyTorch hooks เอง)
4. **ECTI-CIT (จากความรู้ทั่วไปของระบบค้น ยังไม่ได้ verify สด):** ประวัติมักอยู่ **Q3** ใน SJR (ดีกว่าที่แผนเดิมสมมติว่า Q4) — ต้องเช็ค scimagojr.com ก่อนตัดสินใจส่ง

---

## 1. งานที่ใกล้เคียงที่สุด — ต้องอ้างและแยกให้ชัด (Must-cite)

### [C1] Yu et al. (2024) — ✅ ใกล้สุด สำคัญที่สุด
- **Citation:** Lei Yu, Meng Cao, Jackie C.K. Cheung, Yue Dong. *Mechanistic Understanding and Mitigation of Language Model Non-Factual Hallucinations.* Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 7943–7956.
- **ลิงก์:** https://aclanthology.org/2024.findings-emnlp.466/ (DOI: 10.18653/v1/2024.findings-emnlp.466)
- **โมเดล:** Llama-2, Pythia, GPT-J (หลายขนาด)
- **สรุป:** สร้าง diagnostic datasets จาก subject–relation queries แล้ว trace hallucination ผ่าน internal representations; พบสาเหตุ 2 แบบ: (1) ความรู้ subject-attribute บกพร่องใน MLP ชั้นล่าง, (2) การเลือก object attribute ผิดใน attention heads ชั้นบน; เสนอการแก้แบบ "targeted restoration of the fact recall pipeline"
- **บทบาทต่องานเรา:** อ้างใน Related Work ว่าเป็นงาน mechanism ใกล้ชิดที่สุด แล้วแยกว่าเขาศึกษา hallucination จาก **ความรู้บกพร่อง** (โมเดลไม่รู้) ส่วนเราศึกษา **บริบทหลอกล่อกดความรู้ที่มีอยู่** (knows-but-overridden) — corruption regime, การวัด, และ intervention ต่างกัน; งานของ Yu ยังไม่มี dose–response trade-off กับ MMLU/PPL

### [C2] Li et al. (2023) — Inference-Time Intervention (ITI) ✅
- **Citation:** Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg. *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model.* NeurIPS 2023 (Advances in Neural Information Processing Systems 36).
- **ลิงก์:** https://arxiv.org/abs/2306.03341
- **โมเดล:** LLaMA (หลายขนาด, 7B–65B)
- **สรุป:** หา attention heads ที่ linear probe ทำนาย truthfulness ได้ดี แล้ว shift activation ตาม "truthful direction" ตอน inference → TruthfulQA MC ดีขึ้นชัดเจน
- **บทบาทต่องานเรา:** ต้นแบบของ intervention ระดับ head; จุดต่างคือ ITI เลือก heads ด้วย **probe (correlational)** และไม่รายงานต้นทุนต่อ capability เป็นระบบ ส่วนเราเลือกด้วย **causal patching** + รายงาน trade-off curve; ITI เป็นแรงบันดาลใจของ RQ3 โดยตรง

### [C3] Du et al. (2024) — Context versus Prior Knowledge ✅
- **Citation:** Kevin Du, Vésteinn Snæbjarnarson, Niklas Stoehr, Jennifer C. White, Aaron Schein, Ryan Cotterell. *Context versus Prior Knowledge in Language Models.* ACL 2024 (Long Papers), pp. 13211–13235.
- **ลิงก์:** https://arxiv.org/abs/2404.04633 · https://aclanthology.org/2024.acl-long.714/
- **สรุป:** นิยาม *persuasion score* (ต่อ context) และ *susceptibility score* (ต่อ entity) แบบ mutual information; พบว่าโมเดลพึ่งความจำมากขึ้นเมื่อ entity คุ้นเคยใน training corpus และ context บางแบบ "โน้มน้าว" ได้มากกว่า
- **บทบาทต่องานเรา:** อ้างเป็นงาน behavioral ที่ quantifies **เมื่อไร** context ชนะความจำ; เราตอบคำถามถัดไป: **"ที่ไหนในโมเดล"** การชนะนั้นเกิด — ใช้ setup ของเขาอ้างอิงตอนออกแบบ misleading suite; จุดต่าง: ไม่มี internal analysis

### [C4] Chuang et al. (2024) — Lookback Lens ✅
- **Citation:** Yung-Sung Chuang, Linlu Qiu, Cheng-Yu Hsieh, Ranjay Krishna, Yoon Kim, James Glass. *Lookback Lens: Detecting and Mitigating Contextual and Long-Source Hallucinations in Large Language Models Using Only Attention Maps.* Findings of EMNLP 2024. ⚠️ (รายชื่อ co-author บางตำแหน่งควรเช็คซ้ำจากหน้า arXiv)
- **ลิงก์:** https://arxiv.org/abs/2407.07071
- **สรุป:** contextual hallucination สัมพันธ์กับ "lookback ratio" (น้ำหนัก attention ที่มองกลับไปที่ source context); ใช้ classifier บน attention maps ตรวจจับ และ guided decoding ลด hallucination โดยไม่ต้องแตะ logits
- **บทบาทต่องานเรา:** อ้างเป็น evidence ว่า attention สัมพันธ์กับ contextual hallucination (แต่เป็น correlational + ระดับ generation); เราให้มุม causal ระดับ head; ใช้อ้างใน Discussion ว่าอนาคตรวมสองแนวทางได้

### [C5] Jiang et al. (2026) — Attention Head Imbalance (multimodal) ✅
- **Citation:** Jinrui Jiang, Zhangtai Wu, Zhen Wu, Xinyu Dai. *Causal Evidence for Attention Head Imbalance in Modality Conflict Hallucination.* arXiv:2605.19250 (พ.ค. 2026).
- **ลิงก์:** https://arxiv.org/abs/2605.19250
- **สรุป:** patch activation ของ head (l, i) จาก clean run เข้า conflict run เพื่อระบุ heads ที่ "ขับ" vs "ต้าน" hallucination ในลักษณะ modality conflict ของ large vision-language models
- **บทบาทต่องานเรา:** งานที่ **คล้ายวิธีการเราที่สุด** (patching บน conflict runs) แต่เป็น multimodal — อ้างแล้วแยกว่าเราเป็น text-only SLMs + วัด trade-off; สามารถยืม terminology "heads ที่สนับสนุน vs ต้านทาน hallucination" มาใช้ใน E4

### [C6] Karim (2026) — Attention Deficits / Procedural Hallucinations ✅
- **Citation:** A. Karim. *Attention Deficits in Language Models: Causal Explanations for Procedural Hallucinations.* arXiv:2602.19239 (2026). ⚠️ (ชื่อผู้แต่งเต็ม/venue ควรเช็คซ้ำ)
- **ลิงก์:** https://arxiv.org/abs/2602.19239
- **สรุป:** อธิบาย procedural hallucination (ทำตามขั้นตอนถูกแต่รายงานผิดใน step สุดท้าย) ด้วย head-level mechanisms บน Gemma: พบ "misbinding head" และ "anti-recency head" ที่ถ่วงดุลกัน; มี checkpoint analysis และการ patch  concatenated head outputs
- **บทบาทต่องานเรา:** อ้างว่า head-level causal analysis บน Gemma เริ่มมีแล้วในประเภท hallucination อื่น (procedural) — เราเติมประเภท factual/context-conflict ที่ยังว่าง

### [C7] Choi et al. (2026) — Inherited Heads ⚠️
- **Citation:** Miso Choi, Seonga Choi, Mincheol Kwon, Woosung Joung, et al. *The Truth Stays in the Family: Enhancing Contextual Truthfulness via Inherited Heads in Model Lineages.* arXiv (ส.ค. 2026, Korea University Vision AI Lab).
- **ลิงก์:** ค้นหาชื่อเรื่องบน arXiv/OpenReview (ยังไม่มีเลข ID ยืนยัน — ต้อง verify ก่อนอ้าง)
- **สรุป:** ศึกษา attention heads ที่เกี่ยวกับ contextual truthfulness และถูก "สืบทอด" ข้ามรุ่นใน model lineages; ใช้ intervention เพื่อเพิ่มความจริงตามบริบท
- **บทบาทต่องานเรา:** ใกล้ RQ3 ในแง่ contextual truthfulness + heads แต่มุมคือ **ความต่อเนื่องข้ามรุ่นโมเดล**; เราต่างที่เป็น cross-architecture แบบไม่ผูก lineages + โฟกัส trade-off — **ต้องอ่านงานนี้ก่อน submit เพราะเป็น overlap ที่สุดในกลุ่ม 2026**

### [C8] Lamba et al. (2025) — Symbolic Triggers in Gemma ✅ (หลักฐาน gap)
- **Citation:** Lamba et al. *Investigating Symbolic Triggers of Hallucination in Gemma Language Models.* arXiv:2509.09715 (2025); เผยแพร่ที่ CEUR-WS Vol-4064 (SymGenAI4Sci workshop). ⚠️ (initial ผู้แต่นำ + ชื่อไฟล์ CEUR ควรเช็คซ้ำ)
- **ลิงก์:** https://arxiv.org/abs/2509.09715 · https://ceur-ws.org/Vol-4064/SymGenAI4Sci-paper2.pdf
- **สรุป:** วิเคราะห์ hallucination ใน Gemma (รวม Gemma-2-2B) บน HaluEval/TruthfulQA; พบ hallucination rate สูง (เฉลี่ย ~79%) เมื่อมี symbolic triggers (negation, modifiers, named entities, numbers); **ระบุ activation patching + causal tracing เป็น future work อย่างชัดเจน**
- **บทบาทต่องานเรา:** อ้างเป็นหลักฐานว่า gap ที่เรากรอกยังเปิดอยู่ถึงปลายปี 2025; งานต่อของกลุ่มนี้ (พ.ย. 2025: *Symbolic Localization of Hallucination across HaluEval*) ก็ยังอยู่ระดับ symbolic ไม่ลง circuit

### [C9] Li, Chen & Tong (2025) — JuICE / Taming Knowledge Conflicts ✅ (สำคัญรองจาก Yu et al.)
- **Citation:** Gaotang Li, Yuzhong Chen, Hanghang Tong. *Taming Knowledge Conflicts in Language Models.* ICML 2025 (Spotlight).
- **ลิงก์:** https://arxiv.org/abs/2503.10996 · code: https://github.com/GaotangLi/JUICE
- **โมเดล:** 11 datasets, 6 architectures (ชื่อโมเดล/ขนาดดูในฉบับเต็ม)
- **สรุป:** ท้าทายสมมติฐาน "memory heads vs context heads แยกขั้ว" ด้วยการค้นพบ **superposition** — heads ที่ทรงอิทธิพลมีบทบาททั้ง parametric memory และ context พร้อมกัน; เสนอ **JuICE** (Just Run Twice): test-time attention intervention แบบ dual-run ที่ตัดผล superposition เพื่อบังคับทิศ (เชื่อ memory หรือเชื่อ context) ตามที่ต้องการ + การวิเคราะห์เชิงทฤษฎีของ superposition
- **บทบาทต่องานเรา:** **ต้องอ้างและแยกให้ชัดที่สุดในกลุ่ม CM-conflict** — จุดต่างของเรา: (1) เราให้ systematic patching localization (heatmap + dev/held-out) ไม่ใช่แค่ influence-based selection เพื่อ intervention, (2) เรา quantified ต้นทุนต่อ capability (dose–response กับ MMLU/PPL) ซึ่ง JuICE ไม่รายงาน, (3) เราโฟกัส sub-2B + counterfactual distractor; **โบนัส:** E4 ของเราทดสอบข้ออ้าง superposition โดยตรงบน SLMs

### [C10] Pham, Borkakoty & Hou (2026) — Intra-Memory Conflict ✅
- **Citation:** Minh Vu Pham, Hsuvas Borkakoty, Yufang Hou. *Where Knowledge Collides: A Mechanistic Study of Intra-Memory Knowledge Conflict in Language Models.* arXiv:2601.09445 (2026).
- **ลิงก์:** https://arxiv.org/abs/2601.09445
- **สรุป:** ศึกษา conflict **ภายใน** parametric memory (โมเดลเก็บข้อเท็จจริงขัดแย้งกันเองเรื่อง subject เดียวกัน — ต่างจาก CM conflict ของเรา); ใช้ targeted **attention-head interventions** เทียบ layer-wise; พบ conflict เกิด/ถูกแก้ที่ **final layers**, head-targeted ชนะ layer-targeted, และ**ไม่มี universal conflict circuit**
- **บทบาทต่องานเรา:** precedent เชิงวิธีการ (head vs layer granularity) + ผล "final layers" ใช้เทียบกับ heatmap ของเรา; คำเตือนเรื่อง synthetic vs real-world gap ที่เราควรระวังด้วย (ใช้ facts จริงจาก CounterFact ไม่ใช่ synthetic)

### [C11] Zhou et al. (2026) — DCRD (decoding-level) ✅
- **Citation:** Yigeng Zhou, Wu Li, Yifan Lu, Yequan Wang, Xuebo Liu, Wenya Wang, Jun Yu, Min Zhang, Jing Li. *Mitigating Context-Memory Conflicts in LLMs through Dynamic Cognitive Reconciliation Decoding.* IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP), 2026.
- **ลิงก์:** https://arxiv.org/abs/2605.12185
- **สรุป:** ใช้ attention map ทำนายว่าจะเกิด conflict แล้ว route ระหว่าง greedy decoding ปกติกับ dynamic decoding ที่เชื่อ context — แก้ปัญหาของ contrastive decoding แบบคงที่ที่รบกวน output ในกรณีไม่มี conflict; สร้าง benchmark **ConflictKG**
- **บทบาทต่องานเรา:** representative ของสาย **decoding-level** — อ้างเทียบว่าเราแทรกแซงที่ representation (head) ซึ่งอธิบายกลไกได้ ไม่ใช่แค่แก้ที่ output distribution

### [C12] Sun, Bai & Dredze (2026) — Task Matters ✅
- **Citation:** Kaiser Sun, Fan Bai, Mark Dredze. *Task Matters: Knowledge Requirements Shape LLM Responses to Context-Memory Conflict.* ACL 2026.
- **ลิงก์:** https://arxiv.org/abs/2506.06485
- **สรุป:** behavioral — ผลกระทบของ CM conflict ขึ้นกับ task (งานที่ต้องพึ่ง context vs พึ่ง parametric knowledge) และความ plausible ของ conflict; prompting แบบ reiterate context เพิ่ม context-reliance (ช่วยงานประเภทหนึ่ง แต่ทำร้ายอีกประเภท); และ conflict ทำให้ LLM-as-judge เบ้
- **บทบาทต่องานเรา:** กรอบการอภิปรายว่า "บริบทควรชนะหรือความจำควรชนะ" ขึ้นกับ task — เชื่อมกับ Discussion ของเราว่า intervention ที่ดีต้อง selective ไม่ใช่บังคับฝ่ายเดียว

### [C13] (ส.ค. 2025) — Internal Origins of Sycophancy ⚠️
- **Citation:** *Uncovering the Internal Origins of Sycophancy in Large Language Models.* arXiv:2508.02087 (2025). (ผู้แต่ง/venue ต้อง verify เพิ่ม)
- **ลิงก์:** https://arxiv.org/abs/2508.02087
- **สรุป:** ใช้ logit-lens + **causal activation patching** ระบุกำเนิดของ sycophancy (การเข้าข้างความเชื่อผิดของผู้ใช้) แบบสองระยะ: preference ปรากฏที่ late-layer output ก่อน แล้วกลไกภายในหล่อเลี้ยงพฤติกรรมต่อ
- **บทบาทต่องานเรา:** งาน mechanistic ที่ใกล้ระดับวิธีการที่สุดของสาย "ผู้ใช้หลอก" (user-provided misleading signal) — เชื่อมกับ RQ2: บริบทหลอกล่อแบบ counterfactual ของเราเป็น sibling ของ sycophancy; อ้างและชี้ว่าเราโฟกัส factual override ไม่ใช่ preference agreement

---

## 2. Hallucination: ภาพรวม การวัด และการตรวจจับจากภายในโมเดล

### [H1] Ji et al. (2023) — Survey หลัก ✅
- **Citation:** Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, Pascale Fung. *Survey of Hallucination in Natural Language Generation.* ACM Computing Surveys 55(12), 2023.
- **ลิงก์:** https://arxiv.org/abs/2202.03629
- **บทบาท:** นิยาม/ประเภท hallucination ใน Introduction + Related Work

### [H2] Zhang et al. (2024) — Hallucination Snowballing ✅
- **Citation:** Muru Zhang, Ofir Press, William Merrill, Alisa Liu, Noah A. Smith. *How Language Model Hallucinations Can Snowball.* ICML 2024 (PMLR v235).
- **ลิงก์:** https://arxiv.org/abs/2305.13534 · https://proceedings.mlr.press/v235/zhang24ay.html
- **สรุป:** ความผิดพลาดตอนต้นทำให้โมเดล over-commit และสร้างความผิดต่อเนื่อง; ChatGPT/GPT-4 จำแนกข้อผิดของตัวเองได้ 67%/87%
- **บทบาท:** อ้างถึง hallucination แบบ self-reinforcing ใน Introduction; งานเราโฟกัสจุดกำเนิด (context conflict) ไม่ใช่การซ้อนทับ

### [H3] Kalai et al. (2025) — Why Language Models Hallucinate ⚠️
- **Citation:** Adam Kalai et al. (OpenAI). *Why Language Models Hallucinate.* OpenAI technical report / arXiv, 2025.
- **ลิงก์:** https://openai.com/index/why-language-models-hallucinate/ (มี PDF)
- **สรุป:** อธิบายเชิงทฤษฎีว่า training/evaluation ที่ให้รางวัลกับการเดา (ไม่ใช่การยอมรับว่าไม่รู้) ทำให้เกิด hallucination
- **บทบาท:** มุมมอง incentive-level ประกอบ Discussion — บอกว่า intervention ระดับ activation (แบบเรา) แก้ที่ symptom ไม่ใช่ต้นน้ำ

### [H4] Azaria & Mitchell (2023) — Internal state รู้ว่าตัวเองโกหก ✅
- **Citation:** Amos Azaria, Tom Mitchell. *The Internal State of an LLM Knows When It's Lying.* Findings of EMNLP 2023. ⚠️ (ยืนยัน Findings vs main ตอนเขียน)
- **ลิงก์:** https://arxiv.org/abs/2304.13734
- **บทบาท:** สาย probing-for-truth — อ้างเทียบกับแนว causal ของเรา (probe บอก "รู้" แต่ไม่ได้บอก "ตัวการ")

### [H5] Marks & Tegmark (2024) — Geometry of Truth ✅
- **Citation:** Samuel Marks, Max Tegmark. *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets.* COLM 2024 (oral); เวอร์ชันแรกนำเสนอที่ NeurIPS 2023 ATTRIB workshop.
- **ลิงก์:** https://arxiv.org/abs/2310.06824
- **สรุป:** true/false representations แยกเชิงเส้นตาม "truth direction"; probe  generalize ข้าม dataset; มี causal add/remove experiments (difference-in-means, LEACE)
- **บทบาท:** อ้างใน Related Work (สาย truth directions) และเทียบกับ ITI ใน Discussion; สนับสนุนว่ามี structure เชิงเส้นในการแทนความจริง — แต่เราสนใจระดับ head/layer circuit มากกว่าทิศทางเดี่ยว

### [H6] Chen et al. (2025) — INSIDE ⚠️
- **Citation:** Jiahao Chen et al. *INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection.* ICLR 2025.
- **ลิงก์:** ค้น "INSIDE hallucination detection" บน arXiv (ID ที่จำได้คือ 2402.03744 — ต้องยืนยัน)
- **บทบาท:** สาย detection จาก internal states ปีล่าสุด — อ้างสั้นๆ ว่า detection แข็งแรงขึ้น แต่ mitigation เชิงกลไกยังกระจัดกระจาย

### [H7] Lin, Hilton & Evans (2022) — TruthfulQA ✅
- **Citation:** Stephanie Lin, Jacob Hilton, Owain Evans. *TruthfulQA: Measuring How Models Mimic Human Falsehoods.* ACL 2022.
- **ลิงก์:** https://arxiv.org/abs/2109.07958
- **บทบาท:** benchmark ปลายทางของเรา (MC1/MC2); อ้างตอนอธิบายว่าทำไมใช้เป็น secondary metric (contamination risk)

### [H8] Farquhar et al. (2024) — Semantic Entropy (Nature) ✅
- **Citation:** Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, Yarin Gal. *Detecting hallucinations in large language models using semantic entropy.* Nature 630, 2024 (DOI: 10.1038/s41586-024-07421-0).
- **ลิงก์:** https://www.nature.com/articles/s41586-024-07421-0
- **สรุป:** ตรวจจับ hallucination ด้วย entropy ระดับ "ความหมาย" (cluster คำตอบตาม semantic) แบบ unsupervised ไม่ต้อง retrain
- **บทบาท:** representative ของสาย **uncertainty-based detection** — อ้างใน Related Work ว่า detection เก่งขึ้นมากแล้ว แต่ทิศทางของเราคือ *causal understanding + intervention* ไม่ใช่ detection

### [H9] CLAP — Cross-Layer Attention Probing (2025) ⚠️
- **Citation:** *Training-free Truthfulness Detection via Value ... (Cross-Layer Attention Probing/CLAP).* arXiv/ResearchGate, 2025. (ชื่อเต็ม + ผู้แต่งต้อง verify เพิ่ม — ถูก rate limit ระหว่างตรวจ)
- **สรุป:** activation probing แบบข้าม layer บน attention patterns เพื่อตรวจจับ hallucination แบบ training-free (ปรับปรุงจาก Lookback Lens ให้ละเอียดระดับ response เดียวกัน)
- **บทบาท:** สาย detection-via-attention ล่าสุด — คู่เทียบสมัยของ Lookback Lens [C4] ใน Related Work

---

## 3. Context–Memory Conflict (บริบทขัดความจำพารามิเตอร์)

### [K1] Xu et al. (2024) — Knowledge Conflicts Survey ✅
- **Citation:** Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang, Hongru Wang, Yue Zhang, Wei Xu. *Knowledge Conflicts for LLMs: A Survey.* EMNLP 2024 (main), pp. 8541–8565.
- **ลิงก์:** https://arxiv.org/abs/2403.08319 · https://aclanthology.org/2024.emnlp-main.486/
- **สรุป:** taxonomy ของ knowledge conflicts: Context–Memory (CM), Inter-Context (IC), Intra-Memory (IM)
- **บทบาท:** กรอบทฤษฎีของ RQ2 — เรียกสิ่งที่เราศึกษาว่า "CM conflict"; อ้างตอนนิยาม misleading suite; **อ่าน survey นี้ก่อนเขียน Related Work จะได้ครอบคลุมงานใน taxonomy ที่เกี่ยวข้อง**

### [K2] Kassner & Schütze (2020) — Negated/Misprimed Probes ✅
- **Citation:** Nora Kassner, Hinrich Schütze. *Negated and Misprimed Probes for Pretrained Language Models: Birds Can Talk, But Cannot Fly: Negative Probes for Language Models with Systematic Negation or Character Variation.* ACL 2020.
- **ลิงก์:** https://arxiv.org/abs/1911.03343
- **สรุป:** โมเดล (BERT-era) ตอบผิดเมื่อ probe มี negation หรือ mispriming — ต้นตอของแนวคิด "prompt หลอกล่อกดความรู้"
- **บทบาท:** อ้างประวัติของปรากฏการณ์นี้ย้อนหลังถึงยุค PLM; แสดงว่าปัญหาเก่าแก่แต่การอธิบายเชิงกลไกยังไม่ครบ

### [K3] Neeman et al. (2023) — DisentQA ⚠️
- **Citation:** Ella Neeman, Zhilin Wang, Royi Rassin, ... Vered Shwartz. *DisentQA: Disentangling Parametric and Contextual Knowledge in Language Models.* ACL 2023.
- **ลิงก์:** ค้นบน ACL Anthology (arXiv ที่จำได้ ~2306.09500 — ยืนยันก่อนใช้)
- **สรุป:** แยกการใช้ parametric vs contextual knowledge ด้วย contrastive decoding ระหว่าง context/no-context
- **บทบาท:** มุม decoding-level ของ CM conflict — อ้างเทียบว่าเราแทรกแซงที่ representation ไม่ใช่ logits

### [K4] Shi et al. — Context-Aware Decoding ⚠️ (ยังไม่ยืนยันรายละเอียด)
- **Citation:** *Trusting Your Evidence: Hallucinate Less with Context-Aware Decoding.* (~2023–2024, ACL)
- **สรุป:** ผสม output distribution แบบมี/ไม่มี context เพื่อเชื่อ context มากขึ้น
- **บทบาท:** decoding-level baseline เชิงแนวคิดอีกตัว; **ต้อง verify ผู้แต่ง/arXiv ก่อนอ้าง** (ค้นเจอแต่ rate limit ระหว่างตรวจสอบ)

### [K5] Mallen et al. (2023) — PopQA (entity popularity) ✅
- **Citation:** Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, Hannaneh Hajishirzi. *When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories.* ACL 2023.
- **ลิงก์:** https://arxiv.org/abs/2212.10511 · https://aclanthology.org/2023.acl-long.546/ · dataset: https://github.com/AlexTMallen/adaptive-retrieval
- **สรุป:** PopQA ~14k คำถามจาก Wikipedia triples พร้อม **entity popularity score**; พบว่า parametric memory ของ LLM ไม่น่าเชื่อถือกับ long-tail entities
- **บทบาทต่องานเรา:** ใช้ popularity เป็น **covariate ใน E1** — stratify facts ตามความนิยมของ entity แล้ววิเคราะห์ใน E4 ว่า circuit/gullibility เปลี่ยนตาม popularity ไหม (เชื่อมกับ susceptibility score ของ Du et al. [C3] แบบ mechanistic) — เป็นการเพิ่มมิติวิเคราะห์ที่งานอื่นในกลุ่มยังไม่ทำ

---

## 4. Factual Recall Circuits & Model Editing

### [F1] Meng et al. (2022) — ROME + CounterFact dataset ✅
- **Citation:** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS 2022.
- **ลิงก์:** https://arxiv.org/abs/2202.05262
- **สรุป:** causal tracing (Gaussian-noise corruption) ระบุ MLP mid-layers เก็บ factual associations; แก้ความจำด้วย rank-one update; **CounterFact dataset มาจากงานนี้ (dataset หลักของเรา)**
- **บทบาท:** รากฐานวิธีการของ E3a + แหล่ง dataset; อ้างใน Methodology และ Related Work

### [F2] Geva et al. (2021) — FFN เป็น Key-Value Memories ✅
- **Citation:** Mor Geva, Roei Schuster, Jonathan Berant, Omer Levy. *Transformer Feed-Forward Layers Are Key-Value Memories.* EMNLP 2021.
- **ลิงก์:** https://arxiv.org/abs/2012.14913
- **บทบาท:** พื้นฐานการตีความบทบาท MLP ใน E4

### [F3] Geva et al. (2023) — Dissecting Recall ✅
- **Citation:** Mor Geva, Jasmijn Bastings, Katja Filippova, Amir Globerson. *Dissecting Recall of Factual Associations in Auto-Regressive Language Models.* EMNLP 2023.
- **ลิงก์:** ค้นชื่อเรื่องบน ACL Anthology / arXiv
- **สรุป:** แยกขั้นของ fact recall: subject enrichment (attention ต้นๆ) → relation propagation → attribute extraction (MLP กลาง-ปลาย) บน GPT-2/Llama/Mistral
- **บทบาท:** ผลของเราต้องเทียบกับ "สามเฟส" นี้ — เป็น hypothesis ที่ E3/E4 จะไปทดสอบว่ายังคงเป็นจริงหรือไม่ภายใต้บริบทหลอกล่อ

### [F4] Hase et al. (2023) — Does Localization Inform Editing? ✅
- **Citation:** Peter Hase, Mohit Iyyer, ... (Hase, Zhang, Ranganath, Liang, Hasegawa). *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Direct Fine-Tuning of Model Editing.* NeurIPS 2023.
- **ลิงก์:** https://arxiv.org/abs/2301.04213
- **บทบาท:** อ้างคำเตือนว่า "ตำแหน่งที่ causal tracing ชี้" ไม่จำเป็นต้องเป็นจุดที่ดีที่สุดสำหรับ intervention — เหตุผลที่เราต้อง validate ผล ablation แยกจากผล patching (E5 ไม่ใช่แค่ E3)

### [F5] Hernandez et al. (2024) — Linearity of Relation Decoding ⚠️
- **Citation:** Evan Hernandez, Arnab Sen Sharma, Tal Haklay, Kevin Meng, Martin Wattenberg, Jacob Andreas, David Bau. *Linearity of Relation Decoding in Transformer Language Models.* ICLR 2024.
- **ลิงก์:** https://arxiv.org/abs/2308.09124
- **บทบาท:** relation เป็นเส้นตรงเชิงเรขาคณิต — บริบทของการตีความ attention head ที่ขนส่งข้อมูล relation

### [F6] McGrath et al. (2024) — Hydra Effect ⚠️
- **Citation:** Tom McGrath, Jan Betley, ... (McGrath et al.). *The Hydra Effect: Emergent Self-Repair in Model Editing.* ACL 2024.
- **ลิงก์:** arXiv ~2307.05933 (ยืนยันก่อนใช้)
- **สรุป:** โมเดลซ่อมแซมตัวเองหลังถูก edit — redundancy ในวงจร
- **บทบาท:** อ้างใน Discussion ว่าทำไม knockout ไม่กี่ heads อาจให้ผลน้อยกว่าที่ patching เดี่ยวชี้ (เชื่อมกับ dose–response curve ของเรา)

### [F7] Campregher (2025) — Tracing Facts or just Copies? ⚠️ (ยืนยันรายละเอียดเพิ่ม)
- **Citation:** S. Campregher. *Tracing Facts or just Copies? — Attention Heads and Counterfactual Tasks.* arXiv:2507.11809 (2025). (ชื่อเต็ม/venue ยังต้องเช็คจากหน้า arXiv — ถูก rate limit ระหว่างตรวจ)
- **ลิงก์:** https://arxiv.org/abs/2507.11809
- **สรุป:** พบว่า attention heads ที่ส่งเสริม factual output มักทำผ่าน **copy suppression แบบทั่วไป** ไม่ใช่ counterfactual reasoning เชิงเลือกเฉพาะข้อเท็จจริง
- **บทบาทต่องานเรา:** **confound ที่ต้องออกแบบการทดลองกันใน E4** — heads ที่เราตีความว่า "recall ความจำ" อาจเป็นแค่ copier/suppressor ทั่วไป; ต้องมี control (เช่น แทน distractor ด้วย token กลางๆ ที่ไม่ใช่คำตอบ, swap ตำแหน่ง true/false) เพื่อแยก "selective fact recall" ออกจาก "generic copying"

---

## 5. Activation Patching: วิธีการและข้อควรระวัง

### [P1] Vig et al. (2020) — Causal Mediation Analysis ✅
- **Citation:** Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, Stuart Shieber. *Investigating Gender Bias in Language Models Using Causal Mediation Analysis.* NeurIPS 2020.
- **ลิงก์:** ค้นบน NeurIPS proceedings
- **บทบาท:** งานต้นแบบของ activation patching (indirect effect ของ component) — อ้างประวัติวิธีการ

### [P2] Zhang & Nanda (2024) — Best Practices ✅
- **Citation:** Fred Zhang, Neel Nanda. *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods.* ICLR 2024.
- **ลิงก์:** https://arxiv.org/abs/2309.16042
- **สรุป:** metric/method ที่เลือกเปลี่ยนไป สรุป interpretability ก็เปลี่ยนตาม; แนะนำ noising vs denoising + logit-diff metric
- **บทบาท:** **ต้องปฏิบัติตามตอนออกแบบ E3** (ใช้ denoising + logit-diff, รายงาน choices ที่เลือก) และอ้างอิงใน Methodology ตรงๆ — reviewers สาย interp จะหา paper นี้

### [P3] Makelov, Lange & Nanda (2024) — Interpretability Illusion ✅
- **Citation:** Aleksandar Makelov, Georg Lange, Neel Nanda. *Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching.* ICLR 2024.
- **ลิงก์:** https://arxiv.org/abs/2311.17030
- **บทบาท:** เหตุผลที่เราทำ patching ที่ระดับ **component เต็มหน่วย (head/MLP output)** ไม่ใช่ subspace ตามอำเภอใจ — เลี่ยง illusion นี้; อ้างใน Methodology

### [P4] Syed, Rager & Conmy (2024) — Attribution Patching ✅ (⚠️ ยืนยัน arXiv ID)
- **Citation:** Aaquib Syed, Can Rager, Arthur Conmy. *Attribution Patching Outperforms Automated Circuit Discovery.* Proceedings of the 7th BlackboxNLP Workshop @ EMNLP 2024.
- **ลิงก์:** ACL Anthology: 2024.blackboxnlp-1.18 · arXiv ~2310.13548 (ยืนยัน ID)
- **บทบาท:** ตัวเลือก screening ใน E3 ถ้า sweep เต็มช้า; อ้างถ้าใช้

### [P5] Wang et al. (2023) — IOI Circuit in GPT-2 Small ✅
- **Citation:** Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, Jacob Steinhardt. *Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 small.* ICLR 2023.
- **ลิงก์:** https://arxiv.org/abs/2211.00593
- **บทบาท:** ต้นแบบงาน circuit เต็มรูปแบบ; อ้างเทียบว่าเราทำแบบ coarse-grained กว่า (เหมาะกับ scope Q3–Q4)

### [P6] Conmy et al. (2023) — ACDC ✅
- **Citation:** Arthur Conmy, Augustine N. Mavor-Parker, Aengus Lynch, Stefan Heimersheim, Neel Nanda. *Towards Automated Circuit Discovery for Mechanistic Interpretability.* NeurIPS 2023.
- **ลิงก์:** https://arxiv.org/abs/2304.14997
- **บทบาท:** automated circuit discovery — อ้างว่าเป็นทางเลือกที่หนักกว่างานเรา (เราใช้ targeted sweep ตาม RQ)

### [P7] Goldowsky-Dill et al. (2023) — Path Patching ✅ (⚠️ ยืนยัน venue)
- **Citation:** Lucas Goldowsky-Dill, Nicholas Lee, Peter Sato, Neel Nanda. *Localizing Model Behavior with Path Patching.* 2023.
- **ลิงก์:** https://arxiv.org/abs/2304.05969
- **บทบาท:** วิธีของ E7 (stretch goal)

### [P8] Geiger et al. (2021) — Causal Abstractions ✅
- **Citation:** Atticus Geiger, Hanson Lu, Thomas Icard, Christopher Potts. *Causal Abstractions of Neural Networks.* NeurIPS 2021.
- **บทบาท:** ฐานทฤษฎีของ interchange intervention — อ้างสั้นใน Methodology

### [P9] Elhage et al. (2021) — Transformer Circuits Framework ✅
- **Citation:** Nelson Elhage, Neel Nanda, Catherine Olsson, ... Chris Olah. *A Mathematical Framework for Transformer Circuits.* Transformer Circuits Thread, 2021.
- **ลิงก์:** https://transformer-circuits.pub/2021/framework/index.html
- **บทบาท:** นิยาม/สัญลักษณ์ (residual stream, head output z) ที่ใช้ทั้ง paper; ต้นทางของ TransformerLens

### [P10] Lieberum et al. (2023) — Circuits ที่ Scale (Chinchilla) ✅
- **Citation:** Tom Lieberum, Matthew Rahtz, János Kramár, Neel Nanda. *Does Circuit Analysis Interpretability Scale? Evidence from Multiple Choice Capabilities in Chinchilla.* 2023.
- **ลิงก์:** https://arxiv.org/abs/2307.09458
- **บทบาท:** หลักฐานว่าเทคนิค circuit ใช้ข้ามขนาดโมเดลได้ — สนับสนุนความชอบธรรมของ cross-model comparison ของเรา

### [P11] Michel, Levy & Neubig (2019) — Head Pruning ✅
- **Citation:** Paul Michel, Omer Levy, Graham Neubig. *Are Sixteen Heads Really Better than One?* NeurIPS 2019.
- **ลิงก์:** https://arxiv.org/abs/1905.10650
- **บทบาท:** ประวัติ intervention แบบปิด heads ที่ "ไม่จำเป็น" — เทียบกับการปิด heads "ตัวการ" ตามที่ patching ชี้

### [P12] McDougall et al. (2023) — Copy Suppression ✅
- **Citation:** Callum McDougall, Arthur Conmy, Cody Rushing, Tom McGrath, Neel Nanda. *Copy Suppression: Comprehensively Understanding an Attention Head.* NeurIPS 2023 (เวอร์ชันขยาย: BlackboxNLP 2024).
- **ลิงก์:** https://arxiv.org/abs/2310.04625 · https://aclanthology.org/2024.blackboxnlp-1.22/
- **สรุป:** head L10H7 ใน GPT-2 Small ทำหน้าที่เดียวทั้ง distribution: **กดการ copy token ที่เพิ่งปรากฏใน context** เพื่อ calibration — อธิบายปรากฏการณ์ "name mover heads" ที่ robust แปลกๆ ในงาน IOI
- **บทบาทต่องานเรา:** พื้นฐานของกลไก copy/suppress ระดับ head — สำคัญต่อการตีความ E4 ของเราโดยตรง (บริบทหลอกล่อของเรามี distractor token อยู่ใน context พอดี จึงมี copy/suppress heads เข้ามาเกี่ยวข้องแน่นอน)

### [P13] Zheng et al. (2025) — Attention Heads Survey (Patterns) ✅
- **Citation:** Zheng et al. *Attention Heads of Large Language Models: A Survey.* Patterns (Cell Press), 2025.
- **ลิงก์:** https://arxiv.org/abs/2409.03752 · https://www.cell.com/patterns/fulltext/S2666-3899(25)00024-8
- **สรุป:** survey ครอบคลุมงานระบุหน้าที่ของ attention heads ทั้งหมด — เสนอ framework 4 ระยะ: **Knowledge Recalling, In-Context Identification, Latent Reasoning, Expression Preparation** และแบ่งวิธีค้นหา heads เป็น Modeling-Free vs Modeling-Required
- **บทบาทต่องานเรา:** (1) โครงกระดูกของ Related Work สาย head-analysis, (2) ใช้ framework 4 ระยะเป็น **taxonomy จัดกลุ่ม heads ที่เราพบใน E4** (โดยเฉพาะระยะ Knowledge Recalling vs In-Context Identification ซึ่งตรงกับการแข่งขัน memory-vs-context ของเราพอดี) — ทำให้ผลของเราเชื่อมกับสายวิชาการหลักได้; อ้างใน Introduction ได้ด้วยว่า head-level analysis เป็นพื้นที่ที่มีการจัดระเบียบแล้ว แต่ช่อง CM-conflict ใน SLMs ยังว่าง

---

## 6. Inference-time Interventions / Steering

### [S1] Rimsky et al. (2024) — Contrastive Activation Addition ✅ (⚠️ ยืนยัน main vs Findings)
- **Citation:** Nina Rimsky, Gabi Ryzman, Alexander Turner, ... Neel Nanda. *Steering Llama 2 via Contrastive Activation Addition.* ACL 2024.
- **ลิงก์:** https://arxiv.org/abs/2312.06681
- **บทบาท:** สาย steering ด้วย vector จาก contrastive pairs — ทางเลือกของ intervention ที่ควรพูดถึงใน Related Work

### [S2] Zou et al. (2023) — Representation Engineering ✅
- **Citation:** Andy Zou, Long Phan, Sarah Chen, ... Bo Li, ... Dan Hendrycks. *Representation Engineering: A Top-Down Approach to AI Transparency.* 2023.
- **ลิงก์:** https://arxiv.org/abs/2310.01405
- **บทบาท:** กรอบ top-down ของการอ่าน/ควบคุม representation (รวมความจริง) — วางตำแหน่งว่าเราเป็น bottom-up ระดับ component

### [S3] TruthX (ACL 2024) ✅ (⚠️ ยืนยันรายชื่อผู้แต่ง)
- **Citation:** Chen Zhang et al. *TruthX: Alleviating Hallucinations by Editing Large Language Models in Truthful Space.* ACL 2024, pp. 8908–8949.
- **ลิงก์:** https://arxiv.org/abs/2402.17811
- **สรุป:** สกัด "truthful space" (semantic subspace) จาก activation space แล้ว edit/steer activation ใน subspace นั้นตอน inference เพื่อลด hallucination โดยไม่ retrain
- **บทบาทต่องานเรา:** intervention ระดับ **subspace ที่เรียนรู้** (ต่างจาก ITI ที่ใช้ probe direction) — อ้างเทียบใน Related Work ว่าเราทำ intervention ระดับ **component ที่ถูกเลือกด้วย causal criteria** ไม่ใช่ subspace ที่ต้อง train แยก; งานทบทวนโดย ITI/TruthX ช่วยตอบ reviewer ที่ถาม "ทำไมไม่ใช้ steering vector"

---

## 7. โมเดล ชุดข้อมูล และเครื่องมือ

### [T1] Gemma Team (2024) ✅
- **Citation:** Gemma Team (Gemma 2 Authors). *Gemma 2: Improving Open Language Models at a Practical Size.* 2024.
- **ลิงก์:** https://arxiv.org/abs/2408.00118
- **บทบาท:** โมเดลหลัก; อ้างสเปกสถาปัตยกรรม (26 layers/8 heads สำหรับ 2B) + softcapping (จุดที่ต้องระวังใน E0)

### [T1b] Gemma Team (2025) — Gemma 3 ✅
- **Citation:** Gemma Team. *Gemma 3 Technical Report.* 2025 (มีนาคม 2025).
- **ลิงก์:** https://arxiv.org/abs/2503.19786
- **สรุป:** 1B–27B, multimodal (ยกเว้นขนาดเล็กที่ text-only), 128k context · ⚠️ หมายเหตุ (ตรวจแล้ว ต.ค. 2026): ขนาด 1B บน HF มีเฉพาะแบบ `gemma-3-1b-it` — **ไม่มีตัว base 1B** ใน org google (base มีเฉพาะทาง Arm AI Portal)
- **บทบาทต่องานเรา:** **ทางเลือกโมเดลรุ่นใหม่กว่า Gemma-2** — ดู §1 ของแผนหลัก; ถ้าใช้ ต้อง apply chat template ให้สม่ำเสมอ + ระบุใน paper ว่าเป็นตัว instruction-tuned; การสนับสนุนใน TransformerLens เริ่มจาก GitHub issue #898 (มี.ค. 2025) — เช็คสถานะจริงใน E0

### [T1c] Gemma Team (2026) — Gemma 4 ✅
- **Citation:** Gemma Team. *Gemma 4 Technical Report.* 2026 (เปิดตัว 2 เมษายน 2026). arXiv:2607.02770 (⚠️ ยืนยัน ID/ชื่อเต็มตอนทำ bibliography).
- **ลิงก์:** https://arxiv.org/html/2607.02770v1 · https://ai.google.dev/gemma/docs/core · https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
- **สรุป:** สร้างบนเทคโนโลยีเดียวกับ Gemini 3 · ขนาด E2B, E4B, 12B, 31B, 26B-A4B (MoE) · context 256k · multimodal · ใช้ license **Apache 2.0** (ไม่ gated แล้ว)
- **บทบาทต่องานเรา:** ไม่นำมาทดลอง แต่ใช้สองที่: (1) **เหตุผลเลือกโมเดลใน Related Work** — เลือก Gemma-2 เพราะ tooling maturity + Gemma Scope + literature comparability ไม่ใช่ความใหม่ (รุ่นเล็กสุดของ Gemma 4 เป็นสถาปัตยกรรมแนว edge/effective-params ที่ tooling mech interp ยังไม่รองรับ), (2) **Future work** — ทำซ้ำ protocol บน Gemma 4 เมื่อ tooling พร้อม

### [T2] Lieberum et al. (2024) — Gemma Scope ⚠️
- **Citation:** Thomas Lieberum, ... Neel Nanda. *Gemma Scope: Open Sparse Autoencoders All the Way Up in Gemma 2.* NeurIPS 2024 (Datasets & Benchmarks).
- **ลิงก์:** https://arxiv.org/abs/2408.05147 (ยืนยัน ID/ชื่อเต็ม)
- **บทบาท:** artifact ของ Gemma 2 — อ้างเป็น future work (สร้าง SAE-based analysis ต่อจากเรา); หมายเหตุ: Google ปล่อย **Gemma Scope 2** (ธ.ค. 2025) สำหรับ Gemma 3 แล้ว — เหตุผลเสริมว่าการเลือกตระกูล Gemma ทำให้ต่อยอดด้วย SAE ได้สะดวก

### [T3] Grattafiori et al. (2024) — Llama 3 ✅
- **Citation:** Aaron Grattafiori et al. (Meta). *The Llama 3 Herd of Models.* 2024 (+ Llama 3.2 model card สำหรับ 1B/3B).
- **ลิงก์:** https://arxiv.org/abs/2407.21783
- **บทบาท:** โมเดลเทียบข้ามสถาปัตยกรรม

### [T4] Qwen Team (2024–2025) — Qwen2.5 / Qwen3 ✅ (⚠️ ยืนยัน ID ของ 2.5)
- **Citation:** Qwen Team. *Qwen2.5 Technical Report.* 2024 (arXiv:2412.15115) · Qwen Team. *Qwen3 Technical Report.* 2025 (arXiv:2505.09388, พ.ค. 2025).
- **ลิงก์:** https://arxiv.org/abs/2412.15115 · https://arxiv.org/abs/2505.09388
- **บทบาท:** โมเดลเทียบข้ามสถาปัตยกรรม; **ข้อควรระวัง Qwen3:** เป็น hybrid thinking models — ต้องปิด thinking mode (`enable_thinking=False` / `/no_think`) ให้เป็น non-thinking ก่อนทำ patching ไม่งั้นเปรียบเทียบยาก; ทางเลือกที่เรียบง่ายกว่าคือใช้ Qwen2.5-1.5B

### [T5] Bloom (2022–) — TransformerLens ✅ (คำเตือน softcapping ยืนยันแล้ว)
- **Citation:** Joseph Bloom. *TransformerLens.* GitHub repository — https://github.com/TransformerLensOrg/TransformerLens
- **บทบาท:** เครื่องมือ hooking หลัก; อ้าง version ที่ใช้จริงใน repo
- **⚠️ คำเตือนที่ตรวจแล้ว (2026):** มี bug report ว่า outputs ของ HookedSAETransformer/HookedTransformer กับ HF "behave very slightly differently" บน Gemma-2 — สาเหตุ: LayerNorm folding/`center_unembed` ทำลาย invariance ของ tanh softcap, dtype ผสม float32/bfloat16, และตำแหน่งที่ apply softcap ไม่ตรงกัน — **แนวปฏิบัติ:** ใช้รุ่นล่าสุด (default อนุรักษ์ raw HF weights), `center_unembed=False`, dtype คงที่, เทียบ post-softcap logits — และรัน parity check ตาม E0 เสมอ

### [T9] Wu et al. — pyvene (ICLR 2025) ✅ (⚠️ ยืนยันรายชื่อผู้แต่งเต็ม)
- **Citation:** Zhengxuan Wu, Atticus Geiger, Christopher Potts, et al. *pyvene: A Library for Understanding and Improving PyTorch Models via Interventions.* ICLR 2025 (มีเวอร์ชันก่อนหน้า ~2024 ที่ NAACL/Findings).
- **ลิงก์:** https://github.com/stanfordnlp/pyvene
- **สรุป:** framework แบบ declarative สำหรับ intervention บน PyTorch/HF models — ทำ **interchange intervention (activation patching)** และ activation steering ได้โดยไม่ต้อง re-implement โมเดล
- **บทบาทต่องานเรา:** **fallback อันดับแรกของ E0** เมื่อ TransformerLens parity ไม่ผ่าน — ทำงานกับ HF Gemma-2 โดยตรงจึงไม่มีปัญหาการแปลง weight; ยังใช้ระบุ intervention spec แบบ reproducible ได้

### [T10] NNsight / NDIF (ICLR 2025) ⚠️
- **Citation:** *Democratizing Access to Open-Weight Foundation Model Internals.* ICLR 2025. (ชื่อเต็ม/ผู้แต่งต้อง verify เพิ่ม)
- **ลิงก์:** ค้น "NNsight" บน GitHub (ndif-team/nnsight)
- **บทบาท:** ทางเลือกที่สองของ E0 — hooking แบบ scale ได้; ใช้ถ้า pyvene ไม่ fit กับงาน sweep หนัก### [T6] Gao et al. (2021); Biderman et al. (2024) — lm-evaluation-harness ✅
- **Citation:** Leo Gao et al. *A framework for few-shot language model evaluation.* Zenodo, 2021 · Dallas Biderman et al. *Lessons from the Trenches on Reproducible Evaluation of Language Models.* 2024.
- **ลิงก์:** https://github.com/EleutherAI/lm-evaluation-harness · arXiv 2405.14782
- **บทบาท:** เครื่องมือ MMLU/PPL/TruthfulQA ใน E2/E5

### [T7] Li et al. (2023) — HaluEval ⚠️ (ยืนยันรายชื่อผู้แต่งเต็ม)
- **Citation:** Junxian Li, Ximing Li, Fei Wang, et al. *HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models.* EMNLP 2023.
- **ลิงก์:** https://arxiv.org/abs/2305.11747
- **บทบาท:** ตัวเลือก secondary evaluation (QA/knowledge task) ถ้าอยากเพิ่ม dataset นอกเหนือจาก misleading suite + TruthfulQA; เป็น benchmark เดียวกับที่ Lamba et al. [C8] ใช้กับ Gemma — ช่วยให้เทียบผลข้ามงานได้

### [T8] HeadVis (Transformer Circuits, 2026) ✅
- **Citation:** Transformer Circuits Team. *HeadVis: [Interactive tool for investigating attention heads].* Transformer Circuits Thread, 2026.
- **ลิงก์:** https://transformer-circuits.pub/2026/headvis/index.html
- **บทบาท:** เครื่องมือสำรวจ attention heads แบบ interactive ที่รองรับ **Gemma 3** — เป็นไปได้ว่าใช้ประกอบ E4 แทน/ร่วมกับ circuitsvis ได้; อ้างเป็น evidence ว่า ecosystem ของตระกูล Gemma แข็งแรง

---

## 8. รายการที่ต้อง Verify เพิ่มก่อนใส่ Bibliography จริง

| รายการ | สิ่งที่ต้องเช็ค |
|---|---|
| [C9] JuICE (ICML 2025) | **อ่านฉบับเต็ม (สำคัญสุดในลิสต์นี้):** วิธีเลือก heads (influence-based? patching?), โมเดลที่ใช้ (มี sub-2B ไหม), และว่ารายงาน capability trade-off หรือไม่ — ผลลัพธ์กำหนดถ้อยคำ gap สุดท้าย |
| [C13] Sycophancy origins | ผู้แต่งเต็ม + venue + โมเดลที่ใช้ (arXiv:2508.02087) |
| [F7] Campregher | ชื่อเต็มของ paper/ผู้แต่ง + venue (arXiv:2507.11809) |
| [S3] TruthX | รายชื่อผู้แต่งเต็ม (arXiv:2402.17811) |
| [T7] HaluEval | รายชื่อผู้แต่งเต็ม (arXiv:2305.11747) |
| [C7] Choi et al. 2026 | arXiv ID, รายชื่อผู้แต่งเต็ม, สถานะ peer-review (OpenReview) — **สำคัญเพราะ overlap กับ RQ3 มากที่สุด** |
| [P4] Syed et al. | arXiv ID (2310.13548 หรือ 2310.10348) |
| [F3] Geva et al. 2023 | arXiv ID + หน้า ACL Anthology |
| [F6] McGrath et al. | arXiv ID + รายชื่อผู้แต่งเต็ม |
| [H6] INSIDE | รายชื่อผู้แต่ง + arXiv ID + สถานะ ICLR 2025 |
| [K3] DisentQA | arXiv ID + รายชื่อผู้แต่งเต็ม |
| [K4] Context-Aware Decoding | ทั้งรายการ (ผู้แต่ง/ปี/venue) — ยังไม่ผ่านการยืนยันเลย |
| [H4] Azaria & Mitchell | Findings หรือ main conference |
| [S1] CAA | main หรือ Findings |
| [T2] Gemma Scope | ชื่อเต็ม + ID + สถานะ Gemma Scope 2 (ธ.ค. 2025) |
| [C10][C11][C12] (งาน 2026) | ชื่อโมเดลที่ใช้จริงจากฉบับเต็ม (abstract ไม่ระบุ) — สำคัญต่อการเทียบกับ SLMs ของเรา |
| CounterFact dataset | ✅ แก้แล้ว (ต.ค. 2026): ใช้ mirror `wangzn2001/counteract` (counterfact.json ดิบจาก ROME) — สำรอง: `azhx/counterfact`, `NeelNanda/counterfact-tracing` · เหลือเช็คแค่ license ตอนเขียน paper |
| TransformerLens × Gemma 3 | สถานะ issue #898 — เช็คว่ารุ่นที่ติดตั้งรองรับ gemma-3 แล้วหรือยัง ก่อนตัดสินใจใช้ |
| [T1c] Gemma 4 | ยืนยัน arXiv ID (2607.02770) + ชื่อเต็ม report + เช็คว่า E2B เป็น dense หรือ edge-architecture แบบไหนจริง (มีผลต่อ future work ที่เขียน) |
| [T9] pyvene | citation ที่แน่นอน (ICLR 2025 vs เวอร์ชัน NAACL/Findings 2024) + รายชื่อผู้แต่งเต็ม |
| [T10] NNsight/NDIF | ชื่อเต็ม paper + ผู้แต่ง |
| [H9] CLAP | ชื่อเต็ม + ผู้แต่ง + arXiv ID |
| [P13] Zheng survey | รายชื่อผู้แต่งเต็ม + ปีที่ตีพิมพ์ Patterns ยืนยันหน้า |
| SAE-steering × hallucination | sweep ที่ยังทำไม่เสร็จ (โดน rate limit): `"sparse autoencoder" steering hallucination site:arxiv.org` — ถ้าเจองาน 2025–2026 ให้เพิ่มใน §6 |
| ECTI-CIT quartile ปัจจุบัน | เช็คที่ scimagojr.com + CiteScore ล่าสุด (ประวัติมัก Q3 แต่ยังไม่ได้ verify สด) |
| งานใหม่ปี 2026–2027 | สืบค้นซ้ำด้วยคำค้นใน §8.5 ของแผนหลัก ก่อน submit — โดยเฉพาะ "knowledge conflict activation patching" และ "superposition attention heads" |

**เคล็ดลับตรวจเร็ว:** เปิด ACL Anthology / arXiv abs page ของทุกรายการที่จะอ้างจริง แล้วคัดลอก citation จากหน้านั้นโดยตรง (อย่าพิมพ์เองจากความจำ) — Semantic Scholar API (`api.semanticscholar.org/graph/v1/paper/search?query=...`) ช่วยดึงรายชื่อผู้แต่ง/venue แบบ machine-readable ได้
