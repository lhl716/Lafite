# LafitE + LLM Medical Anomaly Detection: Discussion Notes

> These notes condense a research discussion that began with the goal of developing a medical anomaly detection method based on LafitE for a top-tier venue (ICLR/ICML/NeurIPS). Literature review and critical discussion gradually narrowed the direction. **The notes provide the context needed to continue that discussion.** Literature assessments and novelty claims below record the discussion; this translation does not independently revalidate them.

---

## 0. Current direction

Several loosely assembled approaches have been ruled out. The most promising direction identified so far is:
**Knowledge-driven joint anomaly localization and report generation: use an LLM's world knowledge to drive generative anomaly localization. The semantic representation of a proposed `<ANOMALY>` token conditions diffusion reconstruction (producing pixel-level anomalous regions) and autoregressive decoding (producing a diagnostic report), so localization and description are two outputs of the same knowledge-based reasoning process.**

**The decisive unresolved question is the source of training supervision (Section 6).** It should be the first topic in the next discussion.

---

## 1. Project background and goals

- **Base framework, LafitE:** a PyTorch implementation of diffusion-based anomaly detection, originally for industrial datasets MVTec/MPDD. Its three-stage pipeline is:
  1. **Feature extraction:** frozen EfficientNet-b4 produces multiscale features.
  2. **Feature alignment (Neck: MFCN):** multiscale resize and concatenation produce `feature_align` (N×C×32×32, C≈448).
  3. **Diffusion reconstruction (UNetGD):** denoising reconstruction operates in **feature space**, rather than pixel space; the anomaly score is `L2(feature_align - feature_rec)`.
  - **Feature Editing:** a Faiss memory bank (kCenterGreedy coreset sampling + JL random projection) retrieves KNNs for test features and replaces/edits them toward the normal manifold, addressing the identity shortcut in which diffusion simply copies anomalous inputs.
  - Key files: `lib/models/model_helper.py` (dynamic ModelHelper assembly and `membank_inference`), `lib/models/reconstructions/unetgd.py` (UNet + GaussianDiffusion, 6 DDIM steps, infer_time=960), `lib/models/necks/mfcn.py`, `lib/utils/eval_helper.py`, `main/train_val_diff.py`, and `main/train_val_membank.py`.
  - ModelHelper uses a pipeline pattern: each submodule implements `forward(dict)->dict`, with dynamic imports configured through YAML. Modules are replaceable.

- **Established goals and constraints:**
  - Target **general/cross-modal** medical anomaly detection rather than a single modality. The candidate BMAD benchmark covers five domains: brain MRI, liver CT, retinal OCT, chest X-ray, and pathology.
  - Pursue **incremental innovation on LafitE**, with a substantive **LLM/Agent contribution**.
  - Target ICLR/ICML/NeurIPS. Contributions may involve methods/architectures, new problem settings, theory, or interpretability.
  - Resource constraints remain unspecified.
  - **Deep integration is strongly preferred to a loosely assembled pipeline:** removing the LLM must materially reduce detection performance. It cannot serve only as a post-hoc explainer or report generator.
  - The user expects rigorous criticism of weak contributions, discussion on equal terms, factual accuracy, and no flattery.

---

## 2. Main findings from the literature discussion (2023–2026)

### 2.1 Medical anomaly detection with LLMs/Agents is already an active area

The initial impression that the area was almost empty was incorrect. Important competitors identified in the discussion include:

- **MEDIC-AD** (CVPR 2026 Oral): a three-stage VLM pipeline (detection → difference reasoning → heatmap) with an `<Ano>` token. **Limitations identified: discriminative embeddings, no reconstruction residuals, no memory bank, and no Agent orchestration.**
- **MedAgent-Pro** (ICLR 2026): a hierarchical agentic workflow (RAG guidelines → visual tools → stepwise verification → evidence-based diagnosis). **Limitations identified: no AD-specific module, reliance on generic pretrained visual tools, and no reconstruction evidence.**
- **MRAD** (ICLR 2026): dual memory banks and CLIP text alignment for zero-shot AD. **Limitations identified: industrial datasets only, CLIP features rather than diffusion reconstruction, and no reasoning chain.**
- **EviScreen** (2026): retrieval of regional evidence from normal and pathological knowledge bases for reasoning-based screening. **Limitations identified: no diffusion reconstruction, language-based retrieval, or Agent orchestration.**
- **MVFA-AD** (CVPR 2024 Highlight): CLIP multilevel adapters and a memory bank for medical AD. **Limitations identified: no LLM, language-based retrieval, reasoning, or generative evidence.** The user considered its lack of an LLM a sufficient point of differentiation.
- Diffusion-based medical AD, including AnoFPDM (WACV'25) and Reflect/Rectified Flow (MICCAI'25), focuses on detection/segmentation without language reasoning.

### 2.2 Gaps identified in the discussion (high confidence at the time)

1. Injecting **diffusion reconstruction residuals/generative evidence** as grounding into MLLM reasoning chains in medical settings was judged unexplored. ICLR'26's *How Do Medical MLLMs Fail?* was cited for evidence of weak visual grounding in medical MLLMs, motivating stronger grounding signals.
2. Medical **cross-modal memory banks with language-based retrieval**, analogous to a medical version of MRAD, were judged incomplete.
3. Agent pipelines with an **unsupervised AD module (diffusion reconstruction) as the perceptual front end** were distinguished from existing medical Agents built around pretrained classification/segmentation tools.

### 2.3 Hypothesis–verification loops are established, not a novelty by themselves

ReAct, Reflexion, Self-Refine, and LATS are general frameworks. Medical examples include MedKGI (information-gain-driven iterative diagnosis) and MedAgent-Pro. AD examples include AnomaMind (time series), AnomalyRuler (video), and CoLLaTe (LLM + specialized detector collaboration, with LLMs better at point anomalies and smaller models better at contextual anomalies).

- **Shared limitation identified as an opportunity:** feedback in existing loops is relatively coarse (yes/no, scalar, Boolean, or textual reflection). The discussion found no example using **structured spatial residual distributions as feedback**, suggesting an information-theoretic argument that higher feedback bandwidth per round could yield faster convergence.

### 2.4 Candidate theoretical foundations (Eric's literature review)

- **Information-theoretic anomaly scoring:** Rate-Distortion, mutual information, and MDL (anomaly score ∝ MDL increase).
- **Conformal prediction:** distribution-free coverage guarantees and controlled FDR for clinical reliability.
- **Coreset approximation bounds:** formal characterization of the memory-bank compression/accuracy tradeoff.
- **Reasoning-chain faithfulness:** whole-chain reliability ≥ 1−(1−θ)^L.
- **Causal/counterfactual terminology:** the user explicitly rejected labeling normal-reference generation followed by differencing as "counterfactual"; this is already the standard diffusion AD paradigm, so the label was considered unconvincing.

### 2.5 Key observations from EARN (arXiv:2507.00715v1, KDD 2025)

- Figure 2: in **NLP tasks**, attention changes from sparse in shallow layers to dense in deep layers. Deep layers integrate semantics and cannot be pruned aggressively; compressing to k<15 of 32 layers causes collapse. **Recommendation tasks** show the reverse pattern, with dense shallow attention and sparse/redundant deep layers.
- **Implication:** mappings between LLM layers and diffusion steps must be **task-aware**, not uniformly symmetric. Medical AD may combine shallow matching (organ recognition) and deeper reasoning (anomaly discrimination).
- Register tokens may inspire bridge tokens. Integration needs joint training; EARN was cited as showing that architectural changes without training can fail.

---

## 3. How the proposal evolved

| Proposal | Decision | Reason |
|---|---|---|
| A. Agent orchestration (detection → localization → reasoning → report) | De-emphasized | MEDIC-AD/MedAgent-Pro already occupy this space; orchestration alone is not novel |
| B. Cross-modal memory bank + language retrieval | Partly retained | Similar to a medical MRAD; potentially useful as a component, not the central novelty |
| C. Theoretical uncertainty methods such as conformal prediction | Retained as supporting theory | Insufficient as the main contribution alone |
| D. One-step diffusion + feature editing for efficiency | Rejected | Weak LLM/Agent integration and inconsistent with the user's priorities |
| Evidence tokenization by region/scale/channel | Rejected | Genuine disentanglement cannot be assumed; extra losses would divert the project |
| Feeding MLLMs a query + nearest normal reference + residual triplet | Rejected | **Weak gradients to the visual encoder, difficulty learning CT detail, and high medical-image similarity** create a ceiling already observed in the user's earlier experiments |
| Annotating evidence with p-values | Rejected | An LLM is not a statistical calculator and may not use the p-values |
| LLM as a context encoder for reconstruction | Rejected | Input embeddings bypass the backbone; a full LLM used only for embeddings is an unnecessarily heavy text encoder |
| Semantic feature editing through memory-bank routing | Reduced to a component | Too engineering-oriented to support the main contribution |
| **LLM as a reconstruction-policy controller** | **Rejected** | **The user argued that this reduces the role to work a ControlNet-sized adapter could do, fails to use the LLM's world knowledge, resembles RL without step-level supervision, and adds instability and excessive engineering** |

### Turning point: the user's central position

- The LLM should function as a **knowledge expert with world knowledge**, beyond encoding, control, or label provision.
- **Introducing a large model merely to obtain a text label is an unacceptably weak contribution.**
- **The central aim is joint generation of anomalous regions and reports**, with both outputs grounded in the same knowledge-based reasoning.

---

## 4. Current direction in detail

### 4.1 Core formulation

Use LLM world knowledge to **drive generative anomaly localization**. The semantic representation of an `<ANOMALY>` token (or special embedding) simultaneously:

- **Conditions diffusion reconstruction**, producing a pixel-level anomaly mask or a normal version of the region for differencing.
- **Supports autoregressive decoding**, producing a diagnostic report.

Localization and description thus become two outputs of the same reasoning process.

### 4.2 Complementary roles

- **LLM:** supplies world knowledge to determine whether a finding is medically anomalous and what it is, beyond what a small model or visual distance alone can provide.
- **Diffusion reconstruction (LafitE):** supplies precise pixel-level spatial generation, grounding semantic anomaly concepts in masks where the LLM's own grounding is weak.
- **Integration point:** a shared token/embedding. Diffusion residuals feed back into the LLM for visual grounding and correction of its description.

### 4.3 Distinction from earlier work

- **LISA** is an analogy: its `<SEG>` token lets an MLLM drive SAM for reasoning segmentation. The proposed distinction is:
  - LISA uses **discriminative segmentation (SAM)**; this proposal uses **generative reconstruction**, which is crucial for unsupervised/few-shot AD where anomaly annotations are unavailable and normal reconstruction provides a reference.
  - LISA targets generic segmentation; here the mask comes from regions of reconstruction failure and requires diffusion.
- For **ControlNet**, the condition would be inferred by the LLM from whole-image understanding and world knowledge rather than supplied as an external control map. **The same representation must also generate the report**, creating dual decoding. **Whether this is a sufficiently fundamental distinction to convince reviewers remains an open question for the user.**

### 4.4 Optional theoretical foundations

1. Information theory: each feedback round contains a high-dimensional spatial residual, potentially providing more information than a conventional one-bit loop and faster convergence if iteration is retained.
2. Conformal prediction: statistical coverage guarantees or controlled FDR for final localization.
3. Task-dependent layer/step mapping inspired by EARN as an empirical basis for architectural design.

### 4.5 Useful motivation experiment and baseline

**Direct end-to-end MLLM fine-tuning on CT** can serve as a motivation experiment and baseline. The user's existing observations of weak gradients, difficulty learning CT details, and high cross-modal similarity motivate a dedicated integration mechanism rather than relying on direct fine-tuning.

---

## 5. Candidate LafitE–LLM integration architectures (not finalized)

Three possibilities were considered:

1. **A shared memory bank as working memory:** elegant, but relatively engineering-oriented.
2. **A unified feature space with bidirectional projections:** the option the user found interesting, provided the need for alignment is justified.
3. **Interleaved forward passes:** intervention at key decision points.

**Motivation for feature-space alignment:** in a cross-modal unified model, a KNN memory bank based only on visual distance can suffer **cross-domain contamination**. A lung CT patch might be closer to a brain MRI patch than to the correct normal prior, causing Feature Editing and reconstruction to fail. A semantically structured feature space, for example one initialized with BiomedCLIP's language-aligned visual encoder, could encode domain boundaries geometrically and constrain retrieval. **Simply replacing the backbone with CLIP and adding text-guided retrieval would overlap with MVFA-AD; a stronger distinction is needed.**

**EARN-inspired asymmetric coupling**, an early draft potentially superseded in part by joint generation:

- Coarse diffusion steps → shallow LLM layers for perception/organ recognition.
- Fine diffusion steps ← deep LLM layers for semantically guided reconstruction.
- The first half passes visual information to language; the second passes language information to vision.

---

## 6. Unresolved questions to prioritize

### Question 1: Where does training supervision come from?

What data is required for joint mask and report generation?

- Paired pixel-level anomaly annotations and reports are extremely scarce in medicine and could be a critical obstacle.
- Under unsupervised training on normal cases only, how can the LLM learn to generate anomaly hypotheses?
- **Leader's tentative preference, requiring validation:** self-supervised reconstruction on normal cases, plus language-path fine-tuning with a small number of report-bearing cases. The interaction between the two paths and emergence of anomaly concepts without annotations still need to be explained.

### Question 2: Is anomaly-token-driven diffusion fundamentally different enough from ControlNet conditioning?

The distinction needs to be assessed against likely reviewer objections, especially the user's concern that the method could collapse into a ControlNet-like approach.

### Question 3: Should the overall model retain an iterative loop?

Keeping a loop could support an argument about higher-bandwidth residual feedback and faster convergence. Removing it simplifies the architecture but loses a potential theoretical contribution.

### Question 4: What LLM size and freezing strategy should be used?

Avoid treating the LLM merely as a heavy encoder. The tentative strategy is a frozen backbone with LoRA/lightweight adapters and trained projection/bridge layers.

---

## 7. Key references by topic

- **Diffusion AD:** DiAD (AAAI'24), GLAD (ECCV'24), OmiAD (ICML'25, one-step diffusion), One-for-More (CVPR'25, continual learning), VPDM (ICML'24, fuzzy prototypes to avoid shortcuts), Diffusion Implicit Guidance (MICCAI'24), Patched Diffusion (MIDL'24).
- **Memory banks/features:** MRAD (ICLR'26), ReMem (CVPR'26), ADPretrain (NeurIPS'25), PatchCore (coreset theory).
- **VLM/zero-shot AD:** WinCLIP (CVPR'23), AnomalyCLIP (ICLR'24), MVFA-AD (CVPR'24), AA-CLIP (CVPR'25), AnomalyGPT (AAAI'24), LAFT (ICLR'25), Anomaly-OV (CVPR'25), AnomalyR1 (2025, GRPO), ADSeeker (CVPR'26).
- **Medical VLMs/Agents:** LLaVA-Med (NeurIPS'23), MEDIC-AD (CVPR'26), MedAgent-Pro (ICLR'26), MedRAX (ICML'25), EviScreen (2026), MediCLIP (MICCAI'24), MadCLIP (MICCAI'25), Ano-NAViLa (ICCV'25).
- **Loops/reasoning:** ReAct, Reflexion, Self-Refine, LATS, MedKGI, AnomaMind, AnomalyRuler, CoLLaTe, VisProg (CVPR'23).
- **Grounded segmentation:** LISA (reasoning segmentation with a `<SEG>` token).
- **Theory:** Rate-Distortion for Machines (2023), Likelihood-based OOD in representation space (2025), conformal prediction for medical segmentation, AR-Pro (NeurIPS'24).
- **LLM layer mechanisms:** EARN (arXiv:2507.00715v1, KDD'25).
- **Benchmarks:** BMAD (CVPRW'24, medical AD across five domains), NOVA (NeurIPS'25, rare brain MRI pathology and clinical reasoning), and 3D medical UAD benchmarks.
- **Motivation:** *How Do Medical MLLMs Fail?* (ICLR'26), cited for limitations in medical MLLM grounding.

---

## 8. Discussion conventions

- The user expects discussion on equal terms, sharp criticism, factual accuracy, and no flattery. The discussion lead should identify weaknesses proactively and may defend an independent assessment.
- The user questions components from first principles, including why a simpler method cannot replace them. New proposals should explain why each central component is indispensable.

---

*These are condensed discussion notes, not a final specification. Continue with Question 1 in Section 6: training supervision.*
