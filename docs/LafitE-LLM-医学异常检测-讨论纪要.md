# LafitE + LLM 医学异常检测方法论 —— 讨论纪要（供新窗口接续）

> 本文件压缩了一轮完整的头脑风暴讨论：从"基于 LafitE 框架做可发顶会（ICLR/ICML/NeurIPS）的医学异常检测方法"出发，经过多轮文献调研与相互批评，逐步收敛到当前方向。**阅读本文件即可无损接续讨论。**

---

## 0. 一句话现状

我们已排除多个"拼凑式"方案，当前收敛到的最有前景方向是：
**「知识驱动的一体化异常定位与报告生成」—— 让 LLM 的世界知识驱动生成式异常定位，`<异常>` token 的语义表示同时条件化扩散重建（产出像素级异常区域）与自回归解码（产出诊断报告），使"定位"与"描述"成为同一次知识推理的两个出口。**

**该方向的生死取决于一个尚未解决的核心问题：训练监督信号从哪来（见第 6 节）。** 这是新窗口应优先讨论的问题。

---

## 1. 项目背景与目标

- **基础框架 LafitE**：扩散模型异常检测的 PyTorch 实现（工业域 MVTec/MPDD）。三阶段流水线：
  1. **特征提取**：冻结 EfficientNet-b4 → 多尺度特征
  2. **特征对齐（Neck: MFCN）**：多尺度特征 resize+concat → `feature_align` (N×C×32×32, C≈448)
  3. **扩散重建（UNetGD）**：在**特征空间**（非像素空间）做扩散去噪重建；异常分数 = `L2(feature_align - feature_rec)`
  - **核心创新 Feature Editing**：用 Faiss 记忆库（kCenterGreedy coreset 采样 + JL 随机投影）对测试特征做 KNN 检索并"替换/编辑"回正常流形，解决 identity shortcut（扩散直接复制异常输入）。
  - 关键文件：`lib/models/model_helper.py`（ModelHelper 动态组装 + `membank_inference`）、`lib/models/reconstructions/unetgd.py`（UNet+GaussianDiffusion，DDIM 6 步，infer_time=960）、`lib/models/necks/mfcn.py`、`lib/utils/eval_helper.py`、`main/train_val_diff.py`、`main/train_val_membank.py`。
  - ModelHelper 是管道模式：每个子模块 `forward(dict)->dict`，通过 YAML 动态 import 组装，模块可插拔。

- **用户目标与约束（已明确）**：
  - 面向**通用/跨模态**医学异常检测（不锁定单一模态；候选基准 BMAD 覆盖脑MRI/肝CT/视网膜OCT/胸片/病理 5 域）。
  - **增量创新**（在 LafitE 基础上），并且**必须融入 LLM/Agent 创新点**。
  - 冲击 ICLR/ICML/NeurIPS；欢迎方法/架构、新问题设定、理论/可解释性贡献。
  - 资源约束暂不确定。
  - **强烈偏好"深度嵌合"而非"拼凑式 pipeline"**：拿掉 LLM 必须导致检测性能实质下降；LLM 不能只是事后解释器/报告生成器。
  - 用户会犀利批评弱创新点，要求平等讨论、实事求是、不奉承。

---

## 2. 文献调研核心结论（2023–2026）

### 2.1 医学 + LLM/Agent 异常检测**已不是空白**（重要纠正）
早期误判为"几乎空白"，实际已有大量工作。关键竞争者：
- **MEDIC-AD**（CVPR 2026 Oral）：VLM 三阶段（检测→差异推理→热力图），注入 `<Ano>` token。**局限：判别式嵌入、不用重建残差、无记忆库、无 Agent 编排。**
- **MedAgent-Pro**（ICLR 2026）：分层 agentic（RAG 指南→视觉工具→逐步验证→循证诊断）。**局限：无 AD 专用模块、依赖通用预训练视觉工具、无重建证据。**
- **MRAD**（ICLR 2026）：双层记忆库+CLIP 文本对齐做零样本 AD。**局限：仅工业域、用 CLIP 特征非扩散重建、无推理链。**
- **EviScreen**（2026）：正常+病理双知识库检索区域证据→推理筛查。**局限：无扩散重建、无语言检索、无 Agent 编排。**
- **MVFA-AD**（CVPR 2024 Highlight）：CLIP 多级适配器+记忆库做医学 AD。**局限：无 LLM、无语言检索、无推理、无生成证据。**（注：用户认可它"没用 LLM，所以与我们区分度够"）
- 扩散医学 AD：AnoFPDM（WACV'25）、Reflect/Rectified Flow（MICCAI'25）等——**纯检测/分割，不接语言推理。**

### 2.2 仍为真空白的交叉点（高置信）
1. 把**扩散重建残差/生成式证据**作为 grounding 注入 MLLM 推理链（医学域）——没人做。ICLR'26《How Do Medical MLLMs Fail?》指出医学 MLLM 的 visual grounding 严重不足，为"注入强 grounding 信号"提供动机。
2. 医学**跨模态记忆库 + 语言检索**（MRAD 的医学版）——不完整。
3. 以**无监督 AD 模块（扩散重建）为感知前端**的 Agent 流水线——现有医学 Agent 都建立在预训练分类/分割工具上。

### 2.3 "假设-验证循环"范式已成熟（不算新）
ReAct/Reflexion/Self-Refine/LATS 是通用范式；医学有 MedKGI（信息增益驱动迭代诊断）、MedAgent-Pro；AD 有 AnomaMind（时序）、AnomalyRuler（视频）、CoLLaTe（LLM+专用检测器协作，发现 LLM 擅长点异常、小模型擅长上下文异常）。
- **共同局限（潜在机会）**：现有循环的**验证反馈信号都很"粗"**（是/否、标量、布尔、文本反思）；**没人用"结构化空间残差分布"作为反馈**。→ 可提"每轮反馈信息带宽更高 → 收敛更快"的信息论论点。

### 2.4 可强化理论深度的候选支柱（Eric 调研）
- **信息论异常评分**：Rate-Distortion / 互信息 / MDL（异常分数 ∝ MDL 增量）。
- **Conformal Prediction**：分布无关的覆盖保证、可控 FDR（临床可信）。
- **Coreset 近似界**：记忆库压缩-精度权衡的理论刻画。
- **推理链忠实性（faithfulness）**：整链可信度 ≥ 1−(1−θ)^L。
- （因果/反事实——**用户明确否决"反事实"这个词**，认为"生成 normal 对照再做差"就是标准 diffusion AD 范式，贴反事实标签牵强。）

### 2.5 EARN 论文（arXiv:2507.00715v1, KDD 2025）关键洞察
- Figure 2：**NLP 任务**注意力"浅层稀疏→深层密集"（深层做语义整合，不可裁剪，k<15/32 层压缩即崩）；**推荐任务**相反（浅层密集→深层稀疏，深层冗余）。
- **启发**：LLM 层↔扩散步的映射必须**任务感知**；不能对称一刀切。医学异常检测可能是混合任务（"识别器官"偏浅层匹配，"辨别异常"偏深层推理）。
- register token 机制可借鉴为"桥接 token"；嵌合必须配合联合训练（EARN 证明无训练直接改结构会崩）。

---

## 3. 方案演进史（哪些被否决及原因）

| 方案 | 结论 | 否决/保留原因 |
|---|---|---|
| A. Agent 编排（检测→定位→推理→报告） | 弱化 | 已被 MEDIC-AD/MedAgent-Pro 占位；"Agent 编排"本身不新 |
| B. 跨模态记忆库+语言检索 | 部分保留 | 接近 MRAD 医学版；可作组件但非核心 novelty |
| C. 理论不确定性（Conformal 等） | 作为辅助理论支柱保留 | 单独不够撑主线 |
| D. 一步扩散+特征编辑（效率向） | 否决 | LLM/Agent 结合弱，与用户偏好不符 |
| evidence token 化（区域/尺度/通道解耦） | 否决 | 无法保证 token 真解耦，需额外解耦损失，偏离主线 |
| 三元组（query+最像正常+残差）喂 MLLM | 否决 | **梯度到 visual encoder 太弱、CT 细节难学、医学图相似度高** → 存在上限问题（用户前期实验已证实） |
| p-value 标注证据 | 否决 | LLM 不是统计计算器，不会关注 p-value |
| LLM 做 context encoder（条件化重建） | 否决 | 取 input embedding=没用主干；跑完整 LLM 出 embedding=冗余的重型文本编码器 |
| 语义化特征编辑（记忆库语义路由） | 降级为模块 | 创新性偏工程，不足以撑主线 |
| **LLM 作为"重建策略控制器"** | **否决（重要）** | **用户批评：这退化成 ControlNet 的活（几百 M 适配器就能做），没用到 LLM 的世界知识本质；且带 RL 味道、无 step-level 监督、训练不稳定、过度工程** |

### 关键转折：用户的核心主张
- **LLM 的本质是"具有世界知识的知识专家"**，不是编码器、不是控制器、不是标签提供者。
- **"引入大规模模型只为了一个文本标签"是不可接受的弱创新。**
- **真正的重点是"一体化生成异常区域和报告"**——定位与描述应来自同一次知识推理。

---

## 4. 当前收敛方向（详细）

### 4.1 一句话定位
让 LLM 的世界知识**驱动生成式异常定位**：`<异常>` token（或特殊 embedding）的语义表示**同时**：
- **条件化扩散重建** → 产出像素级异常区域（mask / 或该区域的"正常版本"用于做差）
- **自回归解码** → 产出诊断报告
→ 使定位与描述成为同一知识推理的两个出口。

### 4.2 分工（各自不可替代）
- **LLM**：提供世界知识，判断"这在医学上是否异常、是什么异常"（小模型/纯视觉距离做不到）。
- **扩散重建（LafitE）**：提供像素级精确空间生成能力，把 LLM 的"语义级异常概念"落地为精确 mask（LLM 自身 grounding 弱，做不到）。
- **嵌合点**：共享的 token/embedding；扩散残差回传 LLM 做视觉 grounding 与描述修正。

### 4.3 与先例的区分
- 类比 **LISA**（`<SEG>` token 让 MLLM 驱动 SAM 做 reasoning segmentation），但本质不同：
  - LISA 用**判别式分割（SAM）**；我们用**生成式重建**——对无监督/少样本异常检测至关重要（异常无标注，只能"生成正常再做差"）。
  - LISA 是通用分割；我们的 mask 来自"重建失败区域"，需要扩散。
- 与 ControlNet 的区分（用户最敏感）：条件不是外部输入的控制图，而是 LLM **基于全图理解+世界知识推理**出来的，且**同一表示还要负责生成 report**（双重解码）。**⚠️ 此区分是否足够"本质"到说服审稿人，仍需用户判断。**

### 4.4 可叠加的理论支柱（备选，非全部要）
1. 信息论：每轮反馈是高维空间残差 → 信息增益 > 传统 loop 的 1-bit → 收敛更快（如果保留循环机制）。
2. Conformal：最终异常定位的统计覆盖保证 / 可控 FDR。
3. 混合任务的层级-步骤映射（源自 EARN）：可作为架构设计的经验依据。

### 4.5 有价值的副产物（写论文时用）
- **直接端到端微调 MLLM 学 CT = 天然的 motivation experiment + baseline**（用户已有实验证据：梯度弱、CT 细节难学、跨模态相似度高 → 训不动）。这正是我们规避端到端微调、设计嵌合机制的反面证据。

---

## 5. LafitE↔LLM 嵌合的架构候选（尚未定稿）

早期讨论过三种嵌合，逐步聚焦：
1. **共享记忆库作为工作记忆**——最优雅但偏工程。
2. **统一特征空间+双向投影**——用户认为唯一有意思的，但需回答"为什么要对齐"的动机（下述）。
3. **交错式前向传播（Interleaved Forward）**——在关键决策点介入。

**"为什么要对齐特征空间"的动机（已想清）**：跨模态统一模型下，纯视觉距离的记忆库 KNN 会**跨域污染**（肺CT patch 可能在特征距离上更接近脑MRI patch → Feature Editing 用错正常先验 → 重建崩溃）。若特征空间自带语义结构（如用 BiomedCLIP 初始化，其视觉编码器已与语言对齐），跨域边界被几何结构编码，检索天然带语义约束。**但注意：若只是"换 CLIP backbone + 文本引导检索"会撞 MVFA-AD，需更强区分。**

**EARN 启发的不对称耦合（早期草案，可能已被"一体化生成"方向部分取代）**：
- 扩散粗步 → LLM 浅层（感知/器官识别）
- 扩散细步 ← LLM 深层（语义引导精细重建）
- 不对称：前半段"视觉给语言"，后半段"语言给视觉"。

---

## 6. ⚠️ 尚未解决的核心问题（新窗口优先讨论）

### 问题 1（生死攸关）：训练监督信号从哪来？
"一体化生成 mask + report"需要什么数据？
- 若需"像素级异常标注 + 报告"配对数据 → 医学域极稀缺 → 可能是死穴。
- 若无监督（仅正常样本训练）→ LLM 在训练时没见过异常，怎么学会生成异常假设？
- **Leader 的倾向（待验证）**：正常样本自监督训练重建路径 + 少量带报告样本微调语言路径。但需想清楚两条路径如何协同、异常概念如何在无标注下涌现。

### 问题 2：`<异常>` token 驱动扩散 vs ControlNet 条件化，机制区别是否足够本质？
需要用户判断这个区分能否说服审稿人（用户对"退回 ControlNet"最敏感）。

### 问题 3：整体是否保留"迭代循环"？
若保留，可用"高带宽残差反馈→收敛快"的信息论论点；若不保留，架构更简洁但少一个理论点。

### 问题 4：LLM 用多大、冻结策略？
避免"重型编码器"批评。倾向冻结 LLM 主干 + LoRA/轻量适配器 + 训练投影/桥接层。

---

## 7. 关键参考文献（按主题）

- **扩散 AD**：DiAD(AAAI'24)、GLAD(ECCV'24)、OmiAD(ICML'25,一步扩散)、One-for-More(CVPR'25,持续学习)、VPDM(ICML'24,模糊原型避免 shortcut)、Diffusion Implicit Guidance(MICCAI'24)、Patched Diffusion(MIDL'24)。
- **记忆库/特征**：MRAD(ICLR'26)、ReMem(CVPR'26)、ADPretrain(NeurIPS'25)、PatchCore（coreset 理论）。
- **VLM/零样本 AD**：WinCLIP(CVPR'23)、AnomalyCLIP(ICLR'24)、MVFA-AD(CVPR'24)、AA-CLIP(CVPR'25)、AnomalyGPT(AAAI'24)、LAFT(ICLR'25)、Anomaly-OV(CVPR'25)、AnomalyR1(2025,GRPO)、ADSeeker(CVPR'26)。
- **医学 VLM/Agent**：LLaVA-Med(NeurIPS'23)、MEDIC-AD(CVPR'26)、MedAgent-Pro(ICLR'26)、MedRAX(ICML'25)、EviScreen(2026)、MediCLIP(MICCAI'24)、MadCLIP(MICCAI'25)、Ano-NAViLa(ICCV'25)。
- **循环/推理**：ReAct、Reflexion、Self-Refine、LATS、MedKGI、AnomaMind、AnomalyRuler、CoLLaTe、VisProg(CVPR'23)。
- **Grounding 分割**：LISA（reasoning segmentation, `<SEG>` token）。
- **理论**：Rate-Distortion for Machines(2023)、Likelihood-based OOD in representation space(2025)、Conformal Prediction(医学分割)、AR-Pro(NeurIPS'24)。
- **LLM 层级机制**：EARN(arXiv:2507.00715v1, KDD'25)。
- **基准**：BMAD(CVPRW'24,5域医学AD)、NOVA(NeurIPS'25,脑MRI罕见病理+临床推理)、3D医学UAD基准。
- **动机支撑**：《How Do Medical MLLMs Fail?》(ICLR'26,指出医学 MLLM grounding 不足)。

---

## 8. 讨论风格约定（继承到新窗口）
- 用户要求**平等讨论、犀利批评、不奉承、实事求是**；Leader 应主动指出方案弱点，也可坚持自己的判断。
- 用户擅长从第一性原理质疑"为什么需要这个组件/为什么不能用更简单的方法替代"——任何新提案都应预先回答"不可替代性"。

---

*（本纪要为讨论压缩，非最终 spec。新窗口建议从第 6 节的问题 1（训练监督）切入继续。）*
