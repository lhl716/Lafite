# LafitE × MLLM 医学诊断实验

本目录实现三项 **go/no-go 诊断实验**，用于在设计完整融合模型之前检验三个前提：

1. 病灶局部信息是否在视觉编码器、projector 或 LLM 中逐步变得不可解码；
2. 异常输出是否因果依赖病灶以外的非临床干扰；
3. LLM 是否在有视觉证据时提供小模型无法替代的组合知识。

这些程序不是论文结果本身。正式结论必须来自真实医学数据、患者级划分、锁定测试集和患者聚类统计。

## 1. 安装与快速自检

原 LafitE 训练依赖仍在 `requirements.txt`；诊断核心及 NIfTI、DICOM、MLLM 适配依赖在 `requirements-medical.txt`。完整 LafitE 训练应在相应 CUDA 环境另行安装 `requirements.txt`，因为其中 `faiss_gpu` 不适用于 CPU/macOS 环境。

```bash
pip install -r requirements-medical.txt
bash scripts/medical/run-smoke-suite.sh
```

冒烟脚本会生成非医学合成数组，并依次运行三个实验。结果默认写到 `outputs/medical-smoke/`。合成结果只能验证代码和数据契约，不能进入论文表格。

所有命令也可以独立运行：

```bash
python -m medical_experiments --help
python -m medical_experiments validate-manifest data.jsonl --check-paths
```

## 2. 统一数据清单

实验一、二共用 JSONL 清单，一行表示一个 2D 切片或一个 3D 体积：

```json
{"sample_id":"case001-z042","patient_id":"case001","image":"/data/case001.nii.gz","mask":"/data/case001-mask.nii.gz","slice_index":42,"depth_axis":2,"split":"train","modality":"CT","label":1}
```

必需字段：

- `sample_id`：样本唯一标识；
- `patient_id`：患者唯一标识；
- `image`：NPY、NPZ、NIfTI、普通图像或 DICOM 序列目录；
- `split`：`train`、`val`、`test`、`calibration` 或 `external`。

定位任务还需要 `mask`。可选字段包括：

- `image_key`、`mask_key`：NPZ 中的数组键；
- `slice_index`、`depth_axis`、`mask_depth_axis`：从体积抽取切片；
- `normal_reference`：配准后的正常参考；
- `nuisance_mask`：预先定义的非临床干扰区域；
- `placebo_mask`：人工审核过的等面积安慰剂区域；
- `placebo_allowed_mask`：限制自动/显式 placebo 只能落在同器官或同组织的区域；
- `organ`、`region`、`finding`、`report`：实验三语义字段。

必须先检查泄漏：

```bash
python -m medical_experiments validate-manifest /path/to/manifest.jsonl \
  --check-paths --require-mask
```

如果同一 `patient_id` 出现在多个 split，程序直接失败。需要新建划分时：

```bash
python -m medical_experiments split-patients raw.jsonl split.jsonl \
  --train 0.7 --val 0.1 --test 0.2 --stratify modality --seed 42
```

切片不能作为统计独立样本；训练随机种子也不能作为临床样本量。

## 3. 医学预处理约束

`medical_io.py` 支持：

- CT：固定 HU 窗并映射到 `[0,1]`；
- MRI：前景内稳健百分位裁剪和 z-normalization；
- 2.5D：用相邻切片组成三通道；
- NPY/NPZ、NIfTI、常见二维图像和 DICOM 序列。

正式比较中，所有 representation stage 必须使用同一 CT 窗、MRI 序列和切片规则。建议把 224、448 和原生滑窗分别缓存，避免把 resize 损失误写成“语义对齐损失”。

## 4. 实验一：局部细节可解码性

### 4.1 先缓存冻结模型表示

LafitE：

```bash
python -m medical_experiments export-lafite \
  --config experiments/medical-diagnostics/export-lafite.example.yaml
```

缓存包含：

- `feature_align`；
- `feature_rec`；
- `lafite_residual = abs(feature_align - feature_rec)`；
- `lafite_pred`；
- GT `mask` 和患者元数据。

任意 MLLM 通过 factory 与 forward hook 接入：

```bash
python -m medical_experiments export-mllm \
  --config experiments/medical-diagnostics/export-mllm.example.yaml
```

factory 接口为：

```python
def build_bundle(kwargs):
    return {
        "model": frozen_model,
        "preprocess": lambda image_hwc, record: model_inputs,
        # 可选，缺省为 model(**model_inputs)
        "forward": lambda model, model_inputs: model(**model_inputs),
        # 可选，用于无法通过 module hook 取得的输出
        "extract_output": lambda output: {"llm_deep": output.hidden_states[-1]},
    }
```

配置中的 `taps` 指定真实模块名。例如分别挂接 vision early、vision last 和 projector。不同 MLLM 的内部模块名不同，因此仓库不硬编码某个 LLaVA/Qwen/InternVL 版本。

默认会在调用 MLLM preprocess/prompt builder 前移除 `finding`、`report`、`label`、GT mask 等目标字段，避免表征缓存本身已经被标签条件化。

合并两类缓存：

```bash
python -m medical_experiments merge-features \
  --config experiments/medical-diagnostics/merge-features.example.yaml
```

合并严格按 `sample_id`，并检查 `patient_id` 和 `split` 一致。任何样本集合不一致默认失败。

### 4.2 训练等容量 probe

```bash
python -m medical_experiments detail-probe --config /path/to/exp1.yaml
```

核心配置：

```yaml
cache_paths: [/path/to/merged-cache]
stage_keys:
  vision_early: vision_early
  vision_last: vision_last
  projector: projector
  llm_deep: llm_deep
  residual: lafite_residual
stage_layouts:
  vision_early: NLC
  vision_last: NLC
  projector: NLC
  llm_deep: NLC
  residual: NCHW
stage_grids:
  vision_early: [16, 16]
  vision_last: [16, 16]
  projector: [16, 16]
  llm_deep: [16, 16]
prefix_tokens:
  vision_early: 1
  vision_last: 1
target_grid: [32, 32]
channel_budget: 64
channel_projection:
  type: orthogonal
  seed: 2069
feature_sets:
  vision_early: [vision_early]
  vision_last: [vision_last]
  projector: [projector]
  llm_deep: [llm_deep]
  projector_plus_residual: [projector, residual]
probe:
  type: linear
  epochs: 50
  threshold: auto
spatial_shuffle_control:
  enabled: true
coordinate_control:
  enabled: true
  include_radius: true
random_feature_control:
  enabled: true
coarse_label_key: modality
```

支持 NCHW dense feature 和 NLC token。NLC 必须能还原空间网格；CLS/register token 用 `prefix_tokens` 去除。mask 使用 nearest resize，feature 使用 bilinear resize。

程序报告：

- sample/patient Dice；
- positive-only patient Dice；
- pixel AUROC/AUPRC；
- 小病灶 component sensitivity；
- 训练集选择的固定阈值；
- residual 空间打乱后重新训练同容量 probe 的对照结果。
- coordinate-only 与匹配 train 一阶统计的 random-feature 负对照；
- 可选的 GAP + 线性 coarse-label 正对照，检查深层粗语义是否仍可读。

科学解释：

- 早层好、projector 后差：支持 connector bottleneck；
- 只有 224 差、448 恢复：优先归因分辨率；
- 各层 probe 相近、最终决策差：信息仍存在，问题在利用而非编码；
- `projector + residual` 提升且打乱 residual 后消失：支持 LafitE 提供条件增量空间证据。

建议运行两套配置：一套保留各表示的实际系统能力；另一套统一网格、channel budget 和 probe 容量，隔离机制原因。

## 5. 实验二：病灶与非临床背景干预

### 5.1 生成四个严格配对版本

```bash
python -m medical_experiments make-interventions \
  --config experiments/medical-diagnostics/make-interventions.example.yaml
```

每个样本生成：

- `original`：原图；
- `lesion_removed`：匹配正常参考或局部 Gaussian 修复病灶；
- `nuisance_changed`：只修改病灶膨胀保护区之外的噪声/模糊/风格；
- `placebo`：在病灶外找到同形状、同面积区域并执行相同类型编辑。

正式实验应在 manifest 提供经过审核的 `nuisance_mask`，并设置 `require_nuisance_mask: true`。若省略，代码会扰动整个病灶保护区之外的图像，只适合软件调试，不能被解释成“纯背景因果干预”。

代码不会使用黑块遮挡。输出同时记录：

- 编辑区域内外 MAE；
- protected ROI 的变化；
- 边界梯度 artifact score；
- continuity score；
- 实际变化比例。

若设置 `model_factory`，四幅图立即由同一个冻结模型评分。factory 接口为：

```python
def build_predictor(kwargs):
    return {
        "predict": lambda image, record: {
            "score": anomaly_score,
            "mask": predicted_mask,       # 可选
            "report_entities": ["nodule", "effusion"],  # 可选，结构化实体
        }
}
```

调用 predictor 前默认移除 `finding`、`report`、`label`、`target_entity` 和 `ground_truth`，防止测试标签进入提示或模型输入；这些字段只保留在评估记录中。

自动 placebo 会匹配病灶形状、软 mask 质量和局部强度上下文；正式实验还应在每条 manifest 提供 `placebo_allowed_mask`，把候选位置限制在同器官/同组织。`dilation_radius` 与 `gaussian_sigma` 在 3D 中按 voxel 解释，因此强各向异性 CT/MRI 必须先重采样，或在外部按物理 spacing 换算后再运行。

仓库提供了 LafitE 参考 factory：`medical_experiments.factories.lafite:make_lafite_predictor`。mask threshold 必须来自 calibration split；不提供阈值时只输出异常分数，避免逐图 min-max normalization 伪造稳定性。

`cohort: positive` 生成完整四配对病灶实验；`cohort: normal` 仅生成 original 与 nuisance-changed，用于固定阈值下的正常病例 FPR 漂移。两类患者应分别分析，不能让正常病例的零 lesion response 稀释病灶效应。

### 5.2 患者级配对统计

```bash
python -m medical_experiments shortcut-eval --config /path/to/exp2.yaml
```

输入采用 long-form JSONL：

```json
{"patient_id":"case001","sample_id":"case001-z042","variant":"original","score":0.91,"mask":"/pred/original.npy"}
{"patient_id":"case001","sample_id":"case001-z042","variant":"lesion_removed","score":0.23,"mask":"/pred/removed.npy"}
```

可选写入 `label`、`report_entities`、`target_entity`，并在配置中提供 calibration set 锁定的 `decision_threshold`。程序随后额外计算正常患者 FPR 漂移、报告实体 Jaccard，以及去病灶后目标实体消失率；缺少这些字段时相应终点明确标记为不可评估，不会使用 BLEU 代替。

先在患者内聚合，再 bootstrap 患者。主要输出：

- `variant - original` 的配对分数变化；
- nuisance/placebo 与 original 的 mask Dice；
- `nuisance response / lesion-removal response`；
- 预注册 red-flag 规则及逐项判定。

正式研究应至少使用两种独立病灶修复方法，并用 real-vs-edited artifact detector 或盲法审核确认编辑不是新的快捷方式。病灶周围水肿、占位效应等解剖上下文不能自动归类为“坏背景”。

## 6. 实验三：LLM 必要性

实验三使用独立 JSONL，每行至少包含：

```json
{"patient_id":"case001","organ":"lung","region":"right_upper_lobe","finding":"nodule"}
```

程序先按患者去重，再留出医学上有效的 `organ × finding` 边；每个节点必须仍在训练中单独出现。

### 6.1 导出真实 LLM 的四条件分数

```bash
python -m medical_experiments export-semantic-scores \
  --config experiments/medical-diagnostics/export-semantic-scores.example.yaml
```

factory 接口：

```python
def build_scorer(kwargs):
    def score(record, candidates, evidence_present, knowledge_intact):
        return {candidate: logit for candidate in candidates}
    return score
```

程序强制执行四种条件：

1. image/evidence present + correct knowledge；
2. no image/evidence + correct knowledge；
3. image/evidence present + shuffled knowledge；
4. no image/evidence + shuffled knowledge。

在调用模型前会移除 `finding`、`report`、`label` 等 target 字段；no-image 条件还会移除 image、mask、feature cache 和 region。输出字段分别为 `scores`、`scores_no_image`、`scores_shuffled_knowledge`、`scores_no_image_shuffled_knowledge`。

### 6.2 与强基线统一比较

```bash
python -m medical_experiments llm-necessity --config /path/to/exp3.yaml
```

示例配置：

```yaml
manifest: /path/to/records-with-scores.jsonl
output_dir: /path/to/results/exp3
seed: 42
held_out_edges:
  - [lung, nodule]
top_k: [1, 3, 5]
bootstrap:
  iterations: 2000
  confidence: 0.95
  seed: 2069
baselines:
  - {name: uniform, type: uniform}
  - {name: frequency, type: train_frequency}
  - name: small_embedding
    type: learned_embedding
    kwargs: {embedding_dim: 16, epochs: 80}
  - {name: full_mllm, type: precomputed}
```

输出 `per_sample.jsonl` 和 `summary.json`，指标包括：

- seen/unseen MRR、Recall@k、NLL；
- unseen − seen gap；
- evidence × knowledge 的 2×2 interaction；
- 上述指标、simple effect 与 interaction 的 percentile bootstrap CI；
- patient/edge 泄漏审计。

seen CI 以患者为重采样单位；unseen CI 优先进行 `organ × finding` edge→patient 两级重采样。四个 evidence×knowledge 条件始终在同一患者 block 内一起抽样。若测试集只有一条 held-out edge，edge 层不确定性会退化；这些区间也不涵盖新医院/新设备的外部分布漂移。

这一 runner 对应先做的 **3A：oracle/固定视觉证据下的语义排序**。只有 3A 中 LLM 胜过频率先验和小模型，才值得实现 3B 端到端融合；3B 应让各语义模块使用同一 LafitE/视觉证据并另外报告 Dice、FROC/AUPRO。当前仓库不会在融合模型尚不存在时伪造一个“LLM 定位”结果。

只有当完整 LLM 超过最强小模型，而且增益集中于 unseen edge，并在去图像或打乱知识后消失，才能支持“LLM 对异常检测不可替代”。只改善报告语言风格不能支持这一结论。

## 7. 推荐正式执行顺序

1. 先在 BMAD Brain MRI 与一个 CT mask 数据集上运行实验一；
2. 若 residual 有条件增量，再运行实验二；
3. 前两个 gate 通过后，才在带报告和 grounding 的 CT 数据上运行实验三；
4. 最终对锁定测试集只运行一次，阈值和 red-flag 标准均在 calibration/validation 上确定。

建议每次结果同时保存：Git commit、配置文件、manifest SHA256、数据版本、模型 checkpoint hash、软件版本和随机种子。

## 8. 当前边界

- 仓库没有附带医学数据、模型权重或受限报告；
- MLLM 内部模块名和 prompt 模板由所选模型的 factory 决定；
- 当前 LafitE 是 2D，2.5D 可用于 pilot，但不能替代最终 3D 验证；
- 实验三内置 learned embedding 把 region 当证据代理，只是强控制模型；真实 LLM 必须走四条件 precomputed score；
- feature/intervention cache 可能包含患者衍生信息，已被 `.gitignore` 排除，仍应存放在合规受控目录。
