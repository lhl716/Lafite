# LafitE × Qwen3-VL-8B 三项实验运行手册

这套配置使用 **MLLM**，具体模型默认为
`Qwen/Qwen3-VL-8B-Instruct`。LLM 纯文本模型不能提供实验一所需的视觉层、
projector 和图像 token，因此不适合替代它。

默认四卡分配如下：

| 工作 | 默认设备 |
| --- | --- |
| 实验一 LafitE 特征 | `cuda:0` |
| 实验一 Qwen3-VL 特征 | `cuda:1` |
| 实验二 LafitE 干预评分 | `cuda:2` |
| 实验三 Qwen3-VL 语义评分 | `cuda:3` |

所有设备都可用环境变量覆盖，参见本文末尾。

## 1. 安装

建议在 Linux/CUDA 服务器新建 Python 3.10 或 3.11 环境。不要在当前 macOS
机器下载 8B 权重来验证 GPU 实验。

```bash
conda create -n lafite-med python=3.10 -y
conda activate lafite-med
pip install -r requirements-medical.txt
pip install -r requirements.txt
```

默认注意力实现是 PyTorch SDPA，能直接运行。A100 环境若已正确安装
`flash-attn`，可设置 `QWEN_ATTENTION=flash_attention_2`。

先跑不下载模型的自检：

```bash
bash scripts/medical/run-smoke-suite.sh
```

## 2. 准备 manifest

实验一、二至少需要：

```json
{"sample_id":"case001-z042","patient_id":"case001","image":"/data/case001.npy","mask":"/data/case001-mask.npy","nuisance_mask":"/data/case001-nuisance.npy","slice_index":42,"depth_axis":0,"split":"test","modality":"CT","label":1}
```

实验三还要求每行包含 `organ`、`region` 和 `finding`，并且 evidence-present
条件必须有 `image`。可以使用同一个总 manifest，也可以另设一个语义 manifest。

数据要求：

- 同一 `patient_id` 只能出现在一个 split；
- 实验一必须同时有 train、val、test，并且 train/val 中要有正例 mask；
- 实验二的 test 正例需要 lesion mask，所有参与干预的样本需要预先审核的
  `nuisance_mask`；
- `LAFITE_MASK_THRESHOLD` 与 `LAFITE_SCORE_THRESHOLD` 必须只用 calibration/val
  集确定，不能从 test 集反推；
- CT 窗当前固定为 `[level=40, width=400]`，换器官或协议时应先修改全部配置，
  保证三项实验的预处理一致。

## 3. 设置一次环境变量

```bash
export MEDICAL_MANIFEST=/data/project/manifest.jsonl
export SEMANTIC_MANIFEST=/data/project/semantic-manifest.jsonl
export LAFITE_CHECKPOINT=/data/checkpoints/ckpt_best.pth.tar
export LAFITE_MASK_THRESHOLD=0.25
export LAFITE_SCORE_THRESHOLD=0.50
export MEDICAL_OUTPUT_ROOT=/data/results/lafite-medical
export LAFITE_PYTHON=python
```

如果语义字段已经在 `MEDICAL_MANIFEST` 中，可以不设 `SEMANTIC_MANIFEST`。
如果 Qwen 权重已缓存到本地目录，可覆盖模型 ID：

```bash
export MLLM_MODEL_ID=/data/models/Qwen3-VL-8B-Instruct
```

## 4. 运行三个实验

建议先逐项运行，便于发现数据问题：

```bash
bash scripts/medical/run-qwen3vl-diagnostics.sh exp1
bash scripts/medical/run-qwen3vl-diagnostics.sh exp2
bash scripts/medical/run-qwen3vl-diagnostics.sh exp3
```

数据和阈值都确认后也可以连续运行：

```bash
bash scripts/medical/run-qwen3vl-diagnostics.sh all
```

实验一固定 512×512 输入。Qwen3-VL 缓存的空间表示为：

- vision block 2：`32×32` pre-merger patch grid；
- vision block 26：`32×32` pre-merger patch grid；
- visual merger/projector：`16×16` image-token grid；
- LLM hidden state 18、36：从完整序列中只抽取 `16×16` image token。

实验三不是生成报告，而是计算每个候选 finding 的平均 token 条件对数概率，
并自动运行 image present/absent × knowledge intact/shuffled 四条件。默认知识文本只用
候选名称；正式论文实验应在
`experiments/medical-diagnostics/qwen3-vl-8b/exp3-export-scores.yaml` 的
`factory_kwargs.knowledge` 中预注册候选定义，或改用 `knowledge_file` 指向 JSON：

```json
{
  "nodule": "a focal rounded pulmonary opacity",
  "effusion": "fluid in the pleural space"
}
```

## 5. 查看结果

一条命令打印三项核心指标：

```bash
bash scripts/medical/run-qwen3vl-diagnostics.sh results
```

原始结果文件为：

| 实验 | 主结果 |
| --- | --- |
| 实验一 | `$MEDICAL_OUTPUT_ROOT/exp1/probe-summary.json` |
| 实验二 | `$MEDICAL_OUTPUT_ROOT/exp2/evaluation-summary.json` |
| 实验三 | `$MEDICAL_OUTPUT_ROOT/exp3/evaluation-summary.json` |

解释顺序：

1. **实验一**：比较 `vision_early → vision_last → projector → llm_deep` 的
   positive-patient Dice/AUPRC。持续下降说明局部信息在语义对齐链路中变得不可解码；
   `projector_plus_residual` 或 `llm_deep_plus_residual` 明显回升，且 residual 空间打乱
   后增益消失，才支持 LafitE residual 提供增量局部证据。
2. **实验二**：首先看 shortcut gate 是否为 `pass`。病灶移除应显著降低分数，
   nuisance/placebo 改动应很小；重点检查 nuisance/lesion ratio、正常病例 FPR 增幅
   和所有 triggered red flags。
3. **实验三**：只把 unseen `organ×finding` 作为关键结果。Qwen3-VL 应超过
   frequency 与 small_embedding；同时 image removal 和 knowledge shuffle 应让性能下降，
   evidence×knowledge interaction 的 MRR 最好为正且置信区间不跨零。

这些 gate 不能仅凭一个点估计判定。患者数、bootstrap CI、编辑质量和预注册阈值必须
一起检查。

## 6. 设备与重跑控制

可覆盖的设备变量：

```bash
export EXP1_LAFITE_DEVICE=cuda:0
export EXP1_QWEN_DEVICE=cuda:1
export EXP1_PROBE_DEVICE=cuda:0
export EXP2_LAFITE_DEVICE=cuda:2
export EXP3_QWEN_DEVICE=cuda:3
```

缓存默认 `overwrite: false`，中断后重跑会复用已完成且内容一致的样本。若更改了模型、输入尺寸、
CT 窗、prompt 或 checkpoint，应使用新的 `MEDICAL_OUTPUT_ROOT`；不要把不同设置写进
同一缓存目录。
