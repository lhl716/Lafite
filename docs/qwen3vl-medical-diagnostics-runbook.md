# LafitE × Qwen3-VL-8B 三项实验运行手册

这套配置使用 **MLLM**，具体模型默认为
`Qwen/Qwen3-VL-8B-Instruct`。LLM 纯文本模型不能提供实验一所需的视觉层、
projector 和图像 token，因此不适合替代它。

运行脚本现在默认使用单卡并串行执行：

| 工作 | 默认设备 |
| --- | --- |
| 实验一 LafitE 特征 | `cuda:0` |
| 实验一 Qwen3-VL 特征 | `cuda:0` |
| 实验二 LafitE 干预评分 | `cuda:0` |
| 实验三 Qwen3-VL 语义评分 | `cuda:0` |

Qwen3-VL-8B 的 BF16 权重和单图中间状态可以放入一张 A100-80GB。各子命令是
独立进程，前一个模型退出后才加载下一个模型，不会同时把 LafitE 和两份 Qwen 常驻显存。

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

## 2. 一键准备 BMAD Brain MRI 与 Liver CT

仓库默认使用 BMAD 的 Brain MRI 和 Liver CT 两个带像素级 mask 的子集。完整 BMAD
压缩包约 20.3 GiB；脚本会通过 `hf-mirror.com` 下载固定 revision，并且只解压
`Brain/` 和 `Liver/`。请至少预留约 30 GiB 空间。

先安装依赖，再打开 `scripts/medical/prepare-bmad-hf-mirror.sh`。阅读 BMAD 与原始数据源
条款后，把：

```bash
USER_ACCEPT_BMAD_LICENSE="true"
```

推荐把下载与整理分成两步。第一步使用 Hugging Face CLI；脚本自动优先选择新版
`hf`，并兼容旧版 `huggingface-cli`：

```bash
bash scripts/medical/prepare-bmad-hf-mirror.sh download
```

如果希望完全手动执行下载，新版 CLI 的等价命令是：

```bash
mkdir -p data/bmad/downloads
HF_ENDPOINT=https://hf-mirror.com hf download code-dev05/BMAD bmad.zip \
  --repo-type dataset \
  --revision 22cb97e1cfeeb9f677c6c57387347ce727625610 \
  --local-dir data/bmad/downloads
```

旧版 CLI 使用：

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download code-dev05/BMAD bmad.zip \
  --repo-type dataset \
  --revision 22cb97e1cfeeb9f677c6c57387347ce727625610 \
  --local-dir data/bmad/downloads
```

第二步完全离线地校验压缩包、选择性解压、进行患者级重划分，并生成 mask 与 manifest：

```bash
bash scripts/medical/prepare-bmad-hf-mirror.sh prepare
```

网络稳定时也可以连续执行：

```bash
bash scripts/medical/prepare-bmad-hf-mirror.sh all
```

默认输出：

```text
data/bmad/prepared/medical-manifest.jsonl
data/bmad/prepared/semantic-manifest.jsonl
data/bmad/prepared/preparation-summary.json
```

下载支持断点续传，原始压缩包会保留。如果你已在浏览器或其他服务器下载了
`bmad.zip`，设置 `USER_BMAD_ARCHIVE` 或环境变量 `BMAD_ARCHIVE` 后，只运行 `prepare`
即可，不会访问网络。由于 BMAD 原始 valid/test 病人存在交叉，
准备器按数据集和患者异常状态分层，重新进行确定性的 70/10/20 患者级划分；因此该
manifest 用于本文三个诊断实验，但不能冒充官方 BMAD benchmark split。

单卡先做 pilot 时，可以在同一脚本顶部设置：

```bash
USER_MAX_SLICES_PER_PATIENT="4"
USER_SEMANTIC_PATIENTS_PER_EDGE="25"
```

前者限制每位患者用于实验一、二的切片数；后者限制实验三每个
`organ×finding` 组合的患者数。`0` 表示保留完整数据。更改这两个值后重新运行会重写
prepared manifest 和派生 mask，不会重新下载压缩包。

注意：Liver 子集是预处理后的 PNG，不再具有原始 HU 数值，manifest 因而使用
`CT_PNG` modality，不执行 HU window。脚本生成的 `nuisance_mask` 只覆盖体外黑色背景，
`placebo_allowed_mask` 来自前景区域；它们是保守的候选干预区域。论文中的因果结论前，
必须抽样可视化审核，必要时替换为人工/解剖学 mask。

## 3. 自备数据时的 manifest 格式

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

## 4. 在运行脚本中设置路径和阈值

打开 `scripts/medical/run-qwen3vl-diagnostics.sh`，只修改文件顶部的
`User settings` 区域：

```bash
USER_BMAD_DATA_ROOT=""
USER_MEDICAL_MANIFEST=""
USER_SEMANTIC_MANIFEST=""
USER_LAFITE_CHECKPOINT="/data/checkpoints/ckpt_best.pth.tar"
USER_LAFITE_MASK_THRESHOLD="0.25"
USER_LAFITE_SCORE_THRESHOLD="0.50"
USER_MEDICAL_OUTPUT_ROOT="/data/results/lafite-medical"
USER_MLLM_MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
USER_SINGLE_GPU="cuda:0"
USER_QWEN_ATTENTION="sdpa"
USER_LAFITE_PYTHON="python3"
```

三个数据路径都留空时，运行脚本自动使用 `data/bmad/prepared/` 中的两个 manifest。
若数据准备在其他位置，只设置 `USER_BMAD_DATA_ROOT`；使用自备数据时才分别填写两个
manifest 路径。
如果 Qwen 权重已在本地，把 `USER_MLLM_MODEL_ID` 改成本地目录。脚本仍允许同名环境
变量覆盖这些值，因此 Slurm 等调度任务不需要修改仓库文件。

mask threshold 和 score threshold 的示例数字不是通用阈值，必须替换为 calibration/val
集的结果。脚本故意把它们默认留空，避免无意中在 test 集使用示例阈值。

## 5. 运行三个实验

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

### 5.1 单卡大致耗时

耗时主要由样本数 `N` 和实验三候选 finding 数 `K` 决定。以下只是
**A100-80GB、512×512、Qwen3-VL-8B、SDPA 的量级预估**，不是本仓库在你的数据上的
实测 benchmark：

| 实验 | 主要计算量 | 小规模 pilot 的常见量级 |
| --- | --- | --- |
| 实验一 | 每张图一次 LafitE、一次 Qwen，再训练多个等容量 probe | `N≈200–500`：约 20–90 分钟 |
| 实验二 | 正例约 4 个 LafitE 变体，正常例约 2 个变体 | `N≈200–500`：约 5–30 分钟 |
| 实验三 | 约 `4 × N × K` 次 Qwen 条件评分前向 | `N≈100, K≈5`：约 20–90 分钟 |

实验三通常最慢。例如 `N=500, K=10` 就是约 20,000 次模型前向，可能需要数小时。
首次运行还会下载/加载模型，不能把这段时间算入稳定吞吐。建议先准备一个患者级 pilot
manifest（每个 split 均保留，约 50–100 位患者）测出本机每样本耗时，再按下面公式估算：

```text
实验一总时长 ≈ 实验一 pilot 时长 × 完整样本数 / pilot 样本数
实验二总时长 ≈ 实验二 pilot 时长 × 完整样本数 / pilot 样本数
实验三总时长 ≈ 实验三 pilot 时长 × (完整 N×K) / (pilot N×K)
```

不要随机抽切片造成患者泄漏；pilot 也应按 `patient_id` 整体抽取。缓存默认保留，模型
导出中断后可以继续。

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

## 6. 查看结果

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

## 7. 设备与重跑控制

默认只需在 shell 顶部设置 `USER_SINGLE_GPU="cuda:0"`。需要重新拆到多卡时，才覆盖：

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
