# LafitE × Qwen3-VL-8B: Three-Experiment Runbook

This configuration uses an **MLLM**, defaulting to
`Qwen/Qwen3-VL-8B-Instruct`. A text-only LLM cannot supply the vision layers,
projector, and image tokens required by Experiment 1 and is therefore unsuitable as a replacement.

The scripts now default to sequential execution on a single GPU:

| Task | Default device |
| --- | --- |
| Experiment 1: LafitE features | `cuda:0` |
| Experiment 1: Qwen3-VL features | `cuda:0` |
| Experiment 2: LafitE intervention scoring | `cuda:0` |
| Experiment 3: Qwen3-VL semantic scoring | `cuda:0` |

Qwen3-VL-8B's BF16 weights and intermediate states for one image fit on one A100-80GB.
Each subcommand runs in a separate process. The next model loads only after the preceding
process exits, so LafitE and two Qwen instances are not kept in GPU memory simultaneously.

## 1. Installation

Create a Python 3.10 or 3.11 environment on a Linux/CUDA server. Do not download 8B model
weights to the current macOS machine to validate GPU experiments.

```bash
conda create -n lafite-med python=3.10 -y
conda activate lafite-med
pip install -r requirements-medical.txt
pip install -r requirements-lafite-medical.txt
```

The default attention implementation is PyTorch SDPA and works directly. If
`flash-attn` is correctly installed on the A100 environment, set
`QWEN_ATTENTION=flash_attention_2`.

First run the self-check, which does not download models:

```bash
bash scripts/medical/run-smoke-suite.sh
```

## 2. Prepare BMAD Brain MRI and Liver CT

The repository defaults to BMAD's Brain MRI and Liver CT subsets, both with pixel-level masks.
The full BMAD archive is approximately 20.3 GiB. The script downloads a pinned revision from
`huggingface.co` and extracts only `Brain/` and `Liver/`. Reserve at least approximately 30 GiB.

Install the dependencies, then open `scripts/medical/prepare-bmad.sh`. After reading the
terms for BMAD and its original data sources, set:

```bash
USER_ACCEPT_BMAD_LICENSE="true"
```

Download and preparation are best performed in two steps. The first uses `wget`/`curl`
directly to avoid Python Hub/Xet connection problems with large files and supports resuming:

```bash
bash scripts/medical/prepare-bmad.sh download
```

The equivalent manual download command is:

```bash
mkdir -p data/bmad/downloads
wget -c --tries=20 --timeout=30 \
  'https://huggingface.co/datasets/code-dev05/BMAD/resolve/22cb97e1cfeeb9f677c6c57387347ce727625610/bmad.zip?download=true' \
  -O data/bmad/downloads/bmad.zip
```

If `wget` is unavailable, use:

```bash
curl -L -C - --retry 20 --retry-all-errors --connect-timeout 30 \
  -o data/bmad/downloads/bmad.zip \
  'https://huggingface.co/datasets/code-dev05/BMAD/resolve/22cb97e1cfeeb9f677c6c57387347ce727625610/bmad.zip?download=true'
```

The second step runs fully offline: it validates the archive, extracts selected files,
creates new patient-level splits, and generates masks and manifests:

```bash
bash scripts/medical/prepare-bmad.sh prepare
```

With a stable connection, both steps can run consecutively:

```bash
bash scripts/medical/prepare-bmad.sh all
```

The current server preset reads the archive from `/home/yunlin.lhl/dataset/bmad.zip` and
writes extracted data, derived masks, and manifests to
`/data/oss_bucket_0/yunlin/dataset/bmad`. If the download is already complete, run only:

```bash
ACCEPT_BMAD_LICENSE=true bash scripts/medical/prepare-bmad.sh prepare
```

Default outputs:

```text
/data/oss_bucket_0/yunlin/dataset/bmad/prepared/medical-manifest.jsonl
/data/oss_bucket_0/yunlin/dataset/bmad/prepared/semantic-manifest.jsonl
/data/oss_bucket_0/yunlin/dataset/bmad/prepared/preparation-summary.json
```

Downloads can resume, and the original archive is retained. If `bmad.zip` was downloaded
through a browser or another server, set `USER_BMAD_ARCHIVE` or the `BMAD_ARCHIVE`
environment variable and run only `prepare`; no network access is required. Because patients
overlap between BMAD's original validation and test sets, the preparer creates new,
deterministic 70/10/20 patient-level splits, stratified by dataset and patient abnormality
status. This manifest is for these three diagnostic experiments and must not be presented
as the official BMAD benchmark split.

For an initial single-GPU pilot, set these values near the top of the same script:

```bash
USER_MAX_SLICES_PER_PATIENT="4"
USER_SEMANTIC_PATIENTS_PER_EDGE="25"
```

The first limits slices per patient in Experiments 1 and 2; the second limits patients per
`organ×finding` combination in Experiment 3. A value of `0` retains all data. Rerunning after
changing these values rewrites prepared manifests and derived masks without downloading
the archive again.

The Liver subset contains preprocessed PNGs and no longer retains original HU values.
Its manifest therefore uses the `CT_PNG` modality and applies no HU window. The generated
`nuisance_mask` covers only black background outside the body, while `placebo_allowed_mask`
comes from the foreground. These are conservative candidate intervention regions. Before
making causal claims in a paper, visually review a sample and replace them with manual or
anatomical masks where needed.

## 3. Retrain Brain/Liver LafitE

The main medical experiments do not load MVTec/MPDD anomaly-detection checkpoints.
Both LafitE models start from the same general ImageNet EfficientNet-B4 initialization
and train only their diffusion reconstruction modules on the corresponding normal
training cohorts. During both training and diagnostic inference, BMAD PNGs consistently
use original 8-bit values divided by 255 and ImageNet mean/std normalization, avoiding
inconsistent intensity transforms between MRI and CT-PNG.

First generate LafitE training manifests. The default policy is strictly patient-normal:
if any slice from a patient is abnormal, that patient's other normal slices are also
excluded from unsupervised training.

```bash
bash scripts/medical/train-bmad-lafite.sh prepare
```

Train and calibrate sequentially on one GPU:

```bash
bash scripts/medical/train-bmad-lafite.sh brain
bash scripts/medical/train-bmad-lafite.sh calibrate-brain
bash scripts/medical/train-bmad-lafite.sh liver
bash scripts/medical/train-bmad-lafite.sh calibrate-liver
```

Or run the full sequence:

```bash
bash scripts/medical/train-bmad-lafite.sh all
```

By default, the script runs 1000 epochs with batch size 64, validating and saving every
50 epochs. Before the first full run, validate the complete training pipeline with the
following command. The full run will resume automatically from its checkpoint:

```bash
MAX_EPOCHS=2 VAL_FREQ=1 SAVE_STEP=1 BATCH_SIZE=8 \
  bash scripts/medical/train-bmad-lafite.sh brain
```

Default artifacts:

```text
/data/oss_bucket_0/yunlin/dataset/bmad/prepared/lafite/brain-train.jsonl
/data/oss_bucket_0/yunlin/dataset/bmad/prepared/lafite/liver-train.jsonl
/data/oss_bucket_0/yunlin/dataset/bmad/lafite-runs/brain/checkpoints/ckpt_best.pth.tar
/data/oss_bucket_0/yunlin/dataset/bmad/lafite-runs/liver/checkpoints/ckpt_best.pth.tar
/data/oss_bucket_0/yunlin/dataset/bmad/lafite-runs/brain/calibration.json
/data/oss_bucket_0/yunlin/dataset/bmad/lafite-runs/liver/calibration.json
```

If `USER_EFFICIENTNET_B4_WEIGHTS` is empty, the original EfficientNet implementation
downloads and caches ImageNet weights. If the server cannot access GitHub releases,
place `efficientnet-b4-6ed6700e.pth` locally and enter its path near the top of
`scripts/medical/train-bmad-lafite.sh`. This is general ImageNet initialization, not an
industrial-defect checkpoint.

## 4. Manifest Format for Your Own Data

Experiments 1 and 2 require at least:

```json
{"sample_id":"case001-z042","patient_id":"case001","image":"/data/case001.npy","mask":"/data/case001-mask.npy","nuisance_mask":"/data/case001-nuisance.npy","slice_index":42,"depth_axis":0,"split":"test","modality":"CT","label":1}
```

Experiment 3 additionally requires `organ`, `region`, and `finding` on each line, and the
evidence-present condition requires `image`. Use either one shared manifest or a separate
semantic manifest.

Data requirements:

- Each `patient_id` may occur in only one split.
- Experiment 1 requires train, val, and test splits, with positive masks in train and val.
- Positive test cases in Experiment 2 require lesion masks; every sample used for
  interventions requires a previously reviewed `nuisance_mask`.
- Brain and Liver mask and score thresholds must be determined separately using only
  their respective validation sets.
- BMAD MRI and CT-PNG are both preprocessed PNGs and currently use `uint8_passthrough`.
  Switching to raw NIfTI/DICOM requires changing both training and inference preprocessing.

## 5. Set Paths and Thresholds in the Run Script

Open `scripts/medical/run-qwen3vl-diagnostics.sh` and edit only the `User settings`
section at the top:

```bash
USER_BMAD_DATA_ROOT=""
USER_SEMANTIC_MANIFEST=""
USER_LAFITE_RUN_ROOT="/data/checkpoints/lafite-runs"
USER_BRAIN_LAFITE_CHECKPOINT=""
USER_LIVER_LAFITE_CHECKPOINT=""
USER_BRAIN_LAFITE_MASK_THRESHOLD=""
USER_BRAIN_LAFITE_SCORE_THRESHOLD=""
USER_LIVER_LAFITE_MASK_THRESHOLD=""
USER_LIVER_LAFITE_SCORE_THRESHOLD=""
USER_MEDICAL_OUTPUT_ROOT="/data/results/lafite-medical"
USER_MLLM_MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
USER_SINGLE_GPU="cuda:0"
USER_QWEN_ATTENTION="sdpa"
USER_LAFITE_PYTHON="python3"
USER_HF_ENDPOINT="https://huggingface.co"
USER_HF_HOME="/data/model-cache/huggingface"
```

When checkpoint paths and thresholds are left empty, the script reads the default files
produced by training and calibration above. Brain and Liver always run and report
separately; one LafitE checkpoint is no longer used for both modalities.
If Qwen weights are already local, set `USER_MLLM_MODEL_ID` to their directory.
Environment variables with the same names can still override these values, so scheduler
jobs such as Slurm jobs do not need to modify repository files.

Manual threshold entries are intended only for reproducing experiments. New training runs
read the two `calibration.json` files by default, avoiding threshold selection from test data.

Training checkpoints, TensorBoard logs, and calibration results can be stored on OSS.
However, LafitE validation creates temporary ZIP files with `np.savez` that require random
seeks. The training script therefore writes these temporary files to local
`/tmp/lafite-eval` by default and deletes them after validation. Do not set
`USER_LAFITE_LOCAL_SCRATCH` to an OSS path. If local `/tmp` space is insufficient, use a
local NVMe/scratch directory on the compute node.

## 6. Run the Three Experiments

Run experiments individually first to make data problems easier to identify:

```bash
bash scripts/medical/run-qwen3vl-diagnostics.sh exp1
bash scripts/medical/run-qwen3vl-diagnostics.sh exp2
bash scripts/medical/run-qwen3vl-diagnostics.sh exp3
```

Once data and thresholds have been checked, run them consecutively:

```bash
bash scripts/medical/run-qwen3vl-diagnostics.sh all
```

### 6.1 Approximate Single-GPU Runtime

Runtime depends mainly on the number of samples `N` and the number of candidate findings
`K` in Experiment 3. These are **order-of-magnitude estimates for an A100-80GB, 512×512
inputs, Qwen3-VL-8B, and SDPA**, not benchmarks measured by this repository on your data:

| Experiment | Main computation | Typical small-pilot scale |
| --- | --- | --- |
| Brain LafitE training | Normal-patient training, diffusion reconstruction | Full 1000 epochs: about 4–10 hours |
| Liver LafitE training | Normal-patient training, diffusion reconstruction | Full 1000 epochs: about 2–6 hours |
| Experiment 1 | One LafitE and one Qwen pass per image, then multiple equal-capacity probes | `N≈200–500`: about 20–90 minutes |
| Experiment 2 | About four LafitE variants per positive case and two per normal case | `N≈200–500`: about 5–30 minutes |
| Experiment 3 | About `4 × N × K` Qwen conditional-scoring forward passes | `N≈100, K≈5`: about 20–90 minutes |

Experiment 3 is usually the slowest. For example, `N=500, K=10` means approximately
20,000 forward passes and may take several hours. Initial model downloads and loading
must not be included in steady-state throughput. First prepare a patient-level pilot
manifest of about 50–100 patients, retaining every split, measure time per sample on
the actual machine, then estimate:

```text
Experiment 1 total time ≈ Experiment 1 pilot time × full sample count / pilot sample count
Experiment 2 total time ≈ Experiment 2 pilot time × full sample count / pilot sample count
Experiment 3 total time ≈ Experiment 3 pilot time × (full N×K) / (pilot N×K)
```

Do not randomly subsample slices in a way that leaks patients. Pilot sampling must also
select whole `patient_id` groups. Caches are retained by default, and model export can
resume after interruption.

Experiment 1 uses fixed 512×512 inputs. Qwen3-VL caches these spatial representations:

- Vision block 2: a `32×32` pre-merger patch grid.
- Vision block 26: a `32×32` pre-merger patch grid.
- Visual merger/projector: a `16×16` image-token grid.
- LLM hidden states 18 and 36: only the `16×16` image tokens extracted from the full sequence.

Experiment 3 computes the mean token conditional log-probability of each candidate finding,
rather than generating reports. It automatically runs all four image present/absent ×
knowledge intact/shuffled conditions. Default knowledge text contains only candidate names.
For formal paper experiments, preregister candidate definitions in
`factory_kwargs.knowledge` in
`experiments/medical-diagnostics/qwen3-vl-8b/exp3-export-scores.yaml`, or set `knowledge_file`
to a JSON file:

```json
{
  "nodule": "a focal rounded pulmonary opacity",
  "effusion": "fluid in the pleural space"
}
```

## 7. Inspect Results

Print the three core metrics with one command:

```bash
bash scripts/medical/run-qwen3vl-diagnostics.sh results
```

Raw result files:

| Experiment | Main result |
| --- | --- |
| Experiment 1 | `$MEDICAL_OUTPUT_ROOT/{brain,liver}/exp1/probe-summary.json` |
| Experiment 2 | `$MEDICAL_OUTPUT_ROOT/{brain,liver}/exp2/evaluation-summary.json` |
| Experiment 3 | `$MEDICAL_OUTPUT_ROOT/exp3/evaluation-summary.json` |

Interpret results in this order:

1. **Experiment 1:** compare positive-patient Dice/AUPRC along
   `vision_early → vision_last → projector → llm_deep`. A sustained decrease indicates
   that local information becomes less decodable along the semantic-alignment pathway.
   Support for additional local evidence from the LafitE residual requires a clear
   recovery with `projector_plus_residual` or `llm_deep_plus_residual`, and loss of that
   gain after spatially shuffling the residual.
2. **Experiment 2:** first check whether the shortcut gate is `pass`. Lesion removal
   should substantially lower scores, while nuisance/placebo changes should have small
   effects. Focus on the nuisance/lesion ratio, normal-case FPR increases, and all
   triggered red flags.
3. **Experiment 3:** treat unseen `organ×finding` combinations as the key results.
   Qwen3-VL should beat frequency and small_embedding baselines, and performance should
   decline after image removal and knowledge shuffling. Ideally, the MRR
   evidence×knowledge interaction is positive with a confidence interval excluding zero.

Do not decide these gates from a point estimate alone. Review patient counts, bootstrap
confidence intervals, editing quality, and preregistered thresholds together.

## 8. Devices and Rerun Controls

By default, setting `USER_SINGLE_GPU="cuda:0"` at the top of the shell script is sufficient.
To distribute work across multiple GPUs again, override:

```bash
export EXP1_LAFITE_DEVICE=cuda:0
export EXP1_QWEN_DEVICE=cuda:1
export EXP1_PROBE_DEVICE=cuda:0
export EXP2_LAFITE_DEVICE=cuda:2
export EXP3_QWEN_DEVICE=cuda:3
```

Caches default to `overwrite: false`. After interruption, reruns reuse completed samples
whose contents match. If the model, input dimensions, input scaling, prompt, or checkpoint
changes, use a new `MEDICAL_OUTPUT_ROOT`. Do not mix different settings in one cache directory.
