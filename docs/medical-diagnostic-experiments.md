# LafitE × MLLM Medical Diagnostic Experiments

This directory implements three **go/no-go diagnostic experiments** to test three premises before designing a complete fusion model:

1. Whether local lesion information becomes progressively undecodable in the vision encoder, projector, or LLM.
2. Whether anomaly outputs causally depend on nonclinical nuisance factors outside the lesion.
3. Whether an LLM provides compositional knowledge that a small model cannot replace when visual evidence is available.

These programs do not constitute paper results. Formal conclusions require real medical data, patient-level splits, a locked test set, and patient-clustered statistics.

## 1. Installation and Quick Checks

The original LafitE training dependencies remain in `requirements.txt`. Core diagnostic dependencies and NIfTI, DICOM, and MLLM adapters are in `requirements-medical.txt`. Install `requirements.txt` separately in an appropriate CUDA environment for full LafitE training: its `faiss_gpu` dependency is unsuitable for CPU/macOS environments.

```bash
pip install -r requirements-medical.txt
bash scripts/medical/run-smoke-suite.sh
```

The smoke script generates nonmedical synthetic arrays and runs all three experiments in sequence. Results are written to `outputs/medical-smoke/` by default. Synthetic results validate only the code and data contracts; they must not be included in paper result tables.

Commands can also be run independently:

```bash
python -m medical_experiments --help
python -m medical_experiments validate-manifest data.jsonl --check-paths
```

## 2. Shared Data Manifest

Experiments 1 and 2 share a JSONL manifest, with one 2D slice or 3D volume per line:

```json
{"sample_id":"case001-z042","patient_id":"case001","image":"/data/case001.nii.gz","mask":"/data/case001-mask.nii.gz","slice_index":42,"depth_axis":2,"split":"train","modality":"CT","label":1}
```

Required fields:

- `sample_id`: a unique sample identifier.
- `patient_id`: a unique patient identifier.
- `image`: an NPY, NPZ, NIfTI, ordinary image, or DICOM series directory.
- `split`: `train`, `val`, `test`, `calibration`, or `external`.

Localization tasks also require `mask`. Optional fields include:

- `image_key`, `mask_key`: array keys in NPZ files.
- `slice_index`, `depth_axis`, `mask_depth_axis`: settings for extracting slices from volumes.
- `normal_reference`: a registered normal reference.
- `nuisance_mask`: a predefined nonclinical nuisance region.
- `placebo_mask`: a manually reviewed, equal-area placebo region.
- `placebo_allowed_mask`: a region restricting automatic or explicit placebo edits to the same organ or tissue.
- `organ`, `region`, `finding`, `report`: semantic fields for Experiment 3.

Check for leakage first:

```bash
python -m medical_experiments validate-manifest /path/to/manifest.jsonl \
  --check-paths --require-mask
```

The program fails if the same `patient_id` appears in multiple splits. To create a new split:

```bash
python -m medical_experiments split-patients raw.jsonl split.jsonl \
  --train 0.7 --val 0.1 --test 0.2 --stratify modality --seed 42
```

Slices are not statistically independent samples. Training random seeds must not be counted as clinical sample size either.

## 3. Medical Preprocessing Constraints

`medical_io.py` supports:

- CT: a fixed HU window mapped to `[0,1]`.
- MRI: robust percentile clipping and z-normalization within the foreground.
- 2.5D: three channels formed from neighboring slices.
- NPY/NPZ, NIfTI, common 2D images, and DICOM series.

For formal comparisons, all representation stages must use the same CT window, MRI sequence, and slice-selection rules. Cache 224-pixel, 448-pixel, and native-resolution sliding-window inputs separately to avoid attributing resize losses to semantic alignment.

## 4. Experiment 1: Decodability of Local Detail

### 4.1 Cache Frozen-Model Representations First

LafitE:

```bash
python -m medical_experiments export-lafite \
  --config experiments/medical-diagnostics/export-lafite.example.yaml
```

The cache contains:

- `feature_align`.
- `feature_rec`.
- `lafite_residual = abs(feature_align - feature_rec)`.
- `lafite_pred`.
- Ground-truth `mask` and patient metadata.

Other MLLMs can be connected through a factory and forward hooks:

```bash
python -m medical_experiments export-mllm \
  --config experiments/medical-diagnostics/export-mllm.example.yaml
```

The factory interface is:

```python
def build_bundle(kwargs):
    return {
        "model": frozen_model,
        "preprocess": lambda image_hwc, record: model_inputs,
        # Optional; defaults to model(**model_inputs).
        "forward": lambda model, model_inputs: model(**model_inputs),
        # Optional; for outputs unavailable through module hooks.
        "extract_output": lambda output: {"llm_deep": output.hidden_states[-1]},
    }
```

The configuration's `taps` specify actual module names, for example hooks on early vision layers, the final vision layer, and the projector. Internal names vary between MLLMs, so the repository does not hard-code a particular LLaVA, Qwen, or InternVL version.

By default, target fields such as `finding`, `report`, `label`, and the ground-truth mask are removed before the MLLM preprocessor or prompt builder is called, preventing label-conditioned representation caches.

Merge the two types of cache:

```bash
python -m medical_experiments merge-features \
  --config experiments/medical-diagnostics/merge-features.example.yaml
```

Merging uses exact `sample_id` matches and checks that `patient_id` and `split` agree. Any mismatch in sample sets causes failure by default.

### 4.2 Train Equal-Capacity Probes

```bash
python -m medical_experiments detail-probe --config /path/to/exp1.yaml
```

Core configuration:

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

Both NCHW dense features and NLC tokens are supported. NLC tokens must be recoverable as a spatial grid; remove CLS/register tokens with `prefix_tokens`. Masks use nearest-neighbor resizing, while features use bilinear resizing.

The program reports:

- Sample-level and patient-level Dice.
- Positive-only patient Dice.
- Pixel AUROC/AUPRC.
- Small-lesion component sensitivity.
- A fixed threshold selected on the training set.
- A control that spatially shuffles the residual and retrains a probe of the same capacity.
- Coordinate-only and random-feature negative controls, with random features matching first-order training statistics.
- An optional GAP plus linear coarse-label positive control to check whether coarse semantics remain readable at deeper stages.

Scientific interpretation:

- Strong early layers followed by poor performance after the projector support a connector bottleneck.
- Poor performance only at 224 pixels, with recovery at 448, primarily implicates resolution.
- Similar probes across stages but poor final decisions indicate that information remains available and its use is the problem.
- Improvement from `projector + residual` that disappears after residual shuffling supports conditional, additional spatial evidence from LafitE.

Run two configurations where possible: one preserving each representation's actual system capabilities, and another matching spatial grids, channel budgets, and probe capacity to isolate the mechanism.

## 5. Experiment 2: Lesion and Nonclinical Background Interventions

### 5.1 Generate Four Strictly Paired Versions

```bash
python -m medical_experiments make-interventions \
  --config experiments/medical-diagnostics/make-interventions.example.yaml
```

Each sample produces:

- `original`: the original image.
- `lesion_removed`: lesion repair using a matched normal reference or local Gaussian repair.
- `nuisance_changed`: noise, blur, or style changes only outside a dilated lesion-protection region.
- `placebo`: an edit of the same type in an equally shaped, equal-area region outside the lesion.

For formal experiments, supply a reviewed `nuisance_mask` in the manifest and set `require_nuisance_mask: true`. If omitted, the code perturbs the entire image outside the protected lesion region. This is suitable only for software debugging and cannot be interpreted as a purely background causal intervention.

The code does not use black-block occlusion. Outputs also record:

- MAE inside and outside the edited region.
- Changes in the protected ROI.
- A boundary-gradient artifact score.
- A continuity score.
- The fraction of the image actually changed.

When `model_factory` is set, the same frozen model immediately scores all four images. Its factory interface is:

```python
def build_predictor(kwargs):
    return {
        "predict": lambda image, record: {
            "score": anomaly_score,
            "mask": predicted_mask,       # Optional.
            "report_entities": ["nodule", "effusion"],  # Optional structured entities.
        }
}
```

Before the predictor is called, `finding`, `report`, `label`, `target_entity`, and `ground_truth` are removed by default to prevent test labels from entering prompts or model inputs. These fields remain only in evaluation records.

Automatic placebo generation matches lesion shape, soft-mask mass, and local intensity context. Formal experiments should also provide `placebo_allowed_mask` in each manifest record to constrain candidate locations to the same organ or tissue. In 3D, `dilation_radius` and `gaussian_sigma` are interpreted in voxels. Strongly anisotropic CT/MRI must therefore be resampled first, or these values must be converted externally using physical spacing.

The repository includes a reference LafitE factory: `medical_experiments.factories.lafite:make_lafite_predictor`. The mask threshold must come from the calibration split. Without a threshold, only anomaly scores are returned, avoiding artificial stability from per-image min-max normalization.

`cohort: positive` generates the full four-way paired lesion experiment. `cohort: normal` generates only original and nuisance-changed images to measure normal-case FPR drift at a fixed threshold. Analyze these patient groups separately so that zero lesion responses in normal cases do not dilute the lesion effect.

### 5.2 Patient-Level Paired Statistics

```bash
python -m medical_experiments shortcut-eval --config /path/to/exp2.yaml
```

Inputs use long-form JSONL:

```json
{"patient_id":"case001","sample_id":"case001-z042","variant":"original","score":0.91,"mask":"/pred/original.npy"}
{"patient_id":"case001","sample_id":"case001-z042","variant":"lesion_removed","score":0.23,"mask":"/pred/removed.npy"}
```

Optionally include `label`, `report_entities`, and `target_entity`, and supply a `decision_threshold` locked using the calibration set. The program then also computes normal-patient FPR drift, report-entity Jaccard similarity, and the disappearance rate of target entities after lesion removal. If these fields are missing, the corresponding endpoints are explicitly marked unevaluable; BLEU is not used as a substitute.

Aggregate within patients first, then bootstrap patients. Main outputs include:

- Paired score changes, `variant - original`.
- Mask Dice between nuisance/placebo and original images.
- `nuisance response / lesion-removal response`.
- Preregistered red-flag rules and individual decisions.

A formal study should use at least two independent lesion-repair methods and confirm, through a real-versus-edited artifact detector or blinded review, that the edits do not introduce new shortcuts. Anatomical context such as perilesional edema and mass effect must not automatically be classified as harmful background.

## 6. Experiment 3: LLM Necessity

Experiment 3 uses a separate JSONL file, with at least these fields per line:

```json
{"patient_id":"case001","organ":"lung","region":"right_upper_lobe","finding":"nodule"}
```

The program first deduplicates by patient, then holds out medically valid `organ × finding` edges. Every individual node must still appear in training.

### 6.1 Export Real LLM Scores Under Four Conditions

```bash
python -m medical_experiments export-semantic-scores \
  --config experiments/medical-diagnostics/export-semantic-scores.example.yaml
```

Factory interface:

```python
def build_scorer(kwargs):
    def score(record, candidates, evidence_present, knowledge_intact):
        return {candidate: logit for candidate in candidates}
    return score
```

The program enforces four conditions:

1. Image/evidence present + correct knowledge.
2. No image/evidence + correct knowledge.
3. Image/evidence present + shuffled knowledge.
4. No image/evidence + shuffled knowledge.

Target fields such as `finding`, `report`, and `label` are removed before the model is called. The no-image condition also removes the image, mask, feature cache, and region. The output fields are `scores`, `scores_no_image`, `scores_shuffled_knowledge`, and `scores_no_image_shuffled_knowledge`, respectively.

### 6.2 Compare Consistently Against Strong Baselines

```bash
python -m medical_experiments llm-necessity --config /path/to/exp3.yaml
```

Example configuration:

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

Outputs are `per_sample.jsonl` and `summary.json`. Metrics include:

- Seen/unseen MRR, Recall@k, and NLL.
- The unseen-minus-seen gap.
- The 2×2 evidence × knowledge interaction.
- Percentile bootstrap confidence intervals for these metrics, simple effects, and interactions.
- A patient/edge leakage audit.

Seen confidence intervals resample patients. Unseen intervals preferentially use two-stage `organ × finding` edge → patient resampling. All four evidence × knowledge conditions are always sampled together within the same patient block. With only one held-out test edge, edge-level uncertainty degenerates. These intervals also do not cover external distribution shifts to new hospitals or devices.

This runner implements the initial **3A: semantic ranking with oracle/fixed visual evidence**. Implementing 3B end-to-end fusion is worthwhile only if the LLM beats frequency priors and small models in 3A. In 3B, every semantic module should use the same LafitE/visual evidence, and Dice and FROC/AUPRO should be reported separately. The repository does not fabricate LLM localization results before a fusion model exists.

The claim that an LLM is indispensable for anomaly detection requires the full LLM to outperform the strongest small model, with gains concentrated on unseen edges and lost when images are removed or knowledge is shuffled. Improvements in report-writing style alone do not support that claim.

## 7. Recommended Formal Execution Order

1. Run Experiment 1 on BMAD Brain MRI and a CT dataset with masks.
2. If the residual adds conditional information, run Experiment 2.
3. Only after both gates pass, run Experiment 3 on CT data with reports and grounding.
4. Evaluate the locked test set once, after selecting thresholds and red-flag criteria on calibration/validation data.

Save the Git commit, configuration, manifest SHA256, data version, model checkpoint hash, software versions, and random seeds with every result.

## 8. Current Limitations

- The repository includes no medical data, model weights, or restricted reports.
- Internal MLLM module names and prompt templates are determined by the selected model's factory.
- Current LafitE is 2D. A 2.5D pilot cannot replace final 3D validation.
- Experiment 3's built-in learned embedding uses region as an evidence proxy and serves as a strong control model. A real LLM must use four-condition precomputed scores.
- Feature and intervention caches may contain patient-derived information. They are excluded by `.gitignore` and should still be stored in an appropriately controlled directory.

## 9. Qwen3-VL-8B Execution Entry Points

The repository provides formal configurations tied to Qwen3-VL-8B's internal layers, so users do not need to write their own factories:

```bash
bash scripts/medical/run-qwen3vl-diagnostics.sh exp1
bash scripts/medical/run-qwen3vl-diagnostics.sh exp2
bash scripts/medical/run-qwen3vl-diagnostics.sh exp3
bash scripts/medical/run-qwen3vl-diagnostics.sh results
```

For environment setup, threshold sources, device allocation, interpretation of individual outputs, and common errors, see
[`qwen3vl-medical-diagnostics-runbook.md`](qwen3vl-medical-diagnostics-runbook.md).
