#!/usr/bin/env python3
"""Forensic diagnostics for a trained medical LafitE checkpoint.

The brain checkpoint reports pixel AUROC ~0.53 while image-level AUROC is
~0.95.  That dissociation has several mutually exclusive explanations, and each
one has a different fix.  This script runs them all against the same validation
cohort so the next architectural decision rests on a measurement instead of a
guess.

Sections
--------
1. cohort audit      cohort composition, mask sanity, image statistics.
2. metric forensics  pooled vs per-image vs foreground-restricted pixel AUROC.
                     Pooled AUROC compares raw scores *across* images, so a
                     per-image scale that varies with slice position destroys it
                     even when every individual map localizes correctly.
3. residual scale    |F - F_N| / |F| inside lesion / normal tissue / background.
                     A tiny ratio everywhere means an identity shortcut; a large
                     ratio that does not separate lesion from tissue means the
                     reconstruction never anchored to this slice's anatomy.
4. infer_time sweep  localization as a function of the diffusion start step.
                     At the industrial default (960, cosine) only 6% of the
                     input feature survives, so the reconstruction is close to
                     an unconditional sample from the normal manifold.
5. linear probe      is lesion information decodable from the residual channels
                     even when its L2 norm is uninformative?  This is the test
                     that decides whether a learned evidence matcher over the
                     residual field is viable at all.

Example
-------
    python3 scripts/medical/diagnose-lafite-residual.py \
        --config experiments/BMAD-Brain-LafitE/config.yaml \
        --checkpoint /path/to/brain/checkpoints/ckpt_best.pth.tar \
        --val-meta /path/to/prepared/lafite/brain-val.jsonl \
        --infer-times 200,400,600,800,960 \
        --output /path/to/brain-diagnosis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.ndimage import gaussian_filter
from sklearn import metrics as sk_metrics
from tqdm.auto import tqdm

# UNetGD wraps its DDIM sampling loop in tqdm, which emits one bar per batch and
# drowns the report.  Silence it here only -- training keeps its own bars.
import lib.models.reconstructions.unetgd as _unetgd

_unetgd.tqdm = lambda iterable=None, *a, **kw: iterable if iterable is not None else []

EPS = 1e-8


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _auroc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Unflipped AUROC.  eval_helper flips values below 0.5; we do not, because
    a flipped score is itself diagnostic information."""
    labels = labels.astype(np.int8)
    if labels.min() == labels.max():
        return None
    return float(sk_metrics.roc_auc_score(labels, scores))


def _section(title: str) -> str:
    return f"\n{'=' * 78}\n== {title}\n{'=' * 78}"


def _pretty(value: Any) -> str:
    """Render a scalar, list, or nested mapping without dumping raw repr()."""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        return "  ".join(f"{k}={_pretty(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        if len(value) > 6:
            return f"[{len(value)} items] " + ", ".join(_pretty(v) for v in value[:6]) + ", ..."
        return ", ".join(_pretty(v) for v in value)
    return str(value)


def _render_kv(mapping: Mapping[str, Any]) -> str:
    if not mapping:
        return "  (empty)"
    width = max(len(str(key)) for key in mapping)
    return "\n".join(f"  {str(key):<{width}} : {_pretty(value)}" for key, value in mapping.items())


SWEEP_COLUMNS = (
    ("infer_t", 8, lambda t, r: str(t)),
    ("signal", 7, lambda t, r: f"{r['signal_fraction']:.1%}"),
    ("pooled", 8, lambda t, r: _num(r["pixel_auroc_pooled"])),
    ("fg_only", 8, lambda t, r: _num(r["pixel_auroc_pooled_foreground_only"])),
    ("per_img", 8, lambda t, r: _num(r["pixel_auroc_per_image_mean"])),
    ("znorm", 8, lambda t, r: _num(r["pixel_auroc_pooled_after_per_image_znorm"])),
    ("d/F les", 9, lambda t, r: _num(r["relative_residual"]["lesion"])),
    ("d/F tis", 9, lambda t, r: _num(r["relative_residual"]["normal_tissue"])),
    ("d/F bg", 9, lambda t, r: _num(r["relative_residual"]["background"])),
    ("img_max", 8, lambda t, r: _num(r["image_auroc_max_avgpool16"])),
)


def _num(value: Any) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    return "nan" if value != value else f"{value:.4f}"


def _render_sweep(sweep_results: Mapping[str, Any], order: Sequence[int]) -> str:
    header = " ".join(f"{name:>{width}}" for name, width, _ in SWEEP_COLUMNS)
    lines = [header, "-" * len(header)]
    for infer_time in order:
        result = sweep_results.get(str(infer_time))
        if result is None:
            continue
        lines.append(
            " ".join(f"{render(infer_time, result):>{width}}" for _, width, render in SWEEP_COLUMNS)
        )
    return "\n".join(lines)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _signal_fraction(timestep: int, timesteps: int = 1000, s: float = 0.008) -> float:
    """sqrt(alpha_bar) under the cosine schedule LafitE defaults to."""
    f = lambda x: np.cos((x / timesteps + s) / (1 + s) * np.pi / 2) ** 2
    return float(np.sqrt(max(f(timestep) / f(0), 0.0)))


def _pool_to_grid(array: np.ndarray, size: int) -> np.ndarray:
    """Area-average a HxW map down to size x size (used to align masks with the
    32x32 feature grid)."""
    tensor = torch.as_tensor(array, dtype=torch.float32)[None, None]
    pooled = functional.interpolate(tensor, size=(size, size), mode="area")
    return pooled[0, 0].numpy()


# ---------------------------------------------------------------------------
# 1. cohort audit
# ---------------------------------------------------------------------------


def audit_cohort(val_meta: Path, limit: int) -> Dict[str, Any]:
    import cv2

    rows = _read_jsonl(val_meta)
    patients_by_label: Dict[int, set] = defaultdict(set)
    for row in rows:
        patients_by_label[int(row.get("label", 0))].add(str(row.get("patient_id", "?")))

    lesion_fraction: List[float] = []
    lesion_fraction_in_brain: List[float] = []
    foreground_fraction: List[float] = []
    empty_positive_masks: List[str] = []
    nonempty_negative_masks: List[str] = []
    missing_mask_positive: List[str] = []
    native_shapes: set = set()
    grayscale_flags: List[bool] = []

    inspected = rows if limit <= 0 else rows[:limit]
    for row in inspected:
        label = int(row.get("label", 0))
        image = cv2.imread(str(row["filename"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(row["filename"])
        native_shapes.add(tuple(image.shape[:2]))
        grayscale_flags.append(
            bool(np.array_equal(image[..., 0], image[..., 1]) and np.array_equal(image[..., 1], image[..., 2]))
        )
        gray = image.mean(axis=2) / 255.0
        foreground = gray > 0.05
        foreground_fraction.append(float(foreground.mean()))

        maskname = row.get("maskname")
        if not maskname:
            if label == 1:
                missing_mask_positive.append(str(row.get("sample_id", row["filename"])))
            continue
        mask = cv2.imread(str(maskname), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(maskname)
        positive = mask > 0
        if label == 1 and not positive.any():
            empty_positive_masks.append(str(row.get("sample_id", row["filename"])))
        if label == 0 and positive.any():
            nonempty_negative_masks.append(str(row.get("sample_id", row["filename"])))
        if label == 1 and positive.any():
            lesion_fraction.append(float(positive.mean()))
            if foreground.shape == positive.shape and foreground.any():
                lesion_fraction_in_brain.append(float(positive.sum() / max(foreground.sum(), 1)))

    def _quantiles(values: Sequence[float]) -> Optional[Dict[str, float]]:
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        return {
            "p10": float(np.quantile(array, 0.10)),
            "median": float(np.median(array)),
            "p90": float(np.quantile(array, 0.90)),
        }

    return {
        "slices_total": len(rows),
        "slices_normal": sum(1 for row in rows if int(row.get("label", 0)) == 0),
        "slices_anomalous": sum(1 for row in rows if int(row.get("label", 0)) == 1),
        "patients_total": len({str(row.get("patient_id", "?")) for row in rows}),
        "patients_normal_only": len(patients_by_label[0] - patients_by_label[1]),
        "patients_anomalous": len(patients_by_label[1]),
        "slices_inspected": len(inspected),
        "native_shapes": sorted(str(shape) for shape in native_shapes),
        "images_effectively_grayscale": bool(all(grayscale_flags)) if grayscale_flags else None,
        "foreground_fraction": _quantiles(foreground_fraction),
        "lesion_fraction_of_image": _quantiles(lesion_fraction),
        "lesion_fraction_of_brain": _quantiles(lesion_fraction_in_brain),
        "positive_slices_with_empty_mask": empty_positive_masks[:20],
        "positive_slices_with_empty_mask_count": len(empty_positive_masks),
        "negative_slices_with_nonempty_mask": nonempty_negative_masks[:20],
        "negative_slices_with_nonempty_mask_count": len(nonempty_negative_masks),
        "positive_slices_without_mask_count": len(missing_mask_positive),
    }


# ---------------------------------------------------------------------------
# 2-4. model pass
# ---------------------------------------------------------------------------


class Collected:
    """Per-slice quantities gathered in one forward pass."""

    def __init__(self) -> None:
        self.preds: List[np.ndarray] = []          # 224x224 anomaly map, gauss-filtered
        self.masks: List[np.ndarray] = []          # 224x224 bool lesion mask
        self.foregrounds: List[np.ndarray] = []    # 224x224 bool brain mask
        self.labels: List[int] = []
        self.filenames: List[str] = []
        self.relative_lesion: List[float] = []
        self.relative_tissue: List[float] = []
        self.relative_background: List[float] = []
        self.probe_features: List[np.ndarray] = []  # (cells, C) residual channels
        self.probe_labels: List[np.ndarray] = []
        self.probe_norms: List[np.ndarray] = []
        self.probe_patients: List[np.ndarray] = []


def run_pass(
    model,
    loader,
    device: str,
    infer_time: int,
    repeats: int,
    pixel_mean: Sequence[float],
    pixel_std: Sequence[float],
    collect_probe: bool,
    patient_by_filename: Mapping[str, str],
    probe_negatives_per_positive: int,
    limit: int = 0,
    grid: int = 32,
    desc: str = "pass",
) -> Collected:
    model.reconstruction.infer_time = int(infer_time)
    mean = torch.as_tensor(list(pixel_mean)).view(1, -1, 1, 1)
    std = torch.as_tensor(list(pixel_std)).view(1, -1, 1, 1)
    out = Collected()
    rng = np.random.default_rng(0)
    seen = 0

    total = len(loader)
    if limit > 0 and loader.batch_size:
        total = min(total, -(-limit // loader.batch_size))
    progress = tqdm(loader, desc=desc, total=total, leave=False, unit="batch")

    with torch.no_grad():
        for batch in progress:
            if limit > 0 and seen >= limit:
                break
            signed_deltas = []
            norm_maps = []
            preds = []
            feature = None
            for _ in range(max(1, repeats)):
                outputs = model(batch)
                preds.append(outputs["pred"].detach().float().cpu())
                delta_repeat = (outputs["feature_align"] - outputs["feature_rec"]).detach().float().cpu()
                signed_deltas.append(delta_repeat)
                norm_maps.append(delta_repeat.pow(2).sum(dim=1).sqrt())
                feature = outputs["feature_align"].detach().float().cpu()
            # The anomaly map averages per-repeat magnitudes, matching what an
            # ensembled evaluation would report.  The probe consumes the mean
            # signed delta, where independent sampling noise cancels instead of
            # accumulating.
            pred = torch.stack(preds).mean(0)                  # B x 1 x H x W
            delta = torch.stack(signed_deltas).mean(0)         # B x C x g x g
            delta_norm = torch.stack(norm_maps).mean(0)        # B x g x g

            images = batch["image"].detach().float().cpu()
            raw = images * std + mean                          # back to [0, 1]
            gray = raw.mean(dim=1)                             # B x H x W
            masks = batch["mask"].detach().float().cpu()[:, 0] # B x H x W

            feature_norm = feature.pow(2).sum(dim=1).sqrt()    # B x g x g
            relative = (delta_norm / (feature_norm + EPS)).numpy()
            seen += pred.shape[0]

            for index in range(pred.shape[0]):
                filename = batch["filename"][index]
                mask_map = masks[index].numpy() > 0
                foreground = gray[index].numpy() > 0.05
                out.preds.append(gaussian_filter(pred[index, 0].numpy(), sigma=4))
                out.masks.append(mask_map)
                out.foregrounds.append(foreground)
                out.labels.append(int(batch["label"][index]))
                out.filenames.append(filename)

                lesion_grid = _pool_to_grid(mask_map.astype(np.float32), grid) > 0.5
                brain_grid = _pool_to_grid(foreground.astype(np.float32), grid) > 0.5
                tissue_grid = brain_grid & ~lesion_grid
                background_grid = ~brain_grid
                rel = relative[index]
                out.relative_lesion.append(float(rel[lesion_grid].mean()) if lesion_grid.any() else np.nan)
                out.relative_tissue.append(float(rel[tissue_grid].mean()) if tissue_grid.any() else np.nan)
                out.relative_background.append(
                    float(rel[background_grid].mean()) if background_grid.any() else np.nan
                )

                if not collect_probe or not lesion_grid.any():
                    continue
                # Restrict the probe to brain tissue: separating brain from air
                # is trivial and would inflate the probe's apparent skill.
                cells = delta[index].numpy().reshape(delta.shape[1], -1).T   # (g*g, C)
                norms = delta_norm[index].numpy().reshape(-1)
                lesion_flat = lesion_grid.reshape(-1)
                brain_flat = brain_grid.reshape(-1)
                positive_idx = np.flatnonzero(lesion_flat & brain_flat)
                negative_idx = np.flatnonzero(~lesion_flat & brain_flat)
                if positive_idx.size == 0 or negative_idx.size == 0:
                    continue
                take = min(negative_idx.size, positive_idx.size * probe_negatives_per_positive)
                negative_idx = rng.choice(negative_idx, size=take, replace=False)
                chosen = np.concatenate([positive_idx, negative_idx])
                out.probe_features.append(cells[chosen].astype(np.float32))
                out.probe_labels.append(lesion_flat[chosen].astype(np.int64))
                out.probe_norms.append(norms[chosen].astype(np.float32))
                patient = patient_by_filename.get(filename, filename)
                out.probe_patients.append(np.asarray([patient] * chosen.size))
    progress.close()
    return out


# ---------------------------------------------------------------------------
# metric forensics
# ---------------------------------------------------------------------------


def metric_forensics(collected: Collected) -> Dict[str, Any]:
    preds = collected.preds
    masks = collected.masks
    foregrounds = collected.foregrounds
    labels = np.asarray(collected.labels)

    pooled_scores = np.concatenate([pred.reshape(-1) for pred in preds])
    pooled_labels = np.concatenate([mask.reshape(-1) for mask in masks]).astype(np.int64)

    fg_scores = np.concatenate([pred[fg].reshape(-1) for pred, fg in zip(preds, foregrounds)])
    fg_labels = np.concatenate([mask[fg].reshape(-1) for mask, fg in zip(masks, foregrounds)]).astype(np.int64)

    # Per-image ranking: does the lesion outrank this same slice's own healthy
    # tissue?  Immune to any per-slice scale difference.
    per_image: List[float] = []
    znorm_scores: List[np.ndarray] = []
    znorm_labels: List[np.ndarray] = []
    for pred, mask, fg in zip(preds, masks, foregrounds):
        inside = pred[fg]
        target = mask[fg]
        if inside.size == 0:
            continue
        centered = (inside - inside.mean()) / (inside.std() + EPS)
        znorm_scores.append(centered)
        znorm_labels.append(target.astype(np.int64))
        if target.any() and not target.all():
            per_image.append(float(sk_metrics.roc_auc_score(target.astype(np.int64), inside)))

    znorm_auc = _auroc(np.concatenate(znorm_scores), np.concatenate(znorm_labels))

    image_mean = np.asarray([pred.mean() for pred in preds])
    image_std = np.asarray([pred.std() for pred in preds])
    image_max = np.asarray(
        [
            functional.avg_pool2d(torch.as_tensor(pred)[None, None], 16, stride=1).max().item()
            for pred in preds
        ]
    )

    return {
        "pixel_auroc_pooled": _auroc(pooled_scores, pooled_labels),
        "pixel_auroc_pooled_foreground_only": _auroc(fg_scores, fg_labels),
        "pixel_auroc_per_image_mean": float(np.mean(per_image)) if per_image else None,
        "pixel_auroc_per_image_median": float(np.median(per_image)) if per_image else None,
        "pixel_auroc_per_image_count": len(per_image),
        "pixel_auroc_pooled_after_per_image_znorm": znorm_auc,
        "image_auroc_mean": _auroc(image_mean, labels),
        "image_auroc_std": _auroc(image_std, labels),
        "image_auroc_max_avgpool16": _auroc(image_max, labels),
        "relative_residual": {
            "lesion": float(np.nanmean(collected.relative_lesion)),
            "normal_tissue": float(np.nanmean(collected.relative_tissue)),
            "background": float(np.nanmean(collected.relative_background)),
            "lesion_over_tissue": float(
                np.nanmean(collected.relative_lesion) / (np.nanmean(collected.relative_tissue) + EPS)
            ),
        },
    }


# ---------------------------------------------------------------------------
# 5. linear probe on the residual channels
# ---------------------------------------------------------------------------


def run_probe(collected: Collected, seed: int = 0) -> Dict[str, Any]:
    if not collected.probe_features:
        return {"status": "no positive cells collected"}
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    features = np.concatenate(collected.probe_features, axis=0)
    labels = np.concatenate(collected.probe_labels, axis=0)
    norms = np.concatenate(collected.probe_norms, axis=0)
    patients = np.concatenate(collected.probe_patients, axis=0)

    unique = np.unique(patients)
    if unique.size < 2:
        return {"status": "probe needs at least two patients", "patients": int(unique.size)}
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    holdout = set(shuffled[: max(1, int(round(unique.size * 0.3)))].tolist())
    test_mask = np.asarray([patient in holdout for patient in patients])
    train_mask = ~test_mask
    if labels[train_mask].min() == labels[train_mask].max():
        return {"status": "patient-disjoint split left a single class in train"}
    if labels[test_mask].min() == labels[test_mask].max():
        return {"status": "patient-disjoint split left a single class in test"}

    scaler = StandardScaler().fit(features[train_mask])
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(scaler.transform(features[train_mask]), labels[train_mask])
    probe_scores = model.decision_function(scaler.transform(features[test_mask]))

    return {
        "status": "ok",
        "cells_train": int(train_mask.sum()),
        "cells_test": int(test_mask.sum()),
        "positive_rate_test": float(labels[test_mask].mean()),
        "patients_train": int(unique.size - len(holdout)),
        "patients_test": int(len(holdout)),
        "auroc_raw_residual_norm": _auroc(norms[test_mask], labels[test_mask]),
        "auroc_linear_probe": _auroc(probe_scores, labels[test_mask]),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def build_loader(config_path: Path, val_meta: Path, batch_size: int, workers: int):
    import yaml
    from easydict import EasyDict

    from lib.datasets.data_builder import build_dataloader
    from lib.utils.misc_helper import update_config

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = update_config(EasyDict(yaml.safe_load(handle)))
    cfg.dataset.pop("train", None)
    cfg.dataset.test.meta_file = str(val_meta)
    cfg.dataset.batch_size = batch_size
    cfg.dataset.workers = workers
    _, loader = build_dataloader(cfg.dataset, distributed=False)
    return loader, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--val-meta", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--infer-times",
        default="",
        help="comma separated diffusion start steps to sweep; default is the config value only",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="average this many stochastic reconstructions per slice (ddim_sampling_eta=1 makes sampling noisy)",
    )
    parser.add_argument("--probe-infer-time", type=int, default=None, help="default: first swept value")
    parser.add_argument("--probe-negatives-per-positive", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="cap slices per pass; <=0 for the whole cohort")
    parser.add_argument("--audit-limit", type=int, default=400, help="slices to open for the cohort audit; <=0 for all")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from medical_experiments.lafite_adapter import load_lafite

    report: Dict[str, Any] = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "val_meta": str(args.val_meta),
    }

    print(_section("1. cohort audit"), flush=True)
    report["cohort"] = audit_cohort(args.val_meta, args.audit_limit)
    print(_render_kv(report["cohort"]), flush=True)

    rows = _read_jsonl(args.val_meta)
    patient_by_filename = {str(row["filename"]): str(row.get("patient_id", row["filename"])) for row in rows}

    loader, cfg = build_loader(args.config, args.val_meta, args.batch_size, args.workers)
    model, load_info = load_lafite(args.config, args.checkpoint, device=args.device)
    default_infer_time = int(cfg.net[-1].kwargs.get("infer_time", 960))

    if args.infer_times.strip():
        sweep = [int(value) for value in args.infer_times.split(",") if value.strip()]
    else:
        sweep = [default_infer_time]
    probe_infer_time = args.probe_infer_time if args.probe_infer_time is not None else sweep[0]

    print(_section("2-4. residual forensics and infer_time sweep"), flush=True)
    print(f"  sweeping infer_time over {sweep} ({len(sweep)} pass(es))", flush=True)

    sweep_results: Dict[str, Any] = {}
    probe_report: Dict[str, Any] = {"status": "skipped"}
    probe_collected = None
    for position, infer_time in enumerate(sweep, start=1):
        collect_probe = (not args.skip_probe) and infer_time == probe_infer_time
        collected = run_pass(
            model,
            loader,
            args.device,
            infer_time,
            args.repeats,
            load_info["pixel_mean"],
            load_info["pixel_std"],
            collect_probe,
            patient_by_filename,
            args.probe_negatives_per_positive,
            limit=args.limit,
            desc=f"  pass {position}/{len(sweep)} (infer_time={infer_time})",
        )
        result = metric_forensics(collected)
        result["signal_fraction"] = _signal_fraction(infer_time)
        sweep_results[str(infer_time)] = result
        if collect_probe:
            probe_collected = collected
        print(f"  pass {position}/{len(sweep)} done (infer_time={infer_time})", flush=True)

    # The whole table is rendered once, after every pass, so the rows can be
    # compared side by side instead of arriving one at a time between bars.
    print("\n" + _render_sweep(sweep_results, sweep), flush=True)

    if probe_collected is not None:
        print(_section("5. linear probe on residual channels"), flush=True)
        probe_report = run_probe(probe_collected)
        print(_render_kv(probe_report), flush=True)

    report["sweep"] = sweep_results
    report["probe"] = probe_report
    report["default_infer_time"] = default_infer_time

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
