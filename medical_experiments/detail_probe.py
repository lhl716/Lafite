"""Run feature-detail decodability probes on cached medical representations.

The public entry point is :func:`run_detail_probe`.  It accepts a plain mapping
and returns a JSON-serializable dictionary.  A minimal configuration is::

    {
      "cache_paths": ["/path/to/cache_dir"],
      "stage_keys": ["stage_1", "stage_2"],
      "stage_layouts": {"stage_1": "NCHW", "stage_2": "NLC"},
      "stage_grids": {"stage_2": [16, 16]},
      "prefix_tokens": {"stage_2": 1},
      "target_grid": [32, 32],
      "feature_sets": {"all_stages": ["stage_1", "stage_2"]},
      "channel_budget": null,
      "probe": {"type": "linear", "epochs": 50},
      "spatial_shuffle_control": {"enabled": true}
    }

Each ``.npz`` may contain one sample or a batch.  Required metadata keys are
``sample_id``, ``patient_id``, and ``split``; the binary target defaults to
``mask``.  An NCHW stage is stored as ``C,H,W`` or ``N,C,H,W``.  An NLC stage is
stored as ``L,C`` or ``N,L,C`` and needs either a square token count or an
explicit entry in ``stage_grids``.  Prefix tokens (for example CLS) can be
removed with ``prefix_tokens``.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import scipy
import sklearn
import torch
import torch.nn.functional as torch_functional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

try:
    from .probes import (
        channel_statistics,
        coordinate_control_features,
        evaluate_segmentation,
        fixed_random_channel_projection,
        metric_residuals,
        normalize_channels,
        predict_probabilities,
        random_feature_control,
        select_threshold,
        spatial_shuffle_features,
        train_probe,
    )
except ImportError:  # pragma: no cover - permits direct script execution.
    from probes import (  # type: ignore
        channel_statistics,
        coordinate_control_features,
        evaluate_segmentation,
        fixed_random_channel_projection,
        metric_residuals,
        normalize_channels,
        predict_probabilities,
        random_feature_control,
        select_threshold,
        spatial_shuffle_features,
        train_probe,
    )


@dataclass
class RepresentationRecord:
    sample_id: str
    patient_id: str
    split: str
    mask: np.ndarray
    stages: Dict[str, np.ndarray]
    source_path: str
    coarse_label: Optional[str] = None


def _string_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _metadata_vector(
    value: np.ndarray, expected: Optional[int], key: str, allow_broadcast: bool
) -> List[str]:
    array = np.asarray(value)
    if array.ndim == 0:
        values = [_string_value(array.item())]
    else:
        values = [_string_value(item) for item in array.reshape(-1)]
    if expected is None:
        if not values:
            raise ValueError("metadata key '%s' is empty" % key)
        return values
    if len(values) == expected:
        return values
    if allow_broadcast and len(values) == 1:
        return values * expected
    raise ValueError(
        "metadata key '%s' has %d values, expected %d"
        % (key, len(values), expected)
    )


def _expand_cache_paths(paths: Any, recursive: bool = True) -> List[Path]:
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    if not isinstance(paths, Sequence) or not paths:
        raise ValueError("config.cache_paths must be a non-empty path or list of paths")
    expanded: List[Path] = []
    for raw_path in paths:
        text = os.fspath(raw_path)
        if any(character in text for character in "*?["):
            expanded.extend(Path(item) for item in glob.glob(text, recursive=recursive))
            continue
        path = Path(text)
        if path.is_dir():
            iterator = path.rglob("*.npz") if recursive else path.glob("*.npz")
            expanded.extend(iterator)
        elif path.is_file():
            expanded.append(path)
        else:
            raise FileNotFoundError("cache path does not exist: %s" % path)
    unique = sorted({path.resolve() for path in expanded if path.suffix.lower() == ".npz"})
    if not unique:
        raise ValueError("no .npz cache files were found")
    return unique


def _stage_key_map(config: Mapping[str, Any], archive: Any) -> Dict[str, str]:
    configured = config.get("stage_keys")
    if isinstance(configured, Mapping):
        result = {str(alias): str(key) for alias, key in configured.items()}
    elif configured is not None:
        if isinstance(configured, str):
            configured = [configured]
        result = {str(key): str(key) for key in configured}
    else:
        metadata = config.get("metadata_keys", {})
        reserved = {
            str(metadata.get("sample_id", "sample_id")),
            str(metadata.get("patient_id", "patient_id")),
            str(metadata.get("split", "split")),
            str(config.get("mask_key", "mask")),
            str(config.get("metadata_json_key", "metadata_json")),
            "modality",
            "label",
        }
        if config.get("coarse_label_key") is not None:
            reserved.add(str(config.get("coarse_label_key")))
        reserved.update(str(key) for key in config.get("extra_metadata_keys", []))
        result = {
            str(key): str(key)
            for key in archive.files
            if str(key) not in reserved
            and np.asarray(archive[key]).ndim >= 2
            and np.issubdtype(np.asarray(archive[key]).dtype, np.number)
        }
    if not result:
        raise ValueError("no stage arrays were selected; configure stage_keys")
    return result


def _is_square(value: int) -> bool:
    if value < 0:
        return False
    root = int(round(math.sqrt(value)))
    return root * root == value


def _layout_for_array(
    alias: str,
    key: str,
    array: np.ndarray,
    sample_count: int,
    config: Mapping[str, Any],
) -> str:
    layouts = config.get("stage_layouts", {})
    configured = None
    if isinstance(layouts, Mapping):
        configured = layouts.get(alias, layouts.get(key))
    elif isinstance(layouts, str):
        configured = layouts
    if configured is not None and str(configured).lower() != "auto":
        layout = str(configured).upper()
        if layout not in {"NCHW", "NLC"}:
            raise ValueError("stage layout for '%s' must be NCHW, NLC, or auto" % alias)
        return layout

    if array.ndim == 4:
        return "NCHW"
    if array.ndim == 2:
        return "NLC"
    if array.ndim != 3:
        raise ValueError(
            "cannot infer layout for stage '%s' with shape %s"
            % (alias, tuple(array.shape))
        )
    if sample_count > 1:
        if int(array.shape[0]) != sample_count:
            raise ValueError(
                "batched stage '%s' does not have sample dimension %d"
                % (alias, sample_count)
            )
        return "NLC"

    grids = config.get("stage_grids", {})
    grid = grids.get(alias, grids.get(key)) if isinstance(grids, Mapping) else None
    prefixes = config.get("prefix_tokens", {})
    prefix = prefixes.get(alias, prefixes.get(key, 0)) if isinstance(prefixes, Mapping) else prefixes
    prefix = int(prefix or 0)
    if int(array.shape[0]) == 1:
        if grid is not None:
            expected_tokens = int(grid[0]) * int(grid[1]) + prefix
            if int(array.shape[1]) == expected_tokens:
                return "NLC"
        if _is_square(int(array.shape[1]) - prefix) and int(array.shape[1]) != int(
            array.shape[2]
        ):
            return "NLC"
    if int(array.shape[1]) == int(array.shape[2]):
        return "NCHW"
    raise ValueError(
        "ambiguous rank-3 stage '%s' with shape %s; set stage_layouts explicitly"
        % (alias, tuple(array.shape))
    )


def _slice_stage(
    array: np.ndarray,
    layout: str,
    sample_index: int,
    sample_count: int,
    alias: str,
) -> np.ndarray:
    if layout == "NCHW":
        if array.ndim == 4:
            if int(array.shape[0]) != sample_count:
                raise ValueError("stage '%s' has an invalid batch dimension" % alias)
            return np.asarray(array[sample_index])
        if array.ndim == 3 and sample_count == 1:
            return np.asarray(array)
    elif layout == "NLC":
        if array.ndim == 3:
            if int(array.shape[0]) != sample_count:
                raise ValueError("stage '%s' has an invalid batch dimension" % alias)
            return np.asarray(array[sample_index])
        if array.ndim == 2 and sample_count == 1:
            return np.asarray(array)
    raise ValueError(
        "stage '%s' shape %s is incompatible with layout %s and %d samples"
        % (alias, tuple(array.shape), layout, sample_count)
    )


def _grid_and_prefix(
    alias: str,
    key: str,
    token_count: int,
    config: Mapping[str, Any],
) -> Tuple[int, int, int]:
    prefixes = config.get("prefix_tokens", {})
    if isinstance(prefixes, Mapping):
        configured_prefix = prefixes.get(alias, prefixes.get(key))
    else:
        configured_prefix = prefixes

    grids = config.get("stage_grids", {})
    configured_grid = grids.get(alias, grids.get(key)) if isinstance(grids, Mapping) else None
    if configured_grid is not None:
        if len(configured_grid) != 2:
            raise ValueError("stage grid for '%s' must contain [height, width]" % alias)
        height, width = int(configured_grid[0]), int(configured_grid[1])
        if height <= 0 or width <= 0:
            raise ValueError("stage grid dimensions must be positive")
        prefix = int(configured_prefix or 0)
        if height * width + prefix != token_count:
            raise ValueError(
                "stage '%s' has %d tokens, but grid %s plus %d prefix tokens expects %d"
                % (alias, token_count, (height, width), prefix, height * width + prefix)
            )
        return height, width, prefix

    if configured_prefix is not None:
        prefix = int(configured_prefix)
        spatial_tokens = token_count - prefix
        if not _is_square(spatial_tokens):
            raise ValueError(
                "stage '%s' has a non-square token grid after removing %d prefixes"
                % (alias, prefix)
            )
        side = int(round(math.sqrt(spatial_tokens)))
        return side, side, prefix

    if _is_square(token_count):
        side = int(round(math.sqrt(token_count)))
        return side, side, 0
    if _is_square(token_count - 1):
        side = int(round(math.sqrt(token_count - 1)))
        return side, side, 1
    raise ValueError(
        "cannot infer a spatial grid for NLC stage '%s' with %d tokens; set stage_grids"
        % (alias, token_count)
    )


def _to_chw(
    sample_array: np.ndarray,
    layout: str,
    alias: str,
    key: str,
    config: Mapping[str, Any],
) -> np.ndarray:
    array = np.asarray(sample_array)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("stage '%s' must contain numeric values" % alias)
    if layout == "NCHW":
        if array.ndim != 3:
            raise ValueError("NCHW sample '%s' must reduce to C x H x W" % alias)
        result = array
    else:
        if array.ndim != 2:
            raise ValueError("NLC sample '%s' must reduce to L x C" % alias)
        height, width, prefix = _grid_and_prefix(
            alias, key, int(array.shape[0]), config
        )
        spatial = array[prefix:]
        result = spatial.reshape(height, width, int(spatial.shape[1])).transpose(2, 0, 1)
    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("stage '%s' contains NaN or infinite values" % alias)
    return result


def _slice_mask(array: np.ndarray, sample_index: int, sample_count: int) -> np.ndarray:
    mask = np.asarray(array)
    if sample_count > 1:
        if mask.ndim < 3 or int(mask.shape[0]) != sample_count:
            raise ValueError("batched mask must have N as its first dimension")
        mask = mask[sample_index]
    elif mask.ndim == 4:
        if int(mask.shape[0]) != 1:
            raise ValueError("single-sample rank-4 mask must start with dimension 1")
        mask = mask[0]
    elif mask.ndim == 3 and int(mask.shape[0]) == 1:
        mask = mask[0]

    while mask.ndim > 2:
        if int(mask.shape[0]) == 1:
            mask = mask[0]
        elif int(mask.shape[-1]) == 1:
            mask = mask[..., 0]
        else:
            raise ValueError("each mask must reduce to a 2-D array")
    if mask.ndim != 2:
        raise ValueError("each mask must be 2-D")
    if not np.isfinite(mask).all():
        raise ValueError("mask contains NaN or infinite values")
    return np.asarray(mask)


def load_representation_cache(config: Mapping[str, Any]) -> List[RepresentationRecord]:
    """Load one-sample or batched NPZ representation caches."""

    paths = _expand_cache_paths(
        config.get("cache_paths", config.get("npz_paths")),
        recursive=bool(config.get("recursive", True)),
    )
    metadata = config.get("metadata_keys", {})
    sample_key = str(metadata.get("sample_id", "sample_id"))
    patient_key = str(metadata.get("patient_id", "patient_id"))
    split_key = str(metadata.get("split", "split"))
    mask_key = str(config.get("mask_key", "mask"))
    metadata_json_key = str(config.get("metadata_json_key", "metadata_json"))
    coarse_label_key = config.get("coarse_label_key")
    coarse_label_key = None if coarse_label_key is None else str(coarse_label_key)
    mask_threshold = float(config.get("mask_positive_threshold", 0.5))

    records: List[RepresentationRecord] = []
    expected_aliases: Optional[List[str]] = None
    for path in paths:
        with np.load(str(path), allow_pickle=False) as archive:
            for required in (sample_key, patient_key, split_key, mask_key):
                if required not in archive:
                    raise KeyError("%s is missing required key '%s'" % (path, required))
            sample_ids = _metadata_vector(
                archive[sample_key], expected=None, key=sample_key, allow_broadcast=False
            )
            sample_count = len(sample_ids)
            patient_ids = _metadata_vector(
                archive[patient_key],
                expected=sample_count,
                key=patient_key,
                allow_broadcast=True,
            )
            splits = _metadata_vector(
                archive[split_key],
                expected=sample_count,
                key=split_key,
                allow_broadcast=True,
            )
            if coarse_label_key is not None:
                if coarse_label_key in archive:
                    coarse_labels: List[Optional[str]] = _metadata_vector(
                        archive[coarse_label_key],
                        expected=sample_count,
                        key=coarse_label_key,
                        allow_broadcast=True,
                    )
                elif metadata_json_key in archive:
                    packed_values = _metadata_vector(
                        archive[metadata_json_key],
                        expected=sample_count,
                        key=metadata_json_key,
                        allow_broadcast=True,
                    )
                    coarse_labels = []
                    for packed in packed_values:
                        try:
                            unpacked = json.loads(packed)
                        except json.JSONDecodeError as error:
                            raise ValueError(
                                "%s contains invalid %s" % (path, metadata_json_key)
                            ) from error
                        if coarse_label_key not in unpacked:
                            raise KeyError(
                                "%s is missing coarse label '%s' in arrays and %s"
                                % (path, coarse_label_key, metadata_json_key)
                            )
                        coarse_labels.append(_string_value(unpacked[coarse_label_key]))
                else:
                    raise KeyError(
                        "%s is missing configured coarse_label_key '%s'"
                        % (path, coarse_label_key)
                    )
            else:
                coarse_labels = [None] * sample_count
            key_map = _stage_key_map(config, archive)
            for key in key_map.values():
                if key not in archive:
                    raise KeyError("%s is missing configured stage key '%s'" % (path, key))
            aliases = list(key_map.keys())
            if expected_aliases is None:
                expected_aliases = aliases
            elif aliases != expected_aliases:
                raise ValueError("all cache files must expose the same ordered stage aliases")

            stage_arrays = {
                alias: np.asarray(archive[key]) for alias, key in key_map.items()
            }
            stage_layouts = {
                alias: _layout_for_array(
                    alias, key_map[alias], stage_arrays[alias], sample_count, config
                )
                for alias in aliases
            }
            mask_array = np.asarray(archive[mask_key])
            for index in range(sample_count):
                stages: Dict[str, np.ndarray] = {}
                for alias in aliases:
                    sliced = _slice_stage(
                        stage_arrays[alias],
                        stage_layouts[alias],
                        index,
                        sample_count,
                        alias,
                    )
                    stages[alias] = _to_chw(
                        sliced,
                        stage_layouts[alias],
                        alias,
                        key_map[alias],
                        config,
                    )
                raw_mask = _slice_mask(mask_array, index, sample_count)
                binary_mask = (raw_mask > mask_threshold).astype(np.float32)
                records.append(
                    RepresentationRecord(
                        sample_id=sample_ids[index],
                        patient_id=patient_ids[index],
                        split=str(splits[index]).strip().lower(),
                        mask=binary_mask,
                        stages=stages,
                        source_path=str(path),
                        coarse_label=coarse_labels[index],
                    )
                )

    seen: Dict[str, str] = {}
    for record in records:
        if record.sample_id in seen:
            raise ValueError(
                "duplicate sample_id '%s' in %s and %s"
                % (record.sample_id, seen[record.sample_id], record.source_path)
            )
        seen[record.sample_id] = record.source_path
    return records


def _target_grid(records: Sequence[RepresentationRecord], config: Mapping[str, Any]) -> Tuple[int, int]:
    configured = config.get("target_grid")
    if configured is not None:
        if len(configured) != 2:
            raise ValueError("target_grid must contain [height, width]")
        height, width = int(configured[0]), int(configured[1])
        if height <= 0 or width <= 0:
            raise ValueError("target_grid dimensions must be positive")
        return height, width
    heights: List[int] = []
    widths: List[int] = []
    for record in records:
        for stage in record.stages.values():
            heights.append(int(stage.shape[-2]))
            widths.append(int(stage.shape[-1]))
    return max(heights), max(widths)


def _resize_feature(feature: np.ndarray, target_grid: Tuple[int, int]) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(feature, dtype=np.float32)).unsqueeze(0)
    if tuple(tensor.shape[-2:]) != tuple(target_grid):
        tensor = torch_functional.interpolate(
            tensor, size=target_grid, mode="bilinear", align_corners=False
        )
    return tensor[0]


def _resize_mask(mask: np.ndarray, target_grid: Tuple[int, int]) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None]
    if tuple(tensor.shape[-2:]) != tuple(target_grid):
        tensor = torch_functional.interpolate(tensor, size=target_grid, mode="nearest")
    return tensor[0]


def _feature_sets(
    records: Sequence[RepresentationRecord], config: Mapping[str, Any]
) -> Dict[str, List[str]]:
    stage_names = list(records[0].stages.keys())
    configured = config.get("feature_sets")
    if configured is None:
        sets = {stage: [stage] for stage in stage_names}
        if len(stage_names) > 1:
            sets["all_stages"] = list(stage_names)
        return sets
    if not isinstance(configured, Mapping) or not configured:
        raise ValueError("feature_sets must be a non-empty mapping of name to stage list")
    result: Dict[str, List[str]] = {}
    for name, stages in configured.items():
        if isinstance(stages, str):
            stages = [stages]
        selected = [str(stage) for stage in stages]
        if not selected:
            raise ValueError("feature set '%s' is empty" % name)
        missing = [stage for stage in selected if stage not in stage_names]
        if missing:
            raise KeyError("feature set '%s' references missing stages %s" % (name, missing))
        result[str(name)] = selected
    return result


def _materialize(
    records: Sequence[RepresentationRecord],
    target_grid: Tuple[int, int],
    feature_sets: Mapping[str, Sequence[str]],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    masks = torch.stack([_resize_mask(record.mask, target_grid) for record in records])
    materialized: Dict[str, torch.Tensor] = {}
    for set_name, stages in feature_sets.items():
        samples: List[torch.Tensor] = []
        for record in records:
            resized = [_resize_feature(record.stages[stage], target_grid) for stage in stages]
            samples.append(torch.cat(resized, dim=0))
        try:
            materialized[set_name] = torch.stack(samples)
        except RuntimeError as error:
            raise ValueError(
                "feature set '%s' has inconsistent channel counts across samples" % set_name
            ) from error
    return materialized, masks


def _split_indices(
    records: Sequence[RepresentationRecord], config: Mapping[str, Any]
) -> Dict[str, List[int]]:
    split_names = config.get("split_names", {})
    expected = {
        "train": str(split_names.get("train", "train")),
        "val": str(split_names.get("val", "val")),
        "test": str(split_names.get("test", "test")),
    }
    indices = {
        role: [
            index
            for index, record in enumerate(records)
            if record.split.strip().lower() == value.strip().lower()
        ]
        for role, value in expected.items()
    }
    for role, values in indices.items():
        if not values:
            raise ValueError("the configured %s split '%s' is empty" % (role, expected[role]))

    if not bool(config.get("allow_patient_overlap", False)):
        patient_splits: Dict[str, set] = {}
        for record in records:
            patient_splits.setdefault(record.patient_id, set()).add(record.split)
        overlap = {
            patient: sorted(splits)
            for patient, splits in patient_splits.items()
            if len(splits) > 1
        }
        if overlap:
            first_patient = sorted(overlap)[0]
            raise ValueError(
                "patient_id '%s' occurs in multiple splits %s"
                % (first_patient, overlap[first_patient])
            )
    return indices


def _take(tensor: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    return tensor[torch.as_tensor(indices, dtype=torch.long)]


def _take_strings(values: Sequence[str], indices: Sequence[int]) -> List[str]:
    return [str(values[index]) for index in indices]


def _device_from_config(config: Mapping[str, Any]) -> torch.device:
    requested = str(config.get("device", "cpu")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("a CUDA device was requested, but CUDA is unavailable")
    return torch.device(requested)


def _small_lesion_pixels(config: Mapping[str, Any], grid: Tuple[int, int]) -> int:
    small_config = config.get("small_lesion", {})
    configured = small_config.get("max_pixels")
    if configured is not None:
        value = int(configured)
    else:
        fraction = float(small_config.get("max_fraction", 0.01))
        if not 0.0 < fraction <= 1.0:
            raise ValueError("small_lesion.max_fraction must be in (0, 1]")
        value = int(math.ceil(grid[0] * grid[1] * fraction))
    if value <= 0:
        raise ValueError("small lesion size threshold must be positive")
    return value


def _evaluate_split(
    model: torch.nn.Module,
    features: torch.Tensor,
    masks: torch.Tensor,
    patient_ids: Sequence[str],
    threshold: float,
    device: torch.device,
    probe_config: Mapping[str, Any],
    small_config: Mapping[str, Any],
    small_pixels: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    probabilities = predict_probabilities(
        model,
        features,
        device=device,
        batch_size=int(probe_config.get("eval_batch_size", probe_config.get("batch_size", 16))),
    )
    metrics = evaluate_segmentation(
        probabilities,
        masks[:, 0].cpu().numpy(),
        patient_ids,
        threshold=threshold,
        small_lesion_max_pixels=small_pixels,
        small_lesion_min_overlap=float(small_config.get("min_overlap", 0.1)),
        connectivity=int(small_config.get("connectivity", 1)),
    )
    return probabilities, metrics


def _fit_and_evaluate(
    features: torch.Tensor,
    masks: torch.Tensor,
    records: Sequence[RepresentationRecord],
    indices: Mapping[str, Sequence[int]],
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    small_pixels: int,
) -> Dict[str, Any]:
    probe_config = dict(config.get("probe", {}))
    patient_ids = [record.patient_id for record in records]
    train_features = _take(features, indices["train"])
    val_features = _take(features, indices["val"])
    test_features = _take(features, indices["test"])
    train_masks = _take(masks, indices["train"])
    val_masks = _take(masks, indices["val"])
    test_masks = _take(masks, indices["test"])
    val_patients = _take_strings(patient_ids, indices["val"])
    test_patients = _take_strings(patient_ids, indices["test"])

    model, training = train_probe(
        train_features,
        train_masks,
        val_features,
        val_masks,
        probe_config=probe_config,
        device=device,
        seed=seed,
    )
    val_probabilities = predict_probabilities(
        model,
        val_features,
        device=device,
        batch_size=int(probe_config.get("eval_batch_size", probe_config.get("batch_size", 16))),
    )
    configured_threshold = probe_config.get("threshold")
    if configured_threshold is None or str(configured_threshold).lower() == "auto":
        candidates = probe_config.get("threshold_candidates")
        threshold, val_metrics = select_threshold(
            val_probabilities,
            val_masks[:, 0].cpu().numpy(),
            val_patients,
            small_lesion_max_pixels=small_pixels,
            candidates=candidates,
            small_lesion_min_overlap=float(
                config.get("small_lesion", {}).get("min_overlap", 0.1)
            ),
            connectivity=int(config.get("small_lesion", {}).get("connectivity", 1)),
        )
    else:
        threshold = float(configured_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("probe.threshold must be in [0, 1] or 'auto'")
        val_metrics = evaluate_segmentation(
            val_probabilities,
            val_masks[:, 0].cpu().numpy(),
            val_patients,
            threshold=threshold,
            small_lesion_max_pixels=small_pixels,
            small_lesion_min_overlap=float(
                config.get("small_lesion", {}).get("min_overlap", 0.1)
            ),
            connectivity=int(config.get("small_lesion", {}).get("connectivity", 1)),
        )

    _, test_metrics = _evaluate_split(
        model,
        test_features,
        test_masks,
        test_patients,
        threshold,
        device,
        probe_config,
        config.get("small_lesion", {}),
        small_pixels,
    )
    return {
        "training": training,
        "selected_threshold": float(threshold),
        "metrics": {"val": val_metrics, "test": test_metrics},
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    raise TypeError("result contains a non-JSON-serializable value: %r" % (type(value),))


def _classification_metrics(labels: Sequence[str], predictions: Sequence[str]) -> Dict[str, Any]:
    return {
        "num_samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def _run_coarse_label_probe(
    stage_features: Mapping[str, torch.Tensor],
    records: Sequence[RepresentationRecord],
    indices: Mapping[str, Sequence[int]],
    config: Mapping[str, Any],
    seed: int,
    channel_budget: Optional[int],
    projection_type: str,
    projection_seed: int,
) -> Dict[str, Any]:
    """Fit train-only GAP linear classifiers, selecting regularization on val."""

    if config.get("coarse_label_key") is None:
        return {"enabled": False, "reason": "coarse_label_key is not configured"}
    labels = [record.coarse_label for record in records]
    if any(label is None for label in labels):
        raise ValueError("coarse_label_key is configured but at least one label is missing")
    string_labels = [str(label) for label in labels]
    classifier_config = config.get("coarse_label_probe", {})
    c_values = [float(value) for value in classifier_config.get("c_values", [0.01, 0.1, 1.0, 10.0])]
    if not c_values or any(value <= 0.0 for value in c_values):
        raise ValueError("coarse_label_probe.c_values must contain positive values")
    max_iter = int(classifier_config.get("max_iter", 1000))
    class_weight = classifier_config.get("class_weight")

    train_labels = _take_strings(string_labels, indices["train"])
    val_labels = _take_strings(string_labels, indices["val"])
    test_labels = _take_strings(string_labels, indices["test"])
    if len(set(train_labels)) < 2:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "the train split contains fewer than two coarse-label classes",
            "train_classes": sorted(set(train_labels)),
        }

    stage_results: Dict[str, Any] = {}
    for stage_index, (stage_name, raw_features) in enumerate(stage_features.items()):
        original_channels = int(raw_features.shape[1])
        if channel_budget is not None:
            projected, projection = fixed_random_channel_projection(
                raw_features,
                channel_budget=channel_budget,
                seed=projection_seed,
                projection_type=projection_type,
            )
        else:
            projected = raw_features
            projection = {
                "enabled": False,
                "type": None,
                "seed": None,
                "trainable": False,
                "original_channels": original_channels,
                "output_channels": original_channels,
                "relation": "unchanged",
            }
        pooled = projected.mean(dim=(2, 3)).cpu().numpy()
        train_values = pooled[np.asarray(indices["train"], dtype=np.int64)]
        val_values = pooled[np.asarray(indices["val"], dtype=np.int64)]
        test_values = pooled[np.asarray(indices["test"], dtype=np.int64)]
        scaler = StandardScaler()
        train_values = scaler.fit_transform(train_values)
        val_values = scaler.transform(val_values)
        test_values = scaler.transform(test_values)

        candidates: List[Tuple[float, float, LogisticRegression, Dict[str, Any]]] = []
        for c_value in c_values:
            classifier = LogisticRegression(
                C=c_value,
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=seed + stage_index,
                solver="lbfgs",
            )
            classifier.fit(train_values, train_labels)
            val_predictions = classifier.predict(val_values)
            val_metrics = _classification_metrics(val_labels, val_predictions)
            candidates.append((val_metrics["macro_f1"], c_value, classifier, val_metrics))
        candidates.sort(key=lambda item: (-item[0], abs(math.log10(item[1]))))
        _, selected_c, classifier, val_metrics = candidates[0]
        test_predictions = classifier.predict(test_values)
        test_metrics = _classification_metrics(test_labels, test_predictions)
        stage_results[stage_name] = {
            "status": "ok",
            "original_input_channels": original_channels,
            "input_channels": int(projected.shape[1]),
            "channel_projection": projection,
            "pooling": "global_average",
            "classifier": "sklearn.linear_model.LogisticRegression",
            "selected_c": float(selected_c),
            "selection_metric": "val_macro_f1",
            "classes": [str(value) for value in classifier.classes_.tolist()],
            "parameter_count": int(classifier.coef_.size + classifier.intercept_.size),
            "metrics": {"val": val_metrics, "test": test_metrics},
        }
    return {
        "enabled": True,
        "status": "ok",
        "label_key": str(config.get("coarse_label_key")),
        "stages": stage_results,
    }


def run_detail_probe(config_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Train detail probes and return a JSON-serializable result dictionary.

    The spatial-shuffle control retrains an equal-capacity probe after applying
    an independent, stable spatial permutation to every sample.  Reported
    residuals are ``observed_metric - shuffled_metric``.
    """

    if not isinstance(config_dict, Mapping):
        raise TypeError("run_detail_probe expects a mapping")
    config: Dict[str, Any] = dict(config_dict)
    seed = int(config.get("seed", 133))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if "torch_num_threads" in config:
        torch.set_num_threads(max(1, int(config["torch_num_threads"])))

    device = _device_from_config(config)
    records = load_representation_cache(config)
    grid = _target_grid(records, config)
    feature_sets = _feature_sets(records, config)
    feature_tensors, masks = _materialize(records, grid, feature_sets)
    indices = _split_indices(records, config)
    small_pixels = _small_lesion_pixels(config, grid)
    normalize = bool(config.get("normalize_features", True))
    sample_ids = [record.sample_id for record in records]

    shuffle_config = config.get("spatial_shuffle_control", {})
    if isinstance(shuffle_config, bool):
        shuffle_enabled = shuffle_config
        shuffle_config = {}
    else:
        shuffle_enabled = bool(shuffle_config.get("enabled", True))
    shuffle_seed = int(shuffle_config.get("seed", seed + 1009))

    coordinate_config = config.get("coordinate_control", {})
    if isinstance(coordinate_config, bool):
        coordinate_enabled = coordinate_config
        coordinate_config = {}
    else:
        coordinate_enabled = bool(coordinate_config.get("enabled", True))
    coordinate_seed = int(coordinate_config.get("seed", seed + 3011))

    random_config = config.get("random_feature_control", {})
    if isinstance(random_config, bool):
        random_enabled = random_config
        random_config = {}
    else:
        random_enabled = bool(random_config.get("enabled", True))
    random_seed = int(random_config.get("seed", seed + 4001))

    configured_budget = config.get("channel_budget")
    channel_budget = None if configured_budget is None else int(configured_budget)
    if channel_budget is not None and channel_budget <= 0:
        raise ValueError("channel_budget must be a positive integer or null")
    projection_config = config.get("channel_projection", {})
    if isinstance(projection_config, str):
        projection_type = projection_config
        projection_seed = seed + 2027
    else:
        projection_type = str(projection_config.get("type", "orthogonal"))
        projection_seed = int(projection_config.get("seed", seed + 2027))

    set_results: Dict[str, Any] = {}
    for set_index, (set_name, stages) in enumerate(feature_sets.items()):
        raw_features = feature_tensors[set_name]
        original_channels = int(raw_features.shape[1])
        if channel_budget is not None:
            raw_features, projection_summary = fixed_random_channel_projection(
                raw_features,
                channel_budget=channel_budget,
                seed=projection_seed,
                projection_type=projection_type,
            )
        else:
            projection_summary = {
                "enabled": False,
                "type": None,
                "seed": None,
                "trainable": False,
                "original_channels": original_channels,
                "output_channels": original_channels,
                "relation": "unchanged",
            }
        if normalize:
            train_features = _take(raw_features, indices["train"])
            mean, std = channel_statistics(train_features)
            features = normalize_channels(raw_features, mean, std)
            normalization_summary = {
                "enabled": True,
                "min_channel_std": float(std.min().item()),
                "max_channel_std": float(std.max().item()),
            }
        else:
            features = raw_features
            normalization_summary = {"enabled": False}

        observed = _fit_and_evaluate(
            features,
            masks,
            records,
            indices,
            config,
            device,
            seed=seed + set_index * 101,
            small_pixels=small_pixels,
        )
        result: Dict[str, Any] = {
            "stages": list(stages),
            "original_input_channels": int(original_channels),
            "input_channels": int(features.shape[1]),
            "channel_projection": projection_summary,
            "normalization": normalization_summary,
            **observed,
        }

        if shuffle_enabled:
            shuffled = spatial_shuffle_features(features, sample_ids, shuffle_seed)
            control = _fit_and_evaluate(
                shuffled,
                masks,
                records,
                indices,
                config,
                device,
                seed=seed + set_index * 101 + 100000,
                small_pixels=small_pixels,
            )
            result["spatial_shuffle_control"] = {
                "enabled": True,
                "seed": int(shuffle_seed),
                "retrained_equal_capacity_probe": True,
                **control,
                "metric_residuals": {
                    split: metric_residuals(
                        observed["metrics"][split], control["metrics"][split]
                    )
                    for split in ("val", "test")
                },
            }
        else:
            result["spatial_shuffle_control"] = {"enabled": False}

        if coordinate_enabled:
            coordinate_features, coordinate_summary = coordinate_control_features(
                sample_count=int(features.shape[0]),
                channels=int(features.shape[1]),
                height=int(features.shape[2]),
                width=int(features.shape[3]),
                seed=coordinate_seed,
                include_radius=bool(coordinate_config.get("include_radius", True)),
            )
            if normalize:
                coordinate_mean, coordinate_std = channel_statistics(
                    _take(coordinate_features, indices["train"])
                )
                coordinate_features = normalize_channels(
                    coordinate_features, coordinate_mean, coordinate_std
                )
            coordinate_result = _fit_and_evaluate(
                coordinate_features,
                masks,
                records,
                indices,
                config,
                device,
                seed=seed + set_index * 101 + 200000,
                small_pixels=small_pixels,
            )
            if (
                coordinate_result["training"]["parameter_count"]
                != observed["training"]["parameter_count"]
            ):
                raise RuntimeError("coordinate control did not match probe parameter count")
            result["coordinate_control"] = {
                "enabled": True,
                "seed": coordinate_seed,
                "matched_probe_parameter_count": True,
                "feature_construction": coordinate_summary,
                **coordinate_result,
                "metric_residuals": {
                    split: metric_residuals(
                        observed["metrics"][split], coordinate_result["metrics"][split]
                    )
                    for split in ("val", "test")
                },
            }
        else:
            result["coordinate_control"] = {"enabled": False}

        if random_enabled:
            random_features, random_summary = random_feature_control(
                features, indices["train"], seed=random_seed
            )
            random_result = _fit_and_evaluate(
                random_features,
                masks,
                records,
                indices,
                config,
                device,
                seed=seed + set_index * 101 + 300000,
                small_pixels=small_pixels,
            )
            if (
                random_result["training"]["parameter_count"]
                != observed["training"]["parameter_count"]
            ):
                raise RuntimeError("random-feature control did not match probe parameter count")
            result["random_feature_control"] = {
                "enabled": True,
                "seed": random_seed,
                "matched_probe_parameter_count": True,
                "feature_construction": random_summary,
                **random_result,
                "metric_residuals": {
                    split: metric_residuals(
                        observed["metrics"][split], random_result["metrics"][split]
                    )
                    for split in ("val", "test")
                },
            }
        else:
            result["random_feature_control"] = {"enabled": False}
        set_results[set_name] = result

    stage_only_sets = {stage: [stage] for stage in records[0].stages.keys()}
    stage_only_features, _ = _materialize(records, grid, stage_only_sets)
    coarse_label_result = _run_coarse_label_probe(
        stage_only_features,
        records,
        indices,
        config,
        seed=seed + 500000,
        channel_budget=channel_budget,
        projection_type=projection_type,
        projection_seed=projection_seed,
    )

    split_summary = {
        role: {
            "samples": int(len(values)),
            "patients": int(len({records[index].patient_id for index in values})),
            "positive_samples": int(sum(bool(records[index].mask.any()) for index in values)),
        }
        for role, values in indices.items()
    }
    result = {
        "schema_version": 1,
        "seed": int(seed),
        "device": str(device),
        "dependencies": {
            "python_compatible": ">=3.8",
            "numpy": str(np.__version__),
            "torch": str(torch.__version__),
            "sklearn": str(sklearn.__version__),
            "scipy": str(scipy.__version__),
        },
        "data": {
            "num_cache_files": int(len({record.source_path for record in records})),
            "num_samples": int(len(records)),
            "target_grid": [int(grid[0]), int(grid[1])],
            "small_lesion_max_pixels": int(small_pixels),
            "splits": split_summary,
        },
        "feature_sets": set_results,
        "coarse_label_probe": coarse_label_result,
    }
    json_result = _jsonable(result)
    json.dumps(json_result, allow_nan=False)
    return json_result


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run LafitE detail-decoding probes")
    parser.add_argument("--config", required=True, help="Path to a JSON config")
    parser.add_argument("--output", help="Optional JSON result path; stdout otherwise")
    arguments = parser.parse_args(argv)
    with open(arguments.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    result = run_detail_probe(config)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if arguments.output:
        output_path = Path(arguments.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
