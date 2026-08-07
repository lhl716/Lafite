"""Small segmentation probes and patient-level metrics for detail decoding.

This module intentionally contains no LafitE-specific imports.  It operates on
already cached, spatial feature tensors of shape ``N x C x H x W`` and binary
masks of shape ``N x 1 x H x W``.  The helpers are CPU-safe and use CUDA only
when a caller explicitly selects an available CUDA device.
"""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


class LinearProbe(nn.Module):
    """A per-location linear decoder implemented as a 1x1 convolution."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class ShallowProbe(nn.Module):
    """A deliberately small local decoder with one hidden convolution."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)


def build_probe(in_channels: int, config: Mapping[str, Any]) -> nn.Module:
    """Construct a linear or shallow probe from a JSON-compatible config."""

    probe_type = str(config.get("type", "linear")).lower()
    if probe_type == "linear":
        return LinearProbe(in_channels)
    if probe_type == "shallow":
        return ShallowProbe(
            in_channels=in_channels,
            hidden_channels=int(config.get("hidden_channels", 32)),
            kernel_size=int(config.get("kernel_size", 3)),
            dropout=float(config.get("dropout", 0.0)),
        )
    raise ValueError("probe.type must be either 'linear' or 'shallow'")


def _soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    probabilities = probabilities.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)
    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


def bce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return the weighted BCE+Dice objective and detached components."""

    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight
    )
    dice = _soft_dice_loss(logits, targets)
    total = float(bce_weight) * bce + float(dice_weight) * dice
    return total, {
        "loss": float(total.detach().cpu().item()),
        "bce": float(bce.detach().cpu().item()),
        "dice_loss": float(dice.detach().cpu().item()),
    }


def channel_statistics(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute train-only per-channel mean and standard deviation."""

    if features.ndim != 4:
        raise ValueError("features must have shape N x C x H x W")
    mean = features.mean(dim=(0, 2, 3), keepdim=True)
    std = features.std(dim=(0, 2, 3), keepdim=True, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def normalize_channels(
    features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return (features - mean) / std


def fixed_random_channel_projection(
    features: torch.Tensor,
    channel_budget: int,
    seed: int,
    projection_type: str = "orthogonal",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Project every spatial vector to a fixed, non-trainable channel budget.

    ``orthogonal`` uses a semi-orthogonal Gaussian matrix obtained with QR. If
    the source has fewer channels than the budget, the matrix has orthonormal
    rows, so the operation embeds (rather than duplicates) the source vector in
    the larger space. ``gaussian`` uses a Johnson-Lindenstrauss-style matrix
    scaled by ``1 / sqrt(channel_budget)``.  Both cases produce exactly
    ``channel_budget`` channels and therefore equalize downstream probe size.
    """

    if features.ndim != 4:
        raise ValueError("features must have shape N x C x H x W")
    budget = int(channel_budget)
    if budget <= 0:
        raise ValueError("channel_budget must be a positive integer or null")
    source_channels = int(features.shape[1])
    kind = str(projection_type).lower()
    if kind not in {"orthogonal", "gaussian"}:
        raise ValueError("channel projection type must be 'orthogonal' or 'gaussian'")

    if source_channels == budget:
        return features, {
            "enabled": True,
            "type": "identity_equal_budget",
            "seed": int(seed),
            "trainable": False,
            "original_channels": source_channels,
            "output_channels": budget,
            "relation": "equal",
        }

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    if kind == "orthogonal":
        if source_channels > budget:
            gaussian = torch.randn(
                source_channels, budget, generator=generator, dtype=torch.float32
            )
            if hasattr(torch, "linalg") and hasattr(torch.linalg, "qr"):
                matrix = torch.linalg.qr(gaussian, mode="reduced").Q
            else:  # pragma: no cover - compatibility with older project torch.
                matrix = torch.qr(gaussian, some=True).Q
            relation = "reduce"
        else:
            gaussian = torch.randn(
                budget, source_channels, generator=generator, dtype=torch.float32
            )
            # Q is budget x source_channels; Q.T has orthonormal rows.
            if hasattr(torch, "linalg") and hasattr(torch.linalg, "qr"):
                matrix = torch.linalg.qr(gaussian, mode="reduced").Q.transpose(0, 1)
            else:  # pragma: no cover - compatibility with older project torch.
                matrix = torch.qr(gaussian, some=True).Q.transpose(0, 1)
            relation = "expand"
        recorded_type = "semi_orthogonal_gaussian_qr"
    else:
        matrix = torch.randn(
            source_channels, budget, generator=generator, dtype=torch.float32
        ) / math.sqrt(float(budget))
        relation = "reduce" if source_channels > budget else "expand"
        recorded_type = "scaled_gaussian"

    matrix = matrix.to(dtype=features.dtype, device=features.device)
    projected = torch.einsum("nchw,ck->nkhw", features, matrix)
    return projected, {
        "enabled": True,
        "type": recorded_type,
        "seed": int(seed),
        "trainable": False,
        "original_channels": source_channels,
        "output_channels": budget,
        "relation": relation,
    }


def coordinate_control_features(
    sample_count: int,
    channels: int,
    height: int,
    width: int,
    seed: int,
    include_radius: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Create a fixed coordinate-only control with matched output channels."""

    if min(int(sample_count), int(channels), int(height), int(width)) <= 0:
        raise ValueError("coordinate-control dimensions must be positive")
    y_axis = torch.linspace(-1.0, 1.0, int(height), dtype=torch.float32)
    x_axis = torch.linspace(-1.0, 1.0, int(width), dtype=torch.float32)
    y_grid = y_axis[:, None].expand(height, width)
    x_grid = x_axis[None, :].expand(height, width)
    bases = [x_grid, y_grid]
    if include_radius:
        bases.append(torch.sqrt(x_grid.square() + y_grid.square()) / math.sqrt(2.0))
    base = torch.stack(bases, dim=0).unsqueeze(0).expand(sample_count, -1, -1, -1)
    matched, projection = fixed_random_channel_projection(
        base,
        channel_budget=int(channels),
        seed=int(seed),
        projection_type="orthogonal",
    )
    return matched, {
        "base_channels": int(len(bases)),
        "include_radius": bool(include_radius),
        "coordinate_range": [-1.0, 1.0],
        "channel_matching": projection,
    }


def random_feature_control(
    observed_features: torch.Tensor,
    train_indices: Sequence[int],
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Generate equal-shape noise matched to train-only channel moments."""

    if observed_features.ndim != 4:
        raise ValueError("observed_features must have shape N x C x H x W")
    if not train_indices:
        raise ValueError("train_indices cannot be empty")
    index_tensor = torch.as_tensor(train_indices, dtype=torch.long)
    mean, std = channel_statistics(observed_features[index_tensor])
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    noise = torch.randn(
        tuple(observed_features.shape), generator=generator, dtype=torch.float32
    ).to(dtype=observed_features.dtype, device=observed_features.device)
    generated = noise * std.to(observed_features.device) + mean.to(observed_features.device)
    return generated, {
        "seed": int(seed),
        "shape": [int(value) for value in observed_features.shape],
        "moment_source": "observed_train_split_per_channel",
        "uses_val_or_test_statistics": False,
        "train_channel_mean_min": float(mean.min().item()),
        "train_channel_mean_max": float(mean.max().item()),
        "train_channel_std_min": float(std.min().item()),
        "train_channel_std_max": float(std.max().item()),
    }


def _resolve_pos_weight(masks: torch.Tensor, configured: Any, maximum: float) -> float:
    if configured is None or configured is False:
        return 1.0
    if isinstance(configured, str):
        if configured.lower() != "auto":
            raise ValueError("probe.pos_weight must be a number, null, or 'auto'")
        positives = float(masks.sum().item())
        total = float(masks.numel())
        if positives <= 0.0 or positives >= total:
            return 1.0
        return float(min(maximum, max(1.0, (total - positives) / positives)))
    value = float(configured)
    if value <= 0.0:
        raise ValueError("probe.pos_weight must be positive")
    return value


def _mean_epoch_loss(
    model: nn.Module,
    features: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
    pos_weight: torch.Tensor,
    bce_weight: float,
    dice_weight: float,
    batch_size: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for start in range(0, int(features.shape[0]), batch_size):
            end = min(start + batch_size, int(features.shape[0]))
            batch_features = features[start:end].to(device)
            batch_masks = masks[start:end].to(device)
            logits = model(batch_features)
            loss, _ = bce_dice_loss(
                logits,
                batch_masks,
                pos_weight=pos_weight,
                bce_weight=bce_weight,
                dice_weight=dice_weight,
            )
            count = end - start
            total_loss += float(loss.item()) * count
            total_samples += count
    return total_loss / max(total_samples, 1)


def train_probe(
    train_features: torch.Tensor,
    train_masks: torch.Tensor,
    val_features: torch.Tensor,
    val_masks: torch.Tensor,
    probe_config: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Fit a small probe with BCE+Dice and validation-loss early stopping."""

    if train_features.ndim != 4 or train_masks.ndim != 4:
        raise ValueError("train tensors must be NCHW")
    if train_features.shape[0] == 0:
        raise ValueError("the train split is empty")
    if val_features.shape[0] == 0:
        raise ValueError("the validation split is empty")
    if train_features.shape[-2:] != train_masks.shape[-2:]:
        raise ValueError("feature and mask grids must match")

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    model = build_probe(int(train_features.shape[1]), probe_config).to(device)
    learning_rate = float(probe_config.get("learning_rate", 1e-3))
    weight_decay = float(probe_config.get("weight_decay", 0.0))
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    epochs = int(probe_config.get("epochs", 50))
    if epochs <= 0:
        raise ValueError("probe.epochs must be positive")
    batch_size = max(1, int(probe_config.get("batch_size", 16)))
    patience = max(1, int(probe_config.get("patience", 10)))
    min_delta = float(probe_config.get("min_delta", 1e-5))
    bce_weight = float(probe_config.get("bce_weight", 1.0))
    dice_weight = float(probe_config.get("dice_weight", 1.0))
    if bce_weight < 0.0 or dice_weight < 0.0 or bce_weight + dice_weight <= 0.0:
        raise ValueError("BCE/Dice weights must be non-negative and not both zero")
    maximum_pos_weight = float(probe_config.get("max_pos_weight", 100.0))
    if maximum_pos_weight <= 0.0:
        raise ValueError("probe.max_pos_weight must be positive")
    pos_weight_value = _resolve_pos_weight(
        train_masks,
        probe_config.get("pos_weight", "auto"),
        maximum=maximum_pos_weight,
    )
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    dataset = TensorDataset(train_features, train_masks)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"loss": 0.0, "bce": 0.0, "dice_loss": 0.0}
        seen = 0
        for batch_features, batch_masks in loader:
            batch_features = batch_features.to(device)
            batch_masks = batch_masks.to(device)
            optimizer.zero_grad()
            logits = model(batch_features)
            loss, components = bce_dice_loss(
                logits,
                batch_masks,
                pos_weight=pos_weight,
                bce_weight=bce_weight,
                dice_weight=dice_weight,
            )
            loss.backward()
            optimizer.step()
            count = int(batch_features.shape[0])
            seen += count
            for key in running:
                running[key] += components[key] * count

        val_loss = _mean_epoch_loss(
            model,
            val_features,
            val_masks,
            device,
            pos_weight,
            bce_weight,
            dice_weight,
            batch_size,
        )
        entry = {
            "epoch": float(epoch),
            "train_loss": running["loss"] / max(seen, 1),
            "train_bce": running["bce"] / max(seen, 1),
            "train_dice_loss": running["dice_loss"] / max(seen, 1),
            "val_loss": float(val_loss),
        }
        history.append(entry)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = float(val_loss)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "epochs_ran": int(len(history)),
        "pos_weight": float(pos_weight_value),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "history": history,
    }


def predict_probabilities(
    model: nn.Module,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Return probabilities as a float32 N x H x W NumPy array."""

    model.eval()
    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(features.shape[0]), max(1, int(batch_size))):
            batch = features[start : start + batch_size].to(device)
            probabilities = torch.sigmoid(model(batch))[:, 0]
            predictions.append(probabilities.detach().cpu().numpy().astype(np.float32))
    if not predictions:
        return np.empty((0,) + tuple(features.shape[-2:]), dtype=np.float32)
    return np.concatenate(predictions, axis=0)


def binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction_bool = np.asarray(prediction, dtype=bool)
    target_bool = np.asarray(target, dtype=bool)
    denominator = int(prediction_bool.sum()) + int(target_bool.sum())
    if denominator == 0:
        return 1.0
    intersection = int(np.logical_and(prediction_bool, target_bool).sum())
    return float(2.0 * intersection / denominator)


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _safe_pixel_metrics(
    probabilities: np.ndarray, targets: np.ndarray
) -> Tuple[Optional[float], Optional[float]]:
    y_score = probabilities.reshape(-1)
    y_true = targets.reshape(-1).astype(np.uint8)
    if np.unique(y_true).size < 2:
        return None, None
    return (
        float(roc_auc_score(y_true, y_score)),
        float(average_precision_score(y_true, y_score)),
    )


def evaluate_segmentation(
    probabilities: np.ndarray,
    masks: np.ndarray,
    patient_ids: Sequence[str],
    threshold: float,
    small_lesion_max_pixels: int,
    small_lesion_min_overlap: float = 0.1,
    connectivity: int = 1,
) -> Dict[str, Any]:
    """Evaluate hard Dice and connected-component small-lesion sensitivity.

    Patient Dice is computed after concatenating all slices belonging to a
    patient, which prevents patients with many slices from dominating the
    reported mean.  Small lesions are 2-D connected components no larger than
    ``small_lesion_max_pixels`` on the unified feature grid.
    """

    probabilities = np.asarray(probabilities, dtype=np.float32)
    masks = np.asarray(masks) > 0.5
    if probabilities.shape != masks.shape:
        raise ValueError("probabilities and masks must have identical N x H x W shape")
    if len(patient_ids) != int(probabilities.shape[0]):
        raise ValueError("patient_ids length does not match the number of samples")
    if small_lesion_max_pixels <= 0:
        raise ValueError("small_lesion_max_pixels must be positive")
    if not 0.0 <= small_lesion_min_overlap <= 1.0:
        raise ValueError("small_lesion_min_overlap must be in [0, 1]")
    if int(connectivity) not in (1, 2):
        raise ValueError("2-D connectivity must be 1 (4-neighbour) or 2 (8-neighbour)")

    predictions = probabilities >= float(threshold)
    sample_dice = [
        binary_dice(predictions[index], masks[index])
        for index in range(int(probabilities.shape[0]))
    ]
    positive_sample_dice = [
        sample_dice[index]
        for index in range(len(sample_dice))
        if bool(masks[index].any())
    ]

    grouped_indices: Dict[str, List[int]] = {}
    for index, patient_id in enumerate(patient_ids):
        grouped_indices.setdefault(str(patient_id), []).append(index)

    patient_dice: List[float] = []
    positive_patient_dice: List[float] = []
    patient_small_sensitivity: List[float] = []
    patient_rows: List[Dict[str, Any]] = []
    small_count = 0
    small_detected = 0
    structure = ndimage.generate_binary_structure(2, int(connectivity))

    for patient_id, indices in grouped_indices.items():
        patient_prediction = predictions[indices].reshape(-1)
        patient_mask = masks[indices].reshape(-1)
        dice = binary_dice(patient_prediction, patient_mask)
        patient_dice.append(dice)
        if bool(patient_mask.any()):
            positive_patient_dice.append(dice)

        current_count = 0
        current_detected = 0
        for index in indices:
            components, component_count = ndimage.label(masks[index], structure=structure)
            for component_id in range(1, int(component_count) + 1):
                component = components == component_id
                component_size = int(component.sum())
                if component_size == 0 or component_size > int(small_lesion_max_pixels):
                    continue
                current_count += 1
                overlap = int(np.logical_and(predictions[index], component).sum())
                required = max(1, int(math.ceil(component_size * small_lesion_min_overlap)))
                if overlap >= required:
                    current_detected += 1
        if current_count > 0:
            patient_small_sensitivity.append(current_detected / current_count)
            small_count += current_count
            small_detected += current_detected
        patient_rows.append(
            {
                "patient_id": str(patient_id),
                "num_samples": int(len(indices)),
                "positive": bool(patient_mask.any()),
                "dice": float(dice),
                "small_lesion_count": int(current_count),
                "small_lesion_detected": int(current_detected),
                "small_lesion_sensitivity": (
                    float(current_detected / current_count) if current_count > 0 else None
                ),
            }
        )

    pixel_roc_auc, pixel_average_precision = _safe_pixel_metrics(probabilities, masks)
    return {
        "threshold": float(threshold),
        "num_samples": int(probabilities.shape[0]),
        "num_patients": int(len(grouped_indices)),
        "num_positive_samples": int(sum(bool(mask.any()) for mask in masks)),
        "num_positive_patients": int(len(positive_patient_dice)),
        "sample_dice_mean": _safe_mean(sample_dice),
        "positive_sample_dice_mean": _safe_mean(positive_sample_dice),
        "patient_dice_mean": _safe_mean(patient_dice),
        "patient_dice_median": _safe_median(patient_dice),
        "positive_patient_dice_mean": _safe_mean(positive_patient_dice),
        "small_lesion_max_pixels": int(small_lesion_max_pixels),
        "small_lesion_count": int(small_count),
        "small_lesion_detected": int(small_detected),
        "small_lesion_sensitivity_micro": (
            float(small_detected / small_count) if small_count > 0 else None
        ),
        "small_lesion_sensitivity_patient_mean": _safe_mean(
            patient_small_sensitivity
        ),
        "pixel_roc_auc": pixel_roc_auc,
        "pixel_average_precision": pixel_average_precision,
        "per_patient": patient_rows,
    }


def select_threshold(
    probabilities: np.ndarray,
    masks: np.ndarray,
    patient_ids: Sequence[str],
    small_lesion_max_pixels: int,
    candidates: Optional[Iterable[float]] = None,
    small_lesion_min_overlap: float = 0.1,
    connectivity: int = 1,
) -> Tuple[float, Dict[str, Any]]:
    """Choose a validation threshold by positive-patient Dice.

    Ties are broken in favour of the threshold nearest 0.5 and then the lower
    threshold, making the choice deterministic.
    """

    if candidates is None:
        candidates = np.linspace(0.05, 0.95, 19).tolist()
    scored: List[Tuple[float, float, Dict[str, Any]]] = []
    for candidate in candidates:
        value = float(candidate)
        if not 0.0 <= value <= 1.0:
            raise ValueError("threshold candidates must be in [0, 1]")
        metrics = evaluate_segmentation(
            probabilities,
            masks,
            patient_ids,
            threshold=value,
            small_lesion_max_pixels=small_lesion_max_pixels,
            small_lesion_min_overlap=small_lesion_min_overlap,
            connectivity=connectivity,
        )
        score = metrics["positive_patient_dice_mean"]
        if score is None:
            score = metrics["patient_dice_mean"]
        scored.append((float(score or 0.0), value, metrics))
    if not scored:
        raise ValueError("at least one threshold candidate is required")
    scored.sort(key=lambda item: (-item[0], abs(item[1] - 0.5), item[1]))
    _, threshold, metrics = scored[0]
    return float(threshold), metrics


def spatial_shuffle_features(
    features: torch.Tensor,
    sample_ids: Sequence[str],
    seed: int,
) -> torch.Tensor:
    """Independently permute each sample's spatial grid, preserving channels.

    The permutation seed is derived from ``seed`` and ``sample_id`` with
    SHA-256, so it is stable across Python processes and hash randomization.
    """

    if features.ndim != 4:
        raise ValueError("features must have shape N x C x H x W")
    if len(sample_ids) != int(features.shape[0]):
        raise ValueError("sample_ids length does not match features")
    sample_count, channels, height, width = features.shape
    flattened = features.reshape(sample_count, channels, height * width)
    shuffled = torch.empty_like(flattened)
    for index, sample_id in enumerate(sample_ids):
        digest = hashlib.sha256(
            (str(int(seed)) + "::" + str(sample_id)).encode("utf-8")
        ).digest()
        local_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        generator = np.random.default_rng(local_seed)
        permutation = torch.as_tensor(
            generator.permutation(height * width), dtype=torch.long
        )
        shuffled[index] = flattened[index, :, permutation]
    return shuffled.reshape(sample_count, channels, height, width)


def metric_residuals(
    observed: Mapping[str, Any], control: Mapping[str, Any]
) -> Dict[str, float]:
    """Subtract numeric control metrics from observed metrics."""

    residuals: Dict[str, float] = {}
    for key, observed_value in observed.items():
        control_value = control.get(key)
        if isinstance(observed_value, (int, float)) and not isinstance(
            observed_value, bool
        ):
            if isinstance(control_value, (int, float)) and not isinstance(
                control_value, bool
            ):
                if math.isfinite(float(observed_value)) and math.isfinite(
                    float(control_value)
                ):
                    residuals[key] = float(observed_value) - float(control_value)
    return residuals
