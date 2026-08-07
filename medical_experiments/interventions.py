"""Counterfactual lesion/background interventions for 2-D and 3-D images.

The module deliberately avoids constant/black occlusion.  Lesion removal uses
either a matched normal reference or a normalized local-Gaussian estimate.
Background nuisance interventions are prevented from touching a dilated lesion
region.  All masks are treated as soft masks in ``[0, 1]``.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

try:  # Torch is optional at import time, but supported when installed.
    import torch
except ImportError:  # pragma: no cover - the project normally provides torch.
    torch = None  # type: ignore


ArrayLike = Any


def _as_numpy(value: ArrayLike, name: str) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim not in (2, 3):
        raise ValueError("%s must be a 2-D or 3-D array; got shape %r" % (name, array.shape))
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError("%s must contain numeric values" % name)
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError("%s contains NaN or infinite values" % name)
    return array


def _restore_type(array: np.ndarray, like: ArrayLike) -> ArrayLike:
    """Return a floating result on the input tensor device when appropriate."""
    if torch is not None and isinstance(like, torch.Tensor):
        dtype = like.dtype if like.is_floating_point() else torch.float32
        return torch.as_tensor(array, dtype=dtype, device=like.device)
    original = np.asarray(like)
    dtype = original.dtype if np.issubdtype(original.dtype, np.floating) else np.float32
    return array.astype(dtype, copy=False)


def normalize_soft_mask(mask: ArrayLike, shape: Sequence[int]) -> np.ndarray:
    """Validate and clip a 2-D/3-D soft mask to ``[0, 1]``."""
    result = _as_numpy(mask, "mask")
    if tuple(result.shape) != tuple(shape):
        raise ValueError("mask shape %r does not match image shape %r" % (result.shape, tuple(shape)))
    return np.clip(result, 0.0, 1.0)


def dilate_soft_mask(mask: ArrayLike, radius: int = 3) -> np.ndarray:
    """Dilate a soft mask using grey-scale maximum filtering.

    Grey-scale dilation retains partial-volume mask values rather than first
    forcing the mask to be binary.
    """
    array = _as_numpy(mask, "mask")
    array = np.clip(array, 0.0, 1.0)
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return array.copy()
    size = tuple(2 * radius + 1 for _ in range(array.ndim))
    return ndimage.maximum_filter(array, size=size, mode="nearest")


def _local_gaussian_reference(
    image: np.ndarray,
    lesion_mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Estimate normal tissue from neighboring non-lesion pixels/voxels."""
    if sigma <= 0:
        raise ValueError("gaussian_sigma must be positive")
    outside = 1.0 - lesion_mask
    if float(np.sum(outside)) <= 1e-6:
        raise ValueError("local Gaussian replacement needs non-lesion context")
    numerator = ndimage.gaussian_filter(image * outside, sigma=sigma, mode="nearest")
    denominator = ndimage.gaussian_filter(outside, sigma=sigma, mode="nearest")
    fallback = ndimage.gaussian_filter(image, sigma=sigma, mode="nearest")
    return np.where(denominator > 1e-6, numerator / np.maximum(denominator, 1e-6), fallback)


def remove_lesion(
    image: ArrayLike,
    lesion_mask: ArrayLike,
    normal_reference: Optional[ArrayLike] = None,
    method: str = "normal_reference",
    gaussian_sigma: float = 2.0,
    blend_sigma: float = 0.0,
) -> ArrayLike:
    """Remove a lesion with reference replacement or local Gaussian inpainting.

    Parameters
    ----------
    method:
        ``"normal_reference"`` requires a spatially matched normal reference.
        ``"local_gaussian"`` estimates replacement intensities only from the
        surrounding non-lesion region.  Neither method inserts a constant block.
    """
    source = _as_numpy(image, "image")
    mask = normalize_soft_mask(lesion_mask, source.shape)
    method = str(method).lower()

    if blend_sigma < 0:
        raise ValueError("blend_sigma must be non-negative")
    if blend_sigma > 0:
        mask = np.clip(ndimage.gaussian_filter(mask, sigma=blend_sigma, mode="nearest"), 0.0, 1.0)

    if method in ("normal_reference", "reference", "matched_reference"):
        if normal_reference is None:
            raise ValueError("normal_reference is required for normal-reference replacement")
        replacement = _as_numpy(normal_reference, "normal_reference")
        if replacement.shape != source.shape:
            raise ValueError("normal_reference shape must match image shape")
    elif method in ("local_gaussian", "gaussian", "gaussian_inpaint"):
        replacement = _local_gaussian_reference(source, mask, float(gaussian_sigma))
    else:
        raise ValueError("unknown removal method %r" % method)

    result = source * (1.0 - mask) + replacement * mask
    return _restore_type(result, image)


def _nuisance_edit_mask(
    lesion_mask: np.ndarray,
    nuisance_mask: Optional[ArrayLike],
    dilation_radius: int,
    protection_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= float(protection_threshold) < 1.0:
        raise ValueError("protection_threshold must lie in [0, 1)")
    soft_protected = dilate_soft_mask(lesion_mask, radius=dilation_radius)
    # Protection is intentionally hard even when the edit mask is soft: a 0.5
    # lesion probability must not permit half-strength nuisance corruption in
    # the lesion itself.  The threshold remains configurable for dense
    # probabilistic masks with a small non-zero floor.
    protected = (soft_protected > float(protection_threshold)).astype(np.float64)
    if nuisance_mask is None:
        requested = np.ones_like(lesion_mask)
    else:
        requested = normalize_soft_mask(nuisance_mask, lesion_mask.shape)
    editable = np.clip(requested * (1.0 - protected), 0.0, 1.0)
    return editable, protected


def apply_nuisance(
    image: ArrayLike,
    lesion_mask: ArrayLike,
    nuisance_mask: Optional[ArrayLike] = None,
    mode: str = "gaussian_noise",
    strength: float = 0.15,
    dilation_radius: int = 3,
    gaussian_sigma: float = 1.5,
    style_reference: Optional[ArrayLike] = None,
    seed: int = 0,
    protection_threshold: float = 0.05,
) -> ArrayLike:
    """Change background nuisance while protecting a dilated lesion ROI.

    Supported modes are additive ``gaussian_noise``, local ``gaussian_blur``
    and first/second-moment ``style`` transfer.  ``nuisance_mask`` can restrict
    the perturbation to a scanner border, organ exterior, or other predeclared
    nuisance region.
    """
    source = _as_numpy(image, "image")
    lesion = normalize_soft_mask(lesion_mask, source.shape)
    editable, _ = _nuisance_edit_mask(
        lesion, nuisance_mask, dilation_radius, protection_threshold
    )
    mode = str(mode).lower()
    strength = float(strength)
    if strength < 0:
        raise ValueError("strength must be non-negative")

    robust_scale = float(np.percentile(source, 95.0) - np.percentile(source, 5.0))
    if robust_scale <= 1e-8:
        robust_scale = float(np.std(source))
    if robust_scale <= 1e-8:
        robust_scale = 1.0

    if mode in ("gaussian_noise", "noise", "local_noise"):
        rng = np.random.RandomState(int(seed))
        candidate = source + rng.normal(0.0, strength * robust_scale, size=source.shape)
    elif mode in ("gaussian_blur", "local_gaussian", "blur"):
        if gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be positive")
        blurred = ndimage.gaussian_filter(source, sigma=float(gaussian_sigma), mode="nearest")
        candidate = source + min(strength, 1.0) * (blurred - source)
    elif mode in ("style", "style_shift", "moment_match"):
        support = editable > 1e-6
        if not np.any(support):
            return _restore_type(source.copy(), image)
        source_values = source[support]
        source_mean = float(np.mean(source_values))
        source_std = float(np.std(source_values))
        if style_reference is not None:
            reference = _as_numpy(style_reference, "style_reference")
            if reference.shape != source.shape:
                raise ValueError("style_reference shape must match image shape")
            reference_values = reference[support]
            target_mean = float(np.mean(reference_values))
            target_std = float(np.std(reference_values))
        else:
            target_mean = source_mean + 0.25 * robust_scale
            target_std = source_std * 1.15
        standardized = (source - source_mean) / max(source_std, 1e-6)
        styled = standardized * max(target_std, 1e-6) + target_mean
        candidate = source + min(strength, 1.0) * (styled - source)
    else:
        raise ValueError("unknown nuisance mode %r" % mode)

    result = source * (1.0 - editable) + candidate * editable
    return _restore_type(result, image)


def _translate_without_wrap(array: np.ndarray, shift: Sequence[int]) -> np.ndarray:
    result = np.zeros_like(array)
    source_slices = []
    destination_slices = []
    for size, offset in zip(array.shape, shift):
        offset = int(offset)
        if offset >= 0:
            source_slices.append(slice(0, size - offset))
            destination_slices.append(slice(offset, size))
        else:
            source_slices.append(slice(-offset, size))
            destination_slices.append(slice(0, size + offset))
    result[tuple(destination_slices)] = array[tuple(source_slices)]
    return result


def make_placebo_mask(
    lesion_mask: ArrayLike,
    protected_mask: Optional[ArrayLike] = None,
    dilation_radius: int = 3,
    max_overlap: float = 0.05,
    attempts: int = 256,
    seed: int = 0,
    allowed_mask: Optional[ArrayLike] = None,
    context_image: Optional[ArrayLike] = None,
    max_outside: float = 0.05,
) -> np.ndarray:
    """Translate the lesion mask to a matched non-lesion tissue location.

    ``allowed_mask`` can constrain the placebo to the same organ or tissue
    compartment.  When ``context_image`` is supplied, surrounding-ring mean and
    variance are used to prefer locally similar tissue instead of image corners.
    """
    lesion = _as_numpy(lesion_mask, "lesion_mask")
    lesion = np.clip(lesion, 0.0, 1.0)
    support = lesion > 1e-6
    if not np.any(support):
        raise ValueError("lesion_mask is empty; a matched placebo cannot be constructed")
    if not 0.0 <= max_overlap <= 1.0:
        raise ValueError("max_overlap must lie in [0, 1]")
    if not 0.0 <= max_outside <= 1.0:
        raise ValueError("max_outside must lie in [0, 1]")

    if protected_mask is None:
        protected = dilate_soft_mask(lesion, radius=dilation_radius)
    else:
        protected = normalize_soft_mask(protected_mask, lesion.shape)
    allowed = (
        np.ones_like(lesion)
        if allowed_mask is None
        else normalize_soft_mask(allowed_mask, lesion.shape)
    )
    context = None
    context_scale = 1.0
    target_context = (0.0, 0.0)
    if context_image is not None:
        context = _as_numpy(context_image, "context_image")
        if context.shape != lesion.shape:
            raise ValueError("context_image shape must match lesion_mask shape")
        context_scale = float(np.percentile(context, 95.0) - np.percentile(context, 5.0))
        if context_scale <= 1e-8:
            context_scale = max(float(np.std(context)), 1.0)

        def context_signature(region: np.ndarray) -> Tuple[float, float]:
            region_support = region > 1e-6
            structure = ndimage.generate_binary_structure(region.ndim, 1)
            ring = ndimage.binary_dilation(
                region_support, structure=structure, iterations=1
            ) & ~region_support
            values = context[ring] if np.any(ring) else context[region_support]
            if values.size == 0:
                return 0.0, 0.0
            return float(np.mean(values)), float(np.std(values))

        target_context = context_signature(lesion)

    coordinates = np.where(support)
    lower = np.asarray([int(np.min(axis)) for axis in coordinates], dtype=int)
    upper = np.asarray([int(np.max(axis)) for axis in coordinates], dtype=int)
    extent = upper - lower + 1
    maximum_origin = np.asarray(lesion.shape, dtype=int) - extent
    if np.any(maximum_origin < 0):  # Defensive; cannot happen for a valid support.
        raise ValueError("lesion support does not fit inside mask shape")

    candidates = set()
    edge_origins = [[0, int(limit)] if limit > 0 else [0] for limit in maximum_origin]
    for origin in itertools.product(*edge_origins):
        candidates.add(tuple(int(v) for v in origin))
    rng = np.random.RandomState(int(seed))
    for _ in range(max(0, int(attempts))):
        origin = tuple(int(rng.randint(0, limit + 1)) for limit in maximum_origin)
        candidates.add(origin)

    lesion_center = 0.5 * (lower + upper)
    lesion_mass = max(float(np.sum(lesion)), 1e-12)
    best = None
    best_key = None
    best_overlap = math.inf
    best_outside = math.inf
    for origin in candidates:
        shift = np.asarray(origin, dtype=int) - lower
        candidate = _translate_without_wrap(lesion, shift)
        overlap = float(np.sum(np.minimum(candidate, protected))) / lesion_mass
        outside = float(np.sum(candidate * (1.0 - allowed))) / lesion_mass
        candidate_center = lesion_center + shift
        separation = float(np.linalg.norm(candidate_center - lesion_center))
        if context is not None:
            candidate_context = context_signature(candidate)
            context_difference = (
                abs(candidate_context[0] - target_context[0])
                + abs(candidate_context[1] - target_context[1])
            ) / context_scale
        else:
            context_difference = 0.0
        constraint_violation = max(0.0, overlap - float(max_overlap)) + max(
            0.0, outside - float(max_outside)
        )
        # Among feasible candidates prefer no overlap, same compartment,
        # context similarity, and then a nearby deterministic location.
        key = (
            constraint_violation,
            overlap,
            outside,
            context_difference,
            separation,
            tuple(int(value) for value in origin),
        )
        if best_key is None or key < best_key:
            best = candidate
            best_key = key
            best_overlap = overlap
            best_outside = outside

    if (
        best is None
        or best_overlap > float(max_overlap) + 1e-12
        or best_outside > float(max_outside) + 1e-12
    ):
        raise ValueError(
            "could not place a matched placebo mask inside the allowed region; "
            "adjust constraints or provide an explicit placebo_mask"
        )
    return best


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(np.sum(weights))
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(values * weights) / denominator)


def _gradient_magnitude(array: np.ndarray) -> np.ndarray:
    components = []
    for axis, size in enumerate(array.shape):
        if size > 1:
            components.append(np.gradient(array, axis=axis))
    if not components:
        return np.zeros_like(array)
    squared = np.zeros_like(array, dtype=np.float64)
    for component in components:
        squared += np.asarray(component, dtype=np.float64) ** 2
    return np.sqrt(squared)


def intervention_quality_metrics(
    original: ArrayLike,
    intervened: ArrayLike,
    edit_mask: ArrayLike,
    protected_mask: Optional[ArrayLike] = None,
) -> Dict[str, float]:
    """Measure edit locality and newly introduced boundary artifacts.

    ``artifact_score`` is the positive increase in boundary gradient, normalized
    by the original image's robust intensity range.  Lower is better.
    ``continuity_score`` maps it monotonically to ``(0, 1]``.
    """
    before = _as_numpy(original, "original")
    after = _as_numpy(intervened, "intervened")
    if before.shape != after.shape:
        raise ValueError("original and intervened shapes must match")
    edit = normalize_soft_mask(edit_mask, before.shape)
    protected = (
        np.zeros_like(edit)
        if protected_mask is None
        else normalize_soft_mask(protected_mask, before.shape)
    )

    absolute_delta = np.abs(after - before)
    scale = float(np.percentile(before, 95.0) - np.percentile(before, 5.0))
    if scale <= 1e-8:
        scale = max(float(np.std(before)), 1.0)

    binary = edit > 0.05
    if np.any(binary):
        structure = ndimage.generate_binary_structure(before.ndim, 1)
        eroded = ndimage.binary_erosion(binary, structure=structure, border_value=0)
        dilated = ndimage.binary_dilation(binary, structure=structure, border_value=0)
        boundary = np.logical_xor(binary, eroded) | np.logical_xor(dilated, binary)
    else:
        boundary = np.zeros_like(binary)

    before_gradient = _gradient_magnitude(before)
    after_gradient = _gradient_magnitude(after)
    introduced = np.maximum(after_gradient - before_gradient, 0.0)
    boundary_weight = boundary.astype(np.float64)
    artifact_score = _weighted_mean(introduced, boundary_weight) / scale
    gradient_change = _weighted_mean(
        np.abs(after_gradient - before_gradient), boundary_weight
    ) / scale

    outside = 1.0 - edit
    changed_threshold = max(scale * 1e-5, 1e-8)
    return {
        "inside_mae": _weighted_mean(absolute_delta, edit),
        "outside_mae": _weighted_mean(absolute_delta, outside),
        "protected_mae": _weighted_mean(absolute_delta, protected),
        "boundary_gradient_change": float(gradient_change),
        "artifact_score": float(artifact_score),
        "continuity_score": float(math.exp(-artifact_score)),
        "changed_fraction": float(np.mean(absolute_delta > changed_threshold)),
        "max_abs_change": float(np.max(absolute_delta)) if absolute_delta.size else 0.0,
    }


def generate_intervention_variants(
    image: ArrayLike,
    lesion_mask: ArrayLike,
    normal_reference: Optional[ArrayLike] = None,
    nuisance_mask: Optional[ArrayLike] = None,
    placebo_mask: Optional[ArrayLike] = None,
    removal_method: Optional[str] = None,
    nuisance_mode: str = "gaussian_noise",
    nuisance_strength: float = 0.15,
    dilation_radius: int = 3,
    gaussian_sigma: float = 2.0,
    blend_sigma: float = 0.0,
    style_reference: Optional[ArrayLike] = None,
    seed: int = 0,
    protection_threshold: float = 0.05,
    placebo_mass_tolerance: float = 0.05,
    placebo_allowed_mask: Optional[ArrayLike] = None,
    placebo_max_outside: float = 0.05,
) -> Dict[str, Any]:
    """Create original, lesion-removed, nuisance-changed, and placebo variants."""
    source = _as_numpy(image, "image")
    lesion = normalize_soft_mask(lesion_mask, source.shape)
    lesion_mass = float(np.sum(lesion))
    if lesion_mass <= 1e-12:
        raise ValueError("lesion_mask is empty")
    if not 0.0 <= float(placebo_mass_tolerance) <= 1.0:
        raise ValueError("placebo_mass_tolerance must lie in [0, 1]")
    if not 0.0 <= float(placebo_max_outside) <= 1.0:
        raise ValueError("placebo_max_outside must lie in [0, 1]")
    if removal_method is None:
        removal_method = "normal_reference" if normal_reference is not None else "local_gaussian"

    lesion_removed = remove_lesion(
        image,
        lesion,
        normal_reference=normal_reference,
        method=removal_method,
        gaussian_sigma=gaussian_sigma,
        blend_sigma=blend_sigma,
    )
    nuisance_changed = apply_nuisance(
        image,
        lesion,
        nuisance_mask=nuisance_mask,
        mode=nuisance_mode,
        strength=nuisance_strength,
        dilation_radius=dilation_radius,
        protection_threshold=protection_threshold,
        gaussian_sigma=gaussian_sigma,
        style_reference=style_reference,
        seed=seed,
    )
    editable, protected = _nuisance_edit_mask(
        lesion, nuisance_mask, dilation_radius, protection_threshold
    )

    if placebo_mask is None:
        placebo = make_placebo_mask(
            lesion,
            protected_mask=protected,
            dilation_radius=dilation_radius,
            seed=seed,
            allowed_mask=placebo_allowed_mask,
            context_image=source,
            max_outside=placebo_max_outside,
        )
    else:
        placebo = normalize_soft_mask(placebo_mask, source.shape)
        placebo_mass = float(np.sum(placebo))
        if placebo_mass <= 1e-12:
            raise ValueError("placebo_mask is empty")
        relative_mass_error = abs(placebo_mass - lesion_mass) / lesion_mass
        if relative_mass_error > float(placebo_mass_tolerance):
            raise ValueError(
                "placebo_mask mass differs from lesion_mask by more than the allowed tolerance"
            )
        if placebo_allowed_mask is not None:
            allowed = normalize_soft_mask(placebo_allowed_mask, source.shape)
            outside = float(np.sum(placebo * (1.0 - allowed))) / placebo_mass
            if outside > float(placebo_max_outside):
                raise ValueError("placebo_mask lies outside placebo_allowed_mask")
        overlap = float(np.sum(np.minimum(placebo, protected))) / max(float(np.sum(placebo)), 1e-12)
        if overlap > 0.05:
            raise ValueError("placebo_mask overlaps the protected lesion ROI by more than 5%")

    placebo_image = remove_lesion(
        image,
        placebo,
        normal_reference=normal_reference,
        method=removal_method,
        gaussian_sigma=gaussian_sigma,
        blend_sigma=blend_sigma,
    )

    original_output = _restore_type(source.copy(), image)
    quality = {
        "original": intervention_quality_metrics(source, source, np.zeros_like(lesion)),
        "lesion_removed": intervention_quality_metrics(source, lesion_removed, lesion),
        "nuisance_changed": intervention_quality_metrics(
            source, nuisance_changed, editable, protected_mask=protected
        ),
        "placebo": intervention_quality_metrics(
            source, placebo_image, placebo, protected_mask=protected
        ),
    }
    return {
        "original": original_output,
        "lesion_removed": lesion_removed,
        "nuisance_changed": nuisance_changed,
        "placebo": placebo_image,
        "masks": {
            "lesion": lesion,
            "protected_lesion": protected,
            "nuisance_edit": editable,
            "placebo": placebo,
        },
        "quality": quality,
        "metadata": {
            "removal_method": str(removal_method),
            "nuisance_mode": str(nuisance_mode),
            "dilation_radius": int(dilation_radius),
            "protection_threshold": float(protection_threshold),
            "placebo_mass_tolerance": float(placebo_mass_tolerance),
            "placebo_max_outside": float(placebo_max_outside),
            "seed": int(seed),
        },
    }


__all__ = [
    "apply_nuisance",
    "dilate_soft_mask",
    "generate_intervention_variants",
    "intervention_quality_metrics",
    "make_placebo_mask",
    "normalize_soft_mask",
    "remove_lesion",
]
