"""Medical array loading and deterministic CT/MRI preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union


PathLike = Union[str, Path]


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("NumPy is required for medical array I/O") from exc
    return np


def load_array(path: PathLike, *, key: Optional[str] = None):
    """Load NPY/NPZ, NIfTI, common images, or a DICOM series.

    Optional formats fail with an actionable dependency message instead of being
    imported at package import time.
    """

    np = _numpy()
    source = Path(path).expanduser()
    lower_name = source.name.lower()

    if source.is_dir():
        try:
            import pydicom
        except ImportError as exc:
            raise RuntimeError("pydicom is required to load a DICOM directory") from exc
        slices = []
        for file_path in source.iterdir():
            if not file_path.is_file():
                continue
            try:
                dataset = pydicom.dcmread(str(file_path), force=True)
                pixels = dataset.pixel_array.astype("float32")
            except Exception:
                continue
            slope = float(getattr(dataset, "RescaleSlope", 1.0))
            intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
            position = getattr(dataset, "ImagePositionPatient", None)
            order = float(position[2]) if position is not None else float(getattr(dataset, "InstanceNumber", len(slices)))
            slices.append((order, pixels * slope + intercept))
        if not slices:
            raise ValueError("No readable DICOM slices in %s" % source)
        slices.sort(key=lambda item: item[0])
        return np.stack([item[1] for item in slices], axis=0)

    if lower_name.endswith(".npy"):
        return np.load(str(source), allow_pickle=False)
    if lower_name.endswith(".npz"):
        archive = np.load(str(source), allow_pickle=False)
        if key is None:
            if len(archive.files) != 1:
                raise ValueError("NPZ has multiple arrays; specify key from %s" % archive.files)
            key = archive.files[0]
        if key not in archive:
            raise KeyError("Array key %r not found in %s" % (key, source))
        return archive[key]
    if lower_name.endswith(".nii") or lower_name.endswith(".nii.gz"):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise RuntimeError("nibabel is required to load NIfTI files") from exc
        return np.asarray(nib.load(str(source)).get_fdata(dtype=np.float32))
    if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required to load image files") from exc
        return np.asarray(Image.open(str(source)))
    raise ValueError("Unsupported medical array path: %s" % source)


def window_ct(array, center: float, width: float):
    """Clip a HU array to a CT window and map it to [0, 1]."""

    np = _numpy()
    if width <= 0:
        raise ValueError("CT window width must be positive")
    low = float(center) - float(width) / 2.0
    high = float(center) + float(width) / 2.0
    clipped = np.clip(np.asarray(array, dtype="float32"), low, high)
    return (clipped - low) / (high - low)


def multi_window_ct(array, windows: Sequence[Sequence[float]]):
    """Stack deterministic CT windows along a final channel dimension."""

    np = _numpy()
    if not windows:
        raise ValueError("at least one CT window is required")
    channels = [window_ct(array, float(window[0]), float(window[1])) for window in windows]
    return np.stack(channels, axis=-1)


def normalize_mri(array, *, foreground_mask=None, lower: float = 0.5, upper: float = 99.5):
    """Robust foreground z-normalization followed by percentile clipping."""

    np = _numpy()
    image = np.asarray(array, dtype="float32")
    if foreground_mask is None:
        foreground = np.isfinite(image) & (image != 0)
    else:
        foreground = np.asarray(foreground_mask).astype(bool) & np.isfinite(image)
    values = image[foreground]
    if values.size == 0:
        return np.zeros_like(image, dtype="float32")
    low, high = np.percentile(values, [lower, upper])
    clipped = np.clip(image, low, high)
    values = clipped[foreground]
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-8:
        std = 1.0
    output = (clipped - mean) / std
    output[~np.isfinite(output)] = 0
    return output.astype("float32")


def select_slice_stack(volume, index: int, *, depth_axis: int = 0, offsets: Sequence[int] = (-1, 0, 1)):
    """Create a 2.5D HWC stack without crossing volume boundaries."""

    np = _numpy()
    data = np.asarray(volume)
    data = np.moveaxis(data, depth_axis, 0)
    if data.ndim != 3:
        raise ValueError("select_slice_stack expects a 3D scalar volume")
    if not 0 <= index < data.shape[0]:
        raise IndexError("slice index %d outside [0, %d)" % (index, data.shape[0]))
    indices = [min(max(index + int(offset), 0), data.shape[0] - 1) for offset in offsets]
    return np.stack([data[item] for item in indices], axis=-1)


def prepare_model_image(array, modality: str, preprocessing: Mapping[str, Any]):
    """Convert a 2D scalar/HWC medical image into normalized HWC float data."""

    np = _numpy()
    image = np.asarray(array)
    input_scaling = str(preprocessing.get("input_scaling", "modality_default")).lower()
    if input_scaling in {"uint8", "uint8_passthrough", "png_uint8"}:
        if image.dtype.kind not in {"u", "i"}:
            finite = image[np.isfinite(image)]
            if finite.size and (float(finite.min()) < 0.0 or float(finite.max()) > 255.0):
                raise ValueError("uint8_passthrough expects values in [0, 255]")
        image = np.clip(image.astype("float32"), 0.0, 255.0) / 255.0
        if image.ndim == 2:
            image = image[..., None]
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.ndim != 3 or image.shape[-1] < 1:
            raise ValueError("uint8_passthrough expects a 2D or HWC image")
        if image.shape[-1] > 3:
            image = image[..., :3]
        return image.astype("float32")
    if input_scaling not in {"modality_default", "medical"}:
        raise ValueError("input_scaling must be modality_default or uint8_passthrough")
    modality_upper = str(modality).upper()
    from_volume = image.ndim == 3 and preprocessing.get("slice_index") is not None
    if from_volume:
        image = select_slice_stack(
            image,
            int(preprocessing["slice_index"]),
            depth_axis=int(preprocessing.get("depth_axis", 0)),
            offsets=preprocessing.get("slice_offsets", [-1, 0, 1]),
        )
    if modality_upper == "CT":
        windows = preprocessing.get("ct_windows", [[40, 400]])
        if image.ndim == 2:
            image = multi_window_ct(image, windows)
        elif from_volume:
            image = np.stack(
                [window_ct(image[..., channel], *windows[0]) for channel in range(image.shape[-1])],
                axis=-1,
            )
        else:
            image = np.stack([window_ct(image[..., channel], *windows[min(channel, len(windows) - 1)]) for channel in range(image.shape[-1])], axis=-1)
    elif modality_upper in {"MRI", "MR"}:
        if image.ndim == 2:
            image = normalize_mri(image)[..., None]
        else:
            image = np.stack([normalize_mri(image[..., channel]) for channel in range(image.shape[-1])], axis=-1)
    else:
        image = image.astype("float32")
        finite = image[np.isfinite(image)]
        if finite.size:
            low, high = float(finite.min()), float(finite.max())
            if high > low:
                image = (image - low) / (high - low)

    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image.astype("float32")
