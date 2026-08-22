"""Download and prepare the BMAD Brain MRI and Liver CT-PNG subsets.

The public BMAD archive contains six domains.  This module downloads the pinned
archive through a configurable Hugging Face endpoint, selectively extracts only
Brain and Liver, and creates patient-disjoint manifests for the three diagnostic
experiments in this repository.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .manifest import validate_manifest, write_jsonl


BMAD_REPO_ID = "code-dev05/BMAD"
BMAD_REVISION = "22cb97e1cfeeb9f677c6c57387347ce727625610"
BMAD_ARCHIVE_NAME = "bmad.zip"
BMAD_ARCHIVE_BYTES = 21_840_892_398
BMAD_SUBSETS = ("Brain", "Liver")
# An axial slice whose "lesion" fills this much of the frame is a decoding
# artefact, not anatomy.
MAX_LESION_COVERAGE = 0.9


def _deps():
    try:
        import numpy as np
        from PIL import Image
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "BMAD preparation requires numpy, Pillow, and scipy; install requirements-medical.txt"
        ) from exc
    return np, Image, ndimage


def download_bmad(
    data_root: Path,
    *,
    repo_id: str = BMAD_REPO_ID,
    revision: str = BMAD_REVISION,
    endpoint: str = "https://hf-mirror.com",
) -> Path:
    """Download the pinned BMAD archive with Hugging Face cache/resume support."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "huggingface_hub is required; install requirements-medical.txt"
        ) from exc

    data_root = Path(data_root).expanduser().resolve()
    download_dir = data_root / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    cached_archive = download_dir / BMAD_ARCHIVE_NAME
    cache_complete = (
        cached_archive.is_file()
        and (
            repo_id != BMAD_REPO_ID
            or revision != BMAD_REVISION
            or cached_archive.stat().st_size == BMAD_ARCHIVE_BYTES
        )
    )
    free_bytes = shutil.disk_usage(data_root).free
    if not cache_complete and free_bytes < 30 * 1024**3:
        raise RuntimeError(
            "BMAD preparation needs about 30 GiB free for the 20.3 GiB archive plus extraction; "
            f"only {free_bytes / 1024**3:.1f} GiB is available at {data_root}"
        )
    archive = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=BMAD_ARCHIVE_NAME,
            revision=revision,
            endpoint=endpoint,
            local_dir=str(download_dir),
        )
    ).resolve()
    actual_bytes = archive.stat().st_size
    if repo_id == BMAD_REPO_ID and revision == BMAD_REVISION and actual_bytes != BMAD_ARCHIVE_BYTES:
        raise RuntimeError(
            f"downloaded archive has {actual_bytes} bytes; expected {BMAD_ARCHIVE_BYTES}. "
            "Delete the incomplete cached file and retry."
        )
    return archive


def extract_bmad_subsets(archive: Path, raw_root: Path) -> Dict[str, int]:
    """Safely and idempotently extract only Brain/ and Liver/ from BMAD."""

    archive = Path(archive).expanduser().resolve()
    raw_root = Path(raw_root).expanduser().resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    extracted: Counter = Counter()
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        relevant_files = sum(
            1
            for member in members
            if not member.is_dir()
            and Path(member.filename.replace("\\", "/")).parts
            and Path(member.filename.replace("\\", "/")).parts[0] in BMAD_SUBSETS
        )
        print(f"Extracting {relevant_files} Brain/Liver files into {raw_root}", flush=True)
        processed = 0
        for member in members:
            normalized = member.filename.replace("\\", "/")
            parts = Path(normalized).parts
            if not parts or parts[0] not in BMAD_SUBSETS:
                continue
            if normalized.startswith("/") or ".." in parts:
                raise ValueError(f"unsafe path in BMAD archive: {member.filename}")
            destination = (raw_root / Path(*parts)).resolve()
            if raw_root not in destination.parents and destination != raw_root:
                raise ValueError(f"archive member escapes extraction root: {member.filename}")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.stat().st_size == member.file_size:
                extracted[parts[0]] += 1
                processed += 1
                if processed % 2000 == 0 or processed == relevant_files:
                    print(f"Extracted/verified {processed}/{relevant_files} files", flush=True)
                continue
            with bundle.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            extracted[parts[0]] += 1
            processed += 1
            if processed % 2000 == 0 or processed == relevant_files:
                print(f"Extracted/verified {processed}/{relevant_files} files", flush=True)
    missing = [subset for subset in BMAD_SUBSETS if not (raw_root / subset).is_dir()]
    if missing:
        raise ValueError("BMAD archive did not contain expected subsets: " + ", ".join(missing))
    return dict(extracted)


def _patient_id(dataset: str, filename: str, source_split: str) -> str:
    stem = Path(filename).stem
    if dataset == "Brain":
        raw = stem.rsplit("_", 1)[0]
        return f"brain-{raw}"
    match = re.match(r"img(\d+)_", stem, flags=re.IGNORECASE)
    if source_split == "train" and match:
        return f"liver-btcv-{int(match.group(1)):04d}"
    raw = stem.rsplit("_", 1)[0]
    return f"liver-lits-{raw}"


def _discover_subset(raw_root: Path, dataset: str) -> List[Dict[str, Any]]:
    root = raw_root / dataset
    if not root.is_dir():
        raise FileNotFoundError(f"missing extracted BMAD subset: {root}")
    found: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for source_split in ("train", "valid", "test"):
        split_root = root / source_split
        if not split_root.is_dir():
            continue
        for image_path in sorted(split_root.rglob("*.png")):
            relative_parts = [part.lower() for part in image_path.relative_to(split_root).parts]
            if "label" in relative_parts or "anomaly_mask" in relative_parts:
                continue
            if source_split == "train":
                source_class = "good"
            else:
                source_class = next(
                    (part for part in relative_parts if part in {"good", "ungood"}), "good"
                )
                if "img" not in relative_parts:
                    continue
            label = int(source_class != "good")
            mask_path: Optional[Path] = None
            if label:
                parts = list(image_path.parts)
                img_index = next((i for i, part in enumerate(parts) if part.lower() == "img"), None)
                if img_index is not None:
                    parts[img_index] = "label"
                    candidate = Path(*parts)
                    if candidate.exists():
                        mask_path = candidate
                if mask_path is None:
                    raise FileNotFoundError(f"anomaly image has no sibling label mask: {image_path}")
            patient = _patient_id(dataset, image_path.name, source_split)
            identity = (patient, image_path.stem)
            item = {
                "dataset": dataset,
                "patient_id": patient,
                "image_path": image_path.resolve(),
                "raw_mask_path": mask_path.resolve() if mask_path else None,
                "label": label,
                "source_split": source_split,
                "source_class": source_class,
            }
            previous = found.get(identity)
            if previous is None or item["label"] > previous["label"]:
                found[identity] = item
    if not found:
        raise ValueError(f"no PNG images discovered under {root}")
    return list(found.values())


def _stable_order(values: Iterable[str], seed: int, salt: str) -> List[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{seed}:{salt}:{value}".encode("utf-8")).hexdigest(),
    )


def _patient_splits(records: Sequence[Mapping[str, Any]], seed: int) -> Dict[str, str]:
    by_patient: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_patient[str(record["patient_id"])].append(record)
    strata: MutableMapping[Tuple[str, int], List[str]] = defaultdict(list)
    for patient, items in by_patient.items():
        status = max(int(item["label"]) for item in items)
        strata[(str(items[0]["dataset"]), status)].append(patient)

    result: Dict[str, str] = {}
    for (dataset, status), patients in sorted(strata.items()):
        ordered = _stable_order(patients, seed, f"{dataset}:{status}")
        count = len(ordered)
        if count >= 3:
            val_count = max(1, round(count * 0.10))
            test_count = max(1, round(count * 0.20))
            if val_count + test_count >= count:
                val_count, test_count = 1, 1
        else:
            val_count, test_count = (0, 1) if count == 2 else (0, 0)
        train_count = count - val_count - test_count
        for index, patient in enumerate(ordered):
            if index < train_count:
                split = "train"
            elif index < train_count + val_count:
                split = "val"
            else:
                split = "test"
            result[patient] = split
    return result


def _limit_slices(records: Sequence[Dict[str, Any]], maximum: int, seed: int) -> List[Dict[str, Any]]:
    if maximum <= 0:
        return list(records)
    by_patient: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_patient[str(record["patient_id"])].append(record)
    selected: List[Dict[str, Any]] = []
    for patient, items in sorted(by_patient.items()):
        anomaly = _stable_order(
            [str(item["image_path"]) for item in items if item["label"]], seed, patient + ":positive"
        )
        normal = _stable_order(
            [str(item["image_path"]) for item in items if not item["label"]], seed, patient + ":normal"
        )
        order = anomaly[:1] + normal[:1] + anomaly[1:] + normal[1:]
        chosen = set(order[:maximum])
        selected.extend(item for item in items if str(item["image_path"]) in chosen)
    return selected


def _manifest_path(path: Path, manifest_dir: Path) -> str:
    import os

    return os.path.relpath(str(path.resolve()), str(manifest_dir.resolve()))


def _write_binary_image(path: Path, array: Any) -> None:
    np, Image, _ = _deps()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(array).astype(bool) * 255).astype("uint8"), mode="L").save(path)


def _drop_alpha(array: Any) -> Any:
    """Drop a trailing alpha plane, identified by channel count (LA=2, RGBA=4).

    An RGBA/LA PNG stores an opaque alpha channel, so reducing over the last
    axis (max/any) marks every pixel as non-zero -- silently turning a lesion
    mask into a full-frame blob, or a brain outline into the whole slice.
    """
    np, _, _unused = _deps()
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[-1] in (2, 4):
        return array[..., :-1]
    return array


def _decode_lesion_mask(path: Path) -> Any:
    """Read a label PNG as a boolean lesion mask, whatever its channel layout."""
    np, Image, _unused = _deps()
    array = np.asarray(Image.open(path))
    if array.ndim != 3:
        return array > 0
    colour = _drop_alpha(array)
    lesion = colour.any(axis=-1) if colour.ndim == 3 else colour > 0
    if not lesion.any() and array.shape[-1] in (2, 4):
        # Colour planes are empty: the mask lives in the alpha plane instead.
        alpha = array[..., -1]
        if alpha.any() and not alpha.all():
            lesion = alpha > 0
    return lesion


def _derived_masks(image: Any, lesion: Any) -> Tuple[Any, Any]:
    np, _, ndimage = _deps()
    # Same alpha trap as the lesion masks: an opaque alpha plane would make the
    # whole slice "foreground", emptying both derived masks.
    image = _drop_alpha(image)
    gray = image.max(axis=-1) if image.ndim == 3 else image
    foreground = np.isfinite(gray) & (gray > 0)
    if foreground.any():
        labels, number = ndimage.label(foreground)
        if number:
            sizes = ndimage.sum(foreground, labels, range(1, number + 1))
            foreground = labels == (int(np.argmax(sizes)) + 1)
        foreground = ndimage.binary_fill_holes(
            ndimage.binary_closing(foreground, iterations=2)
        )
    else:
        foreground = np.ones(gray.shape, dtype=bool)
    lesion_bool = np.asarray(lesion).astype(bool)
    if lesion_bool.ndim == 3:
        lesion_bool = lesion_bool.any(axis=-1)
    protected = ndimage.binary_dilation(lesion_bool, iterations=5)
    placebo_allowed = foreground & ~protected
    nuisance = ~ndimage.binary_dilation(foreground, iterations=3) & ~protected
    return nuisance, placebo_allowed


def prepare_bmad_manifests(
    raw_root: Path,
    prepared_root: Path,
    *,
    seed: int = 42,
    max_slices_per_patient: int = 0,
    semantic_patients_per_edge: int = 0,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Create diagnostic and semantic manifests from extracted BMAD PNGs."""

    np, Image, _ = _deps()
    raw_root = Path(raw_root).expanduser().resolve()
    prepared_root = Path(prepared_root).expanduser().resolve()
    prepared_root.mkdir(parents=True, exist_ok=True)
    if max_slices_per_patient < 0 or semantic_patients_per_edge < 0:
        raise ValueError("slice/patient limits must be non-negative (0 means unlimited)")
    # Discovery walks every PNG under raw_root, which takes minutes on a network
    # mount.  Announce it, or the command looks hung before the first slice.
    records = []
    for subset in BMAD_SUBSETS:
        print(f"Scanning {raw_root / subset} for slices (slow on network storage)...", flush=True)
        found = _discover_subset(raw_root, subset)
        print(f"  {subset}: {len(found)} slices", flush=True)
        records.extend(found)
    patient_to_split = _patient_splits(records, seed)
    records = _limit_slices(records, max_slices_per_patient, seed)
    print(
        f"Preparing {len(records)} slices from {len(set(patient_to_split))} patients",
        flush=True,
    )
    output: List[Dict[str, Any]] = []
    by_patient_output: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)

    ordered_records = sorted(
        records,
        key=lambda row: (row["dataset"], row["patient_id"], str(row["image_path"])),
    )
    for record_index, item in enumerate(ordered_records, start=1):
        dataset = str(item["dataset"])
        digest = hashlib.sha1(
            f"{dataset}:{item['patient_id']}:{item['image_path'].stem}".encode("utf-8")
        ).hexdigest()[:12]
        sample_id = f"{dataset.lower()}-{item['image_path'].stem}-{digest}"
        raw_image = Image.open(item["image_path"])
        image = _colour_planes(np.asarray(raw_image), raw_image.mode)
        shape = image.shape[:2]
        if item["raw_mask_path"]:
            lesion = _decode_lesion_mask(item["raw_mask_path"])
        else:
            lesion = np.zeros(shape, dtype=bool)
        if lesion.shape != shape:
            raise ValueError(f"image/mask shape mismatch for {item['image_path']}: {shape} vs {lesion.shape}")
        # A lesion covering essentially the whole slice is never real; it means
        # the label PNG was decoded wrong.  Fail here rather than let every
        # pixel-level metric downstream silently become meaningless.
        coverage = float(np.asarray(lesion).mean())
        if coverage > MAX_LESION_COVERAGE:
            raise ValueError(
                f"lesion mask covers {coverage:.1%} of the slice for {item['raw_mask_path']} "
                f"(limit {MAX_LESION_COVERAGE:.0%}); the label PNG is not being decoded correctly"
            )

        mask_path = prepared_root / "masks" / dataset.lower() / f"{sample_id}.png"
        nuisance_path = prepared_root / "nuisance_masks" / dataset.lower() / f"{sample_id}.png"
        placebo_path = prepared_root / "placebo_allowed_masks" / dataset.lower() / f"{sample_id}.png"
        _write_binary_image(mask_path, lesion)
        nuisance, placebo_allowed = _derived_masks(image, lesion)
        _write_binary_image(nuisance_path, nuisance)
        _write_binary_image(placebo_path, placebo_allowed)

        organ = "brain" if dataset == "Brain" else "liver"
        row = {
            "sample_id": sample_id,
            "patient_id": item["patient_id"],
            "image": _manifest_path(item["image_path"], prepared_root),
            "mask": _manifest_path(mask_path, prepared_root),
            "nuisance_mask": _manifest_path(nuisance_path, prepared_root),
            "placebo_allowed_mask": _manifest_path(placebo_path, prepared_root),
            "split": patient_to_split[str(item["patient_id"])],
            "modality": "MRI" if dataset == "Brain" else "CT_PNG",
            "label": int(item["label"]),
            "organ": organ,
            "region": f"axial_{organ}",
            "finding": "tumor" if item["label"] else "normal",
            "source_dataset": f"BMAD-{dataset}",
            "source_split": item["source_split"],
            "source_class": item["source_class"],
        }
        output.append(row)
        by_patient_output[str(item["patient_id"])].append(row)
        if record_index % 1000 == 0 or record_index == len(ordered_records):
            print(f"Prepared {record_index}/{len(ordered_records)} slices", flush=True)

    manifest_path = prepared_root / "medical-manifest.jsonl"
    write_jsonl(manifest_path, output)
    validation = validate_manifest(
        output, manifest_path=manifest_path, check_paths=True, require_mask=True, require_semantics=True
    )
    if not validation["valid"]:
        raise ValueError("prepared BMAD manifest is invalid: " + "; ".join(validation["errors"]))

    semantic: List[Dict[str, Any]] = []
    for patient, items in sorted(by_patient_output.items()):
        positives = [row for row in items if row["label"]]
        representative = dict((positives or items)[0])
        patient_positive = bool(positives)
        representative["finding"] = "tumor" if patient_positive else "normal"
        representative["label"] = int(patient_positive)
        representative["sample_id"] = "semantic-" + representative["sample_id"]
        semantic.append(representative)

    if semantic_patients_per_edge > 0:
        by_edge: MutableMapping[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for row in semantic:
            by_edge[(str(row["organ"]), str(row["finding"]))].append(row)
        limited: List[Dict[str, Any]] = []
        for edge, items in sorted(by_edge.items()):
            ordered_ids = _stable_order(
                [str(row["patient_id"]) for row in items], seed, ":".join(edge)
            )[:semantic_patients_per_edge]
            allowed = set(ordered_ids)
            limited.extend(row for row in items if str(row["patient_id"]) in allowed)
        semantic = sorted(limited, key=lambda row: str(row["sample_id"]))

    semantic_path = prepared_root / "semantic-manifest.jsonl"
    write_jsonl(semantic_path, semantic)
    semantic_validation = validate_manifest(
        semantic, manifest_path=semantic_path, check_paths=True, require_mask=True, require_semantics=True
    )
    if not semantic_validation["valid"]:
        raise ValueError("prepared semantic manifest is invalid: " + "; ".join(semantic_validation["errors"]))

    edge_counts = Counter(f"{row['organ']}×{row['finding']}" for row in semantic)
    summary = {
        "medical_manifest": str(manifest_path),
        "semantic_manifest": str(semantic_path),
        "medical_validation": validation,
        "semantic_validation": semantic_validation,
        "semantic_edge_counts": dict(sorted(edge_counts.items())),
        "max_slices_per_patient": max_slices_per_patient,
        "semantic_patients_per_edge": semantic_patients_per_edge,
        "split_policy": "deterministic 70/10/20 patient-level split, stratified by dataset and patient anomaly status",
        "warnings": [
            "These patient-disjoint diagnostic splits differ from the official BMAD benchmark splits.",
            "Liver images are preprocessed PNGs (modality CT_PNG), not raw Hounsfield-unit CT.",
            "nuisance_mask and placebo_allowed_mask are automatically derived candidates; visually audit them before formal causal claims.",
            "The semantic manifest is a coarse normal/tumor compositional pilot, not a report-level clinical benchmark.",
        ],
        "provenance": dict(provenance or {}),
    }
    summary_path = prepared_root / "preparation-summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    summary["summary"] = str(summary_path)
    return summary


def run_bmad_preparation(
    data_root: Path,
    *,
    accept_license: bool,
    endpoint: str = "https://hf-mirror.com",
    repo_id: str = BMAD_REPO_ID,
    revision: str = BMAD_REVISION,
    seed: int = 42,
    max_slices_per_patient: int = 0,
    semantic_patients_per_edge: int = 0,
    skip_download: bool = False,
    archive: Optional[Path] = None,
) -> Dict[str, Any]:
    if not accept_license:
        raise ValueError(
            "BMAD license acceptance is required. Review https://github.com/DorisBao/BMAD "
            "and pass --accept-license only if the dataset terms fit your use."
        )
    data_root = Path(data_root).expanduser().resolve()
    raw_root = data_root / "raw"
    extracted: Dict[str, int] = {}
    archive_path: Optional[Path] = Path(archive).expanduser().resolve() if archive else None
    if archive_path is None and not skip_download:
        archive_path = download_bmad(data_root, repo_id=repo_id, revision=revision, endpoint=endpoint)
    if archive_path is not None:
        if not archive_path.is_file():
            raise FileNotFoundError(f"BMAD archive does not exist: {archive_path}")
        archive_bytes = archive_path.stat().st_size
        if repo_id == BMAD_REPO_ID and revision == BMAD_REVISION and archive_bytes != BMAD_ARCHIVE_BYTES:
            raise ValueError(
                f"BMAD archive has {archive_bytes} bytes; expected {BMAD_ARCHIVE_BYTES}. "
                "The direct download is incomplete; rerun the wget/curl command to resume it."
            )
        extracted = extract_bmad_subsets(archive_path, raw_root)
    elif not all((raw_root / subset).is_dir() for subset in BMAD_SUBSETS):
        raise FileNotFoundError(
            f"--skip-download requires already extracted {raw_root / 'Brain'} and {raw_root / 'Liver'}"
        )
    return prepare_bmad_manifests(
        raw_root,
        data_root / "prepared",
        seed=seed,
        max_slices_per_patient=max_slices_per_patient,
        semantic_patients_per_edge=semantic_patients_per_edge,
        provenance={
            "license_acknowledged": True,
            "official_bmad": "https://github.com/DorisBao/BMAD",
            "bmad_paper": "https://openaccess.thecvf.com/content/CVPR2024W/VAND/html/Bao_BMAD_Benchmarks_for_Medical_Anomaly_Detection_CVPRW_2024_paper.html",
            "hf_endpoint": endpoint,
            "hf_repo_id": repo_id,
            "hf_revision": revision,
            "archive": str(archive_path) if archive_path else None,
            "archive_bytes": archive_path.stat().st_size if archive_path else None,
            "extracted_files": extracted,
        },
    )
