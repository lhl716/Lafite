"""Versioned NPZ cache format for LafitE and MLLM representations."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from .manifest import normalize_split


CACHE_VERSION = 1
PathLike = Union[str, Path]
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy is required for feature caching") from exc
    return np


def safe_sample_name(sample_id: str) -> str:
    original = str(sample_id)
    name = _SAFE_NAME.sub("_", original).strip("._")
    if not name:
        raise ValueError("sample_id does not contain a safe filename character")
    # Sanitisation alone is not injective (for example, ``case/a`` and
    # ``case?a`` both become ``case_a``).  Keep a readable prefix and append a
    # digest of the exact ID so overwrite=True cannot replace another case.
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:12]
    return "%s--%s" % (name[:96], digest)


def save_feature_record(
    cache_dir: PathLike,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically save one sample's arrays plus JSON metadata."""

    np = _numpy()
    if not metadata.get("sample_id"):
        raise ValueError("metadata must contain sample_id")
    output_dir = Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (safe_sample_name(str(metadata["sample_id"])) + ".npz")
    if output.exists() and not overwrite:
        raise FileExistsError("feature cache already exists: %s" % output)
    payload = {str(key): np.asarray(value) for key, value in arrays.items()}
    if "metadata_json" in payload:
        raise ValueError("metadata_json is a reserved cache key")
    packed_metadata = dict(metadata)
    if "split" in packed_metadata:
        packed_metadata["split"] = normalize_split(packed_metadata["split"])
    packed_metadata["cache_version"] = CACHE_VERSION
    # Duplicate the core identity fields as scalar arrays.  This keeps the
    # cache directly consumable by NumPy-only experiment runners while the
    # full, extensible record remains available in metadata_json.
    for field in ("sample_id", "patient_id", "split", "modality", "label"):
        if field in packed_metadata and field not in payload:
            payload[field] = np.asarray(packed_metadata[field])
    payload["metadata_json"] = np.asarray(json.dumps(packed_metadata, ensure_ascii=False, sort_keys=True))
    _write_npz(np, output, payload)
    return output


def _write_npz(np, output: Path, payload: Mapping[str, Any]) -> None:
    """Write the archive, staging through local scratch when the target cannot seek.

    ``np.savez_compressed`` builds a ZIP, and ``zipfile.close`` seeks backwards
    to write the central directory.  Object-storage FUSE mounts accept ordinary
    writes but fail that seek with ``OSError: [Errno 95]``, so build the archive
    on a local filesystem and copy the finished bytes across.
    """
    temporary = output.with_suffix(".npz.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
    except OSError as error:
        if error.errno != errno.EOPNOTSUPP:
            raise
        temporary.unlink(missing_ok=True)
        scratch_root = os.environ.get("LAFITE_CACHE_SCRATCH") or tempfile.gettempdir()
        os.makedirs(scratch_root, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=scratch_root, suffix=".npz", delete=False) as staged:
            staged_path = Path(staged.name)
        try:
            np.savez_compressed(str(staged_path), **payload)
            # savez appends .npz when handed a path without one.
            produced = staged_path if staged_path.exists() and staged_path.stat().st_size else Path(str(staged_path) + ".npz")
            shutil.copyfile(produced, temporary)
            if produced != staged_path:
                produced.unlink(missing_ok=True)
        finally:
            staged_path.unlink(missing_ok=True)
    temporary.replace(output)


def load_feature_record(path: PathLike) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    np = _numpy()
    with np.load(str(path), allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError("not a versioned feature cache: %s" % path)
        metadata = json.loads(str(archive["metadata_json"].item()))
        if int(metadata.get("cache_version", -1)) != CACHE_VERSION:
            raise ValueError("unsupported cache version in %s" % path)
        arrays = {key: archive[key] for key in archive.files if key != "metadata_json"}
    return metadata, arrays


def iter_feature_records(cache_dir: PathLike, *, split: Optional[str] = None) -> Iterator[Tuple[Path, Dict[str, Any], Dict[str, Any]]]:
    for path in sorted(Path(cache_dir).glob("*.npz")):
        metadata, arrays = load_feature_record(path)
        if split is None or normalize_split(metadata.get("split")) == normalize_split(split):
            yield path, metadata, arrays


def cache_index(cache_dir: PathLike) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for path, metadata, arrays in iter_feature_records(cache_dir):
        sample_id = str(metadata["sample_id"])
        if sample_id in index:
            raise ValueError("duplicate cached sample_id: %s" % sample_id)
        index[sample_id] = {
            "path": str(path.resolve()),
            "patient_id": metadata.get("patient_id"),
            "split": metadata.get("split"),
            "stages": sorted(
                key
                for key in arrays
                if key not in {"sample_id", "patient_id", "split", "modality", "label"}
            ),
        }
    return index


def merge_feature_caches(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Join model-specific cache directories by ``sample_id``.

    ``sources`` is a list of mappings with ``cache_dir`` and optional
    ``prefix``, ``include``, and ``rename`` values.  Patient and split metadata
    must agree exactly.  By default every source must contain the same samples.
    """

    from .config import resolve_path

    np = _numpy()
    source_specs = config.get("sources")
    if not isinstance(source_specs, Sequence) or len(source_specs) < 2:
        raise ValueError("merge config requires at least two sources")
    prepared: List[Tuple[Mapping[str, Any], Dict[str, Path]]] = []
    sample_sets = []
    for raw_spec in source_specs:
        if not isinstance(raw_spec, Mapping) or "cache_dir" not in raw_spec:
            raise ValueError("each source requires cache_dir")
        directory = resolve_path(raw_spec["cache_dir"], config)
        paths: Dict[str, Path] = {}
        for path, metadata, _arrays in iter_feature_records(directory):
            sample_id = str(metadata["sample_id"])
            if sample_id in paths:
                raise ValueError("duplicate sample_id %s in %s" % (sample_id, directory))
            paths[sample_id] = path
        if not paths:
            raise ValueError("source cache is empty: %s" % directory)
        prepared.append((raw_spec, paths))
        sample_sets.append(set(paths))

    strict = bool(config.get("strict_sample_set", True))
    if strict and any(values != sample_sets[0] for values in sample_sets[1:]):
        counts = [len(values) for values in sample_sets]
        raise ValueError("source sample sets differ (counts=%s)" % counts)
    sample_ids = sorted(set.intersection(*sample_sets))
    output_dir = resolve_path(config["output_dir"], config)
    overwrite = bool(config.get("overwrite", False))
    exported = 0
    for sample_id in sample_ids:
        merged_metadata: Optional[Dict[str, Any]] = None
        merged_arrays: Dict[str, Any] = {}
        for spec, paths in prepared:
            metadata, arrays = load_feature_record(paths[sample_id])
            if merged_metadata is None:
                merged_metadata = dict(metadata)
            else:
                for field in ("patient_id", "split"):
                    left = merged_metadata.get(field)
                    right = metadata.get(field)
                    if field == "split":
                        left, right = normalize_split(left), normalize_split(right)
                    if str(left) != str(right):
                        raise ValueError("metadata mismatch for %s field %s" % (sample_id, field))
            include = spec.get("include")
            identity_keys = {"sample_id", "patient_id", "split", "modality", "label"}
            selected = (
                set(str(value) for value in include)
                if include
                else set(arrays).difference(identity_keys)
            )
            prefix = str(spec.get("prefix", ""))
            rename = {str(key): str(value) for key, value in dict(spec.get("rename", {})).items()}
            for key, value in arrays.items():
                if key not in selected:
                    continue
                target_key = rename.get(key, prefix + key)
                if target_key in merged_arrays:
                    if target_key == "mask" and np.array_equal(merged_arrays[target_key], value):
                        continue
                    raise ValueError("array key collision %r for sample %s" % (target_key, sample_id))
                merged_arrays[target_key] = value
        assert merged_metadata is not None
        merged_metadata["source"] = "merged"
        merged_metadata["merged_sources"] = [str(spec.get("cache_dir")) for spec, _ in prepared]
        save_feature_record(output_dir, merged_metadata, merged_arrays, overwrite=overwrite)
        exported += 1
    return {
        "output_dir": str(output_dir),
        "exported": exported,
        "sources": len(prepared),
        "strict_sample_set": strict,
    }
