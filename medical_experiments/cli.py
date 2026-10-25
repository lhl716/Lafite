"""Command-line entry point for the three medical diagnostic gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .config import load_config, resolve_known_paths, resolve_path, write_json
from .manifest import assign_patient_splits, read_jsonl, validate_manifest, write_jsonl


def _run_configured(args: argparse.Namespace, runner: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> int:
    config = load_config(args.config)
    config = resolve_known_paths(config)
    result = dict(runner(config))
    output = args.output or config.get("summary")
    if output:
        write_json(resolve_path(output, config), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _validate(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest).expanduser().resolve()
    records = read_jsonl(manifest)
    result = validate_manifest(
        records,
        manifest_path=manifest,
        check_paths=args.check_paths,
        require_mask=args.require_mask,
        require_semantics=args.require_semantics,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2


def _split(args: argparse.Namespace) -> int:
    source = Path(args.manifest).expanduser().resolve()
    records = read_jsonl(source)
    ratios = {"train": args.train, "val": args.val, "test": args.test}
    output_records = assign_patient_splits(records, ratios, seed=args.seed, stratify_field=args.stratify)
    validation = validate_manifest(output_records, manifest_path=source)
    if not validation["valid"]:
        raise ValueError("generated split is invalid: %s" % validation["errors"])
    write_jsonl(args.output, output_records)
    print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _export_lafite(args: argparse.Namespace) -> int:
    from .lafite_adapter import export_lafite_features

    return _run_configured(args, export_lafite_features)


def _export_mllm(args: argparse.Namespace) -> int:
    from .mllm_adapter import export_mllm_features

    return _run_configured(args, export_mllm_features)


def _merge_features(args: argparse.Namespace) -> int:
    from .feature_cache import merge_feature_caches

    return _run_configured(args, merge_feature_caches)


def _detail_probe(args: argparse.Namespace) -> int:
    from .detail_probe import run_detail_probe

    return _run_configured(args, run_detail_probe)


def _shortcut(args: argparse.Namespace) -> int:
    from .shortcut_eval import run_shortcut_evaluation

    return _run_configured(args, run_shortcut_evaluation)


def _make_interventions(args: argparse.Namespace) -> int:
    from .intervention_pipeline import run_intervention_pipeline

    return _run_configured(args, run_intervention_pipeline)


def _llm_necessity(args: argparse.Namespace) -> int:
    from .semantic_baselines import run_llm_necessity

    return _run_configured(args, run_llm_necessity)


def _export_semantic_scores(args: argparse.Namespace) -> int:
    from .semantic_score_export import export_semantic_scores

    return _run_configured(args, export_semantic_scores)


def _make_smoke_data(args: argparse.Namespace) -> int:
    from .synthetic import generate_smoke_data

    return _run_configured(args, generate_smoke_data)


def _prepare_bmad(args: argparse.Namespace) -> int:
    from .bmad_prepare import run_bmad_preparation

    result = run_bmad_preparation(
        Path(args.data_root),
        accept_license=args.accept_license,
        endpoint=args.endpoint,
        repo_id=args.repo_id,
        revision=args.revision,
        seed=args.seed,
        max_slices_per_patient=args.max_slices_per_patient,
        semantic_patients_per_edge=args.semantic_patients_per_edge,
        skip_download=args.skip_download,
        archive=Path(args.archive) if args.archive else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _prepare_lafite_data(args: argparse.Namespace) -> int:
    from .lafite_medical import prepare_lafite_data

    result = prepare_lafite_data(
        Path(args.manifest),
        Path(args.output_dir),
        strict_normal_patients=not args.allow_anomalous_patient_normal_slices,
    )
    if args.output:
        write_json(Path(args.output).expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _calibrate_lafite(args: argparse.Namespace) -> int:
    from .lafite_calibration import calibrate_lafite

    return _run_configured(args, calibrate_lafite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m medical_experiments",
        description="Run the LafitE/MLLM medical diagnostic gates.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-manifest", help="check schema, paths, and patient leakage")
    validate.add_argument("manifest")
    validate.add_argument("--check-paths", action="store_true")
    validate.add_argument("--require-mask", action="store_true")
    validate.add_argument("--require-semantics", action="store_true")
    validate.set_defaults(handler=_validate)

    split = subparsers.add_parser("split-patients", help="create deterministic patient-level splits")
    split.add_argument("manifest")
    split.add_argument("output")
    split.add_argument("--train", type=float, default=0.7)
    split.add_argument("--val", type=float, default=0.1)
    split.add_argument("--test", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--stratify")
    split.set_defaults(handler=_split)

    bmad = subparsers.add_parser(
        "prepare-bmad", help="download and prepare BMAD Brain MRI + Liver CT-PNG"
    )
    bmad.add_argument("--data-root", required=True)
    bmad.add_argument("--endpoint", default="https://huggingface.co")
    bmad.add_argument("--repo-id", default="code-dev05/BMAD")
    bmad.add_argument(
        "--revision", default="22cb97e1cfeeb9f677c6c57387347ce727625610"
    )
    bmad.add_argument("--seed", type=int, default=42)
    bmad.add_argument("--max-slices-per-patient", type=int, default=0)
    bmad.add_argument("--semantic-patients-per-edge", type=int, default=0)
    bmad.add_argument("--archive", help="use an existing bmad.zip")
    bmad.add_argument("--skip-download", action="store_true")
    bmad.add_argument("--accept-license", action="store_true")
    bmad.set_defaults(handler=_prepare_bmad)

    lafite_data = subparsers.add_parser(
        "prepare-lafite-data",
        help="create modality-specific BMAD manifests and LafitE training metadata",
    )
    lafite_data.add_argument("--manifest", required=True)
    lafite_data.add_argument("--output-dir", required=True)
    lafite_data.add_argument("--output", help="optional preparation summary JSON")
    lafite_data.add_argument(
        "--allow-anomalous-patient-normal-slices",
        action="store_true",
        help="allow label=0 slices from patients who have another anomalous slice",
    )
    lafite_data.set_defaults(handler=_prepare_lafite_data)

    for name, help_text, handler in (
        ("export-lafite", "cache LafitE aligned/reconstructed features", _export_lafite),
        ("export-mllm", "cache configured MLLM intermediate representations", _export_mllm),
        ("merge-features", "join LafitE and MLLM caches by sample_id", _merge_features),
        ("detail-probe", "experiment 1: representation detail decodability", _detail_probe),
        ("make-interventions", "generate and optionally score four paired variants", _make_interventions),
        ("shortcut-eval", "experiment 2: paired lesion/background interventions", _shortcut),
        ("export-semantic-scores", "run a real LLM/MLLM under four controlled conditions", _export_semantic_scores),
        ("llm-necessity", "experiment 3: compositional semantic baselines", _llm_necessity),
        ("make-smoke-data", "generate non-medical data for installation QA only", _make_smoke_data),
        ("calibrate-lafite", "select LafitE thresholds on a validation split", _calibrate_lafite),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", required=True)
        command.add_argument("--output", help="override summary JSON path")
        command.set_defaults(handler=handler)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(2, "error: %s\n" % exc)
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

