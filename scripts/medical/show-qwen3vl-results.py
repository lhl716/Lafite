#!/usr/bin/env python3
"""Print the three diagnostic gates from their JSON artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _read(path: Path):
    if not path.exists():
        print("[missing] %s" % path)
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _number(value):
    return "NA" if value is None else "%.4f" % float(value)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/medical-real").expanduser().resolve()
    print("Results root: %s" % root)

    for modality in ("brain", "liver"):
        exp1 = _read(root / modality / "exp1" / "probe-summary.json")
        if exp1:
            print(
                "\nExperiment 1 (%s) — positive-patient Dice / pixel AUPRC"
                % modality
            )
            for name, value in exp1.get("feature_sets", {}).items():
                metrics = value.get("metrics", {}).get("test", {})
                print(
                    "  %-28s %s / %s"
                    % (
                        name,
                        _number(metrics.get("positive_patient_dice_mean")),
                        _number(metrics.get("pixel_average_precision")),
                    )
                )

        exp2 = _read(root / modality / "exp2" / "evaluation-summary.json")
        if exp2:
            red = exp2.get("red_flag_assessment", {})
            derived = red.get("derived_metrics", {})
            print(
                "\nExperiment 2 (%s) — shortcut gate: %s"
                % (modality, red.get("status", "unknown"))
            )
            print(
                "  lesion score drop:       %s"
                % _number(derived.get("mean_lesion_score_drop"))
            )
            print(
                "  nuisance/lesion ratio:   %s"
                % _number(derived.get("nuisance_to_lesion_ratio"))
            )
            print(
                "  normal FPR increase (pp): %s"
                % _number(
                    derived.get("normal_nuisance_fpr_increase_percentage_points")
                )
            )
            triggered = [
                item.get("rule")
                for item in red.get("decisions", [])
                if item.get("triggered") is True
            ]
            print("  triggered red flags:     %s" % (", ".join(triggered) or "none"))

    exp3 = _read(root / "exp3" / "evaluation-summary.json")
    if exp3:
        print("\nExperiment 3 — unseen organ×finding edges")
        for name, value in exp3.get("baselines", {}).items():
            intact = value.get("metrics", {}).get("unseen", {}).get(
                "evidence=1|knowledge=1", {}
            )
            interaction = value.get("interaction", {}).get("unseen", {})
            print(
                "  %-20s MRR=%s  NLL=%s  interaction(MRR)=%s"
                % (
                    name,
                    _number(intact.get("mrr")),
                    _number(intact.get("nll")),
                    _number(interaction.get("mrr")),
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
