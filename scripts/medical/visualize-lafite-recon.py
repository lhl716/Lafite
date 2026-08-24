#!/usr/bin/env python3
"""Save per-sample reconstruction and localization figures for anomalous slices.

Three figures per selected slice, plus the raw arrays:

  <id>_localization.png  image | GT mask | anomaly heatmap | overlay | thresholded
  <id>_process.png       anomaly heatmap at each swept infer_time
  <id>_ddim_steps.png    residual after each DDIM sampling step (the actual
                         reconstruction trajectory, recorded in feature space)
  <id>.npz               image, mask, and every heatmap, for re-plotting later

The built-in `vis_process_step` path only works for the full p_sample_loop; the
brain/liver configs set `sampling_timesteps: 6`, which routes to `ddim_sample`,
where that argument is swallowed by **kwargs and ignored.  This script records
the trajectory itself by wrapping `model_predictions`.

Example:
    python3 scripts/medical/visualize-lafite-recon.py \
      --config experiments/BMAD-Liver-LafitE/config.yaml \
      --checkpoint .../liver/checkpoints/ckpt.pth.tar \
      --val-meta .../liver-val.jsonl \
      --output-dir .../liver/figures --num-samples 8
"""
import argparse
import json
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import numpy as np
import torch
import torch.nn.functional as functional
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

EPS = 1e-8


def _normalize01(array, window=None):
    """Scale to [0, 1].

    `window` restricts the min/max to a region.  Organ-versus-background is by
    far the largest contrast in the residual, so a global stretch saturates the
    whole organ and hides the within-organ variation -- which is the only part
    the per-image metrics actually score.
    """
    array = np.asarray(array, dtype=np.float64)
    reference = array if window is None or not np.any(window) else array[window]
    low, high = float(np.nanmin(reference)), float(np.nanmax(reference))
    if high - low < EPS:
        return np.zeros_like(array)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def _overlay(gray, heat, alpha=0.5, window=None):
    """Red-hot anomaly map composited over the grayscale slice."""
    base = np.repeat(_normalize01(gray)[..., None], 3, axis=2)
    colour = plt.get_cmap("jet")(_normalize01(heat, window))[..., :3]
    return np.clip((1 - alpha) * base + alpha * colour, 0, 1)


def _contour(axis, mask, colour, label):
    if mask.any():
        axis.contour(mask.astype(float), levels=[0.5], colors=[colour], linewidths=1.2)
        axis.plot([], [], color=colour, linewidth=1.2, label=label)


class DDIMTrajectory:
    """Record the clean-feature estimate produced at every DDIM step."""

    def __init__(self, diffusion_model):
        self.diffusion = diffusion_model
        self.states = []
        self._original = None

    def __enter__(self):
        self._original = self.diffusion.model_predictions

        def recording(*args, **kwargs):
            output = self._original(*args, **kwargs)
            self.states.append(output[1].detach().float().cpu())
            return output

        self.diffusion.model_predictions = recording
        return self

    def __exit__(self, *exc):
        self.diffusion.model_predictions = self._original
        return False


def residual_map(feature_align, feature_rec, size, sigma):
    """||F - F_rec|| on the feature grid, upsampled to image size and smoothed."""
    delta = (feature_align - feature_rec).pow(2).sum(dim=1, keepdim=True).sqrt()
    upsampled = functional.interpolate(delta, size=size, mode="bilinear", align_corners=False)
    array = upsampled[0, 0].numpy()
    return gaussian_filter(array, sigma=sigma) if sigma > 0 else array


def select_samples(val_meta, num_samples, sample_ids):
    rows = []
    with Path(val_meta).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if sample_ids:
        wanted = set(sample_ids)
        chosen = [r for r in rows if str(r.get("sample_id")) in wanted]
        missing = wanted - {str(r.get("sample_id")) for r in chosen}
        if missing:
            raise SystemExit(f"sample ids not found in {val_meta}: {sorted(missing)}")
        return chosen
    positives = [r for r in rows if int(r.get("label", 0)) == 1]
    if not positives:
        raise SystemExit(f"no label=1 rows in {val_meta}")
    # Spread the picks across the cohort instead of taking one patient's run.
    step = max(1, len(positives) // max(1, num_samples))
    return positives[::step][:num_samples]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--val-meta", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sample-ids", default="", help="comma separated sample_id values to force")
    parser.add_argument("--infer-times", default="200,400,600,800,960")
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=4.0,
        help="Gaussian sigma on the anomaly map; eval_helper uses 4, which is wider "
        "than a small liver lesion -- try 0/1/2 to see the effect",
    )
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--skip-ddim-steps", action="store_true")
    args = parser.parse_args()

    from medical_experiments.lafite_adapter import load_lafite
    from lib.datasets.data_builder import build_dataloader
    from lib.utils.misc_helper import update_config
    import yaml
    from easydict import EasyDict

    sweep = [int(v) for v in args.infer_times.split(",") if v.strip()]
    records = select_samples(args.val_meta, args.num_samples, [s for s in args.sample_ids.split(",") if s])
    print(f"visualizing {len(records)} anomalous slices at infer_times {sweep}", flush=True)

    with args.config.open("r", encoding="utf-8") as handle:
        cfg = update_config(EasyDict(yaml.safe_load(handle)))
    # build_dataloader builds train *and* test; drop train so it does not try to
    # open the config's placeholder training meta_file.
    cfg.dataset.pop("train", None)
    cfg.dataset.test.meta_file = str(args.val_meta)
    cfg.dataset.batch_size = 1
    cfg.dataset.workers = 0

    model, load_info = load_lafite(args.config, args.checkpoint, device=args.device)
    size = tuple(load_info["input_size"])
    mean = torch.as_tensor(load_info["pixel_mean"]).view(1, -1, 1, 1)
    std = torch.as_tensor(load_info["pixel_std"]).view(1, -1, 1, 1)

    _, loader = build_dataloader(cfg.dataset, distributed=False)
    wanted = {str(r["filename"]) for r in records}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for batch in loader:
            filename = batch["filename"][0]
            if filename not in wanted:
                continue
            stem = Path(filename).stem

            image = batch["image"].detach().float().cpu()
            gray = (image * std + mean)[0].mean(0).numpy()
            gt = batch["mask"].detach().float().cpu()[0, 0].numpy() > 0
            organ = gray > 0.05

            heatmaps = {}
            trajectory = []
            for infer_time in sweep:
                model.reconstruction.infer_time = int(infer_time)
                record_steps = (not args.skip_ddim_steps) and infer_time == sweep[-1]
                if record_steps:
                    with DDIMTrajectory(model.reconstruction.diffusion_model) as tracker:
                        outputs = model(batch)
                    align = outputs["feature_align"].detach().float().cpu()
                    trajectory = [
                        residual_map(align, state, size, args.smooth_sigma)
                        for state in tracker.states
                    ]
                else:
                    outputs = model(batch)
                pred = outputs["pred"].detach().float().cpu()[0, 0].numpy()
                heatmaps[infer_time] = (
                    gaussian_filter(pred, sigma=args.smooth_sigma)
                    if args.smooth_sigma > 0
                    else pred
                )

            best = heatmaps[sweep[-1]]
            # Rank inside the organ: a global percentile is dominated by the
            # organ-versus-background step and never reaches the lesion.
            cutoff = float(np.percentile(best[organ], args.threshold_percentile)) if organ.any() else float("inf")
            predicted = organ & (best >= cutoff)

            # --- localization -------------------------------------------------
            figure, axes = plt.subplots(1, 5, figsize=(20, 4.4))
            for axis in axes:
                axis.set_xticks([])
                axis.set_yticks([])
            axes[0].imshow(gray, cmap="gray")
            axes[0].set_title("slice")
            axes[1].imshow(gt, cmap="gray")
            axes[1].set_title(f"ground truth ({gt.mean():.4%} of frame)")
            axes[2].imshow(np.where(organ, _normalize01(best, organ), np.nan), cmap="jet")
            axes[2].set_title(f"anomaly map in organ (t={sweep[-1]}, sigma={args.smooth_sigma:g})")
            axes[3].imshow(_overlay(gray, best, window=organ))
            _contour(axes[3], gt, "white", "ground truth")
            axes[3].set_title("overlay")
            axes[3].legend(loc="lower right", fontsize=7)
            axes[4].imshow(gray, cmap="gray")
            _contour(axes[4], gt, "lime", "ground truth")
            _contour(axes[4], predicted, "red", f"pred (top {100 - args.threshold_percentile:g}%)")
            axes[4].set_title("prediction vs truth")
            axes[4].legend(loc="lower right", fontsize=7)
            figure.suptitle(stem, fontsize=10)
            figure.tight_layout()
            figure.savefig(args.output_dir / f"{stem}_localization.png", dpi=130)
            plt.close(figure)

            # --- infer_time progression ---------------------------------------
            figure, axes = plt.subplots(2, len(sweep), figsize=(3.4 * len(sweep), 7))
            axes = np.atleast_2d(axes)
            for column, infer_time in enumerate(sweep):
                heat = heatmaps[infer_time]
                axes[0, column].imshow(np.where(organ, _normalize01(heat, organ), np.nan), cmap="jet")
                axes[0, column].set_title(f"t={infer_time}")
                axes[1, column].imshow(_overlay(gray, heat, window=organ))
                _contour(axes[1, column], gt, "white", "gt")
                for row in (0, 1):
                    axes[row, column].set_xticks([])
                    axes[row, column].set_yticks([])
            axes[0, 0].set_ylabel("anomaly map (organ)")
            axes[1, 0].set_ylabel("overlay")
            figure.suptitle(f"{stem} -- noise level sweep", fontsize=11)
            figure.tight_layout()
            figure.savefig(args.output_dir / f"{stem}_process.png", dpi=130)
            plt.close(figure)

            # --- DDIM trajectory ----------------------------------------------
            if trajectory:
                figure, axes = plt.subplots(1, len(trajectory), figsize=(3.4 * len(trajectory), 3.8))
                axes = np.atleast_1d(axes)
                for step, (axis, heat) in enumerate(zip(axes, trajectory), start=1):
                    axis.imshow(np.where(organ, _normalize01(heat, organ), np.nan), cmap="jet")
                    axis.set_title(f"DDIM step {step}/{len(trajectory)}")
                    axis.set_xticks([])
                    axis.set_yticks([])
                figure.suptitle(f"{stem} -- residual after each sampling step (t={sweep[-1]})", fontsize=11)
                figure.tight_layout()
                figure.savefig(args.output_dir / f"{stem}_ddim_steps.png", dpi=130)
                plt.close(figure)

            np.savez_compressed(
                args.output_dir / f"{stem}.npz",
                image=gray.astype(np.float32),
                mask=gt,
                ddim_steps=np.asarray(trajectory, dtype=np.float32) if trajectory else np.zeros(0),
                infer_times=np.asarray(sweep),
                **{f"heat_{t}": heatmaps[t].astype(np.float32) for t in sweep},
            )

            saved += 1
            print(f"  [{saved}/{len(records)}] {stem}  lesion={gt.mean():.4%}", flush=True)
            if saved >= len(records):
                break

    print(f"\nwrote {saved} sample(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
