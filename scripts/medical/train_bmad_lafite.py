#!/usr/bin/env python3
"""Python entry point for the BMAD/LafitE training + calibration pipeline.

Replaces `bash scripts/medical/train-bmad-lafite.sh` for remote job
submission (`python3 xxx.py --args` is friendlier to most job queues than a
bash script with env-var defaults). Unlike the bash version, the train step
here is launched through `torchrun`, so `--gpus 0,1,2,3` actually starts one
process per GPU with DistributedSampler-sharded data -- the bash script only
ever ran a single process, so multi-GPU `-g` there would hang waiting for
ranks that were never spawned. Calibration stays single-process (it doesn't
use DDP).

Usage:
    python3 scripts/medical/train_bmad_lafite.py --mode all --gpus 0,1,2,3
    python3 scripts/medical/train_bmad_lafite.py --mode brain --gpus 0
    python3 scripts/medical/train_bmad_lafite.py --mode status
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]

MODES = (
    "prepare",
    "brain",
    "liver",
    "calibrate-brain",
    "calibrate-liver",
    "calibrate",
    "all",
    "status",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=MODES, default="all")
    parser.add_argument("--bmad-data-root", default="/data/oss_bucket_0/yunlin/dataset/bmad")
    parser.add_argument("--lafite-run-root", default=None, help="default: <bmad-data-root>/lafite-runs")
    parser.add_argument("--lafite-data-dir", default=None, help="default: <bmad-data-root>/prepared/lafite")
    parser.add_argument("--medical-manifest", default=None, help="default: <bmad-data-root>/prepared/medical-manifest.jsonl")
    parser.add_argument("--efficientnet-b4-weights", default="")
    parser.add_argument("--gpus", default="0", help="comma-separated GPU ids for the train step, e.g. 0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--val-freq", type=int, default=50)
    parser.add_argument("--save-step", type=int, default=50)
    parser.add_argument("--local-scratch", default="/tmp/lafite-eval")
    parser.add_argument("--calibration-device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable, help="interpreter for non-torchrun steps")
    return parser.parse_args()


def _torchrun_launcher():
    if shutil.which("torchrun"):
        return ["torchrun"]
    # torchrun is a console-script shim around this module; fall back to it
    # directly in case the shim isn't on PATH but torch itself is importable.
    return [sys.executable, "-m", "torch.distributed.run"]


def run(cmd, env):
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, cwd=str(REPO_DIR), check=True)


def prepare_data(args, env, lafite_data_dir, medical_manifest):
    print(f"[prepare] Creating Brain/Liver LafitE metadata from {medical_manifest}")
    run(
        [
            args.python, "-m", "medical_experiments", "prepare-lafite-data",
            "--manifest", medical_manifest,
            "--output-dir", lafite_data_dir,
            "--output", os.path.join(lafite_data_dir, "preparation-summary.json"),
        ],
        env,
    )


def ensure_data(args, env, lafite_data_dir, medical_manifest):
    if not (
        os.path.isfile(os.path.join(lafite_data_dir, "brain-train.jsonl"))
        and os.path.isfile(os.path.join(lafite_data_dir, "liver-train.jsonl"))
    ):
        prepare_data(args, env, lafite_data_dir, medical_manifest)
    else:
        print(f"[prepare] Reusing existing LafitE metadata in {lafite_data_dir}")


def train_one(args, env, name, lafite_data_dir, lafite_run_root):
    title = name[:1].upper() + name[1:]
    config = str(REPO_DIR / "experiments" / f"BMAD-{title}-LafitE" / "config.yaml")
    run_dir = os.path.join(lafite_run_root, name)
    gpu_ids = [g for g in args.gpus.split(",") if g != ""]
    nproc = len(gpu_ids)

    print(f"[train] modality={name} gpus={args.gpus} nproc_per_node={nproc} epochs={args.max_epochs} batch={args.batch_size} workers={args.workers}")
    print(f"[train] train metadata: {lafite_data_dir}/{name}-train.jsonl")
    print(f"[train] validation metadata: {lafite_data_dir}/{name}-val.jsonl")
    print(f"[train] validation scratch (local): {args.local_scratch}")

    cmd = _torchrun_launcher() + [
        "--standalone",
        f"--nproc_per_node={nproc}",
        "main/train_val_diff.py",
        "-c", config,
        "-g", args.gpus,
        "--train-meta", os.path.join(lafite_data_dir, f"{name}-train.jsonl"),
        "--val-meta", os.path.join(lafite_data_dir, f"{name}-val.jsonl"),
        "--output-dir", run_dir,
        "--epochs", str(args.max_epochs),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--val-freq", str(args.val_freq),
        "--save-step", str(args.save_step),
        "--eval-temp-dir", args.local_scratch,
    ]
    if args.efficientnet_b4_weights:
        if not os.path.isfile(args.efficientnet_b4_weights):
            print(f"error: EfficientNet-B4 weights not found: {args.efficientnet_b4_weights}", file=sys.stderr)
            sys.exit(2)
        cmd += ["--pretrained-model", args.efficientnet_b4_weights]
    else:
        print("[train] EfficientNet-B4 weights: automatic torch cache lookup/download")
    run(cmd, env)


def calibrate_one(args, env, name, lafite_run_root):
    title = name[:1].upper() + name[1:]
    checkpoint = os.path.join(lafite_run_root, name, "checkpoints", "ckpt_best.pth.tar")
    if not os.path.isfile(checkpoint):
        print(f"error: best checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(2)
    env = dict(env)
    if name == "brain":
        env["BRAIN_LAFITE_CHECKPOINT"] = checkpoint
    else:
        env["LIVER_LAFITE_CHECKPOINT"] = checkpoint
    run(
        [
            args.python, "-m", "medical_experiments", "calibrate-lafite",
            "--config", str(REPO_DIR / "experiments" / f"BMAD-{title}-LafitE" / "calibrate.yaml"),
        ],
        env,
    )


def show_status(lafite_data_dir, lafite_run_root):
    for name in ("brain", "liver"):
        print(f"{name}:")
        print(f"  train metadata: {lafite_data_dir}/{name}-train.jsonl")
        print(f"  latest:         {lafite_run_root}/{name}/checkpoints/ckpt.pth.tar")
        print(f"  best:           {lafite_run_root}/{name}/checkpoints/ckpt_best.pth.tar")
        print(f"  calibration:    {lafite_run_root}/{name}/calibration.json")


def main():
    args = parse_args()

    lafite_run_root = args.lafite_run_root or os.path.join(args.bmad_data_root, "lafite-runs")
    lafite_data_dir = args.lafite_data_dir or os.path.join(args.bmad_data_root, "prepared", "lafite")
    medical_manifest = args.medical_manifest or os.path.join(args.bmad_data_root, "prepared", "medical-manifest.jsonl")

    env = os.environ.copy()
    env["LAFITE_DATA_DIR"] = lafite_data_dir
    env["LAFITE_RUN_ROOT"] = lafite_run_root
    env["LAFITE_CALIBRATION_DEVICE"] = args.calibration_device
    env.setdefault("LAFITE_EVAL_TEMP_DIR", args.local_scratch)

    if args.mode == "status":
        show_status(lafite_data_dir, lafite_run_root)
        return
    if args.mode == "prepare":
        prepare_data(args, env, lafite_data_dir, medical_manifest)
        return

    if args.mode in ("brain", "liver", "all"):
        ensure_data(args, env, lafite_data_dir, medical_manifest)

    if args.mode == "brain":
        train_one(args, env, "brain", lafite_data_dir, lafite_run_root)
    elif args.mode == "liver":
        train_one(args, env, "liver", lafite_data_dir, lafite_run_root)
    elif args.mode == "calibrate-brain":
        ensure_data(args, env, lafite_data_dir, medical_manifest)
        calibrate_one(args, env, "brain", lafite_run_root)
    elif args.mode == "calibrate-liver":
        ensure_data(args, env, lafite_data_dir, medical_manifest)
        calibrate_one(args, env, "liver", lafite_run_root)
    elif args.mode == "calibrate":
        ensure_data(args, env, lafite_data_dir, medical_manifest)
        calibrate_one(args, env, "brain", lafite_run_root)
        calibrate_one(args, env, "liver", lafite_run_root)
    elif args.mode == "all":
        train_one(args, env, "brain", lafite_data_dir, lafite_run_root)
        calibrate_one(args, env, "brain", lafite_run_root)
        train_one(args, env, "liver", lafite_data_dir, lafite_run_root)
        calibrate_one(args, env, "liver", lafite_run_root)


if __name__ == "__main__":
    main()
