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
    parser.add_argument("--gpus", default="0", help="comma-separated GPU ids visible to each process, e.g. 0 or 0,1,2,3")
    parser.add_argument(
        "--launcher",
        choices=("auto", "plain", "torchrun"),
        default="auto",
        help="auto: use the platform's RANK/WORLD_SIZE when it supplies them (multi-worker), "
        "else torchrun for multi-GPU on one node, else a plain single process. "
        "plain: never wrap in torchrun (correct for multi-worker jobs). "
        "torchrun: force single-node torchrun with one process per --gpus entry.",
    )
    parser.add_argument(
        "--expect-world-size",
        type=int,
        default=None,
        help="abort unless the resolved world size equals this. Set it to the number of "
        "workers you submitted: without it, a platform that does not inject RANK/WORLD_SIZE "
        "silently runs every worker as its own rank-0 trainer, and they overwrite each "
        "other's checkpoints in the shared run directory.",
    )
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


RANK_VARS = ("RANK", "SLURM_PROCID", "MV2_COMM_WORLD_RANK", "PMI_RANK")
WORLD_VARS = ("WORLD_SIZE", "SLURM_NTASKS", "MV2_COMM_WORLD_SIZE", "PMI_SIZE")


def _first_env(env, names):
    for name in names:
        if name in env:
            return name, env[name]
    return None, None


def resolve_dist_env(env):
    """Mirror lib/utils/dist_helper.get_rank/get_world_size without importing torch."""
    rank_var, rank = _first_env(env, RANK_VARS)
    world_var, world = _first_env(env, WORLD_VARS)
    return {
        "rank_var": rank_var,
        "rank": int(rank) if rank is not None else None,
        "world_var": world_var,
        "world_size": int(world) if world is not None else None,
        "master_addr": env.get("MASTER_ADDR"),
        "master_port": env.get("MASTER_PORT"),
    }


def preflight(args, env):
    """Print, and sanity-check, the distributed setup before burning GPU hours."""
    info = resolve_dist_env(env)
    print("[dist] --- distributed pre-flight ---")
    print(f"[dist] rank        : {info['rank']} (from {info['rank_var'] or 'NOT SET -> dist_helper falls back to 0'})")
    print(f"[dist] world_size  : {info['world_size']} (from {info['world_var'] or 'NOT SET -> dist_helper falls back to len(CUDA_VISIBLE_DEVICES)'})")
    print(f"[dist] master_addr : {info['master_addr'] or 'NOT SET -> dist_helper falls back to localhost'}")
    print(f"[dist] master_port : {info['master_port'] or 'NOT SET -> taken from config.port'}")

    effective_world = info["world_size"] if info["world_size"] is not None else len(
        [g for g in args.gpus.split(",") if g != ""]
    )
    if args.expect_world_size is not None and effective_world != args.expect_world_size:
        print(
            f"\nerror: expected world size {args.expect_world_size} but resolved {effective_world}.\n"
            "       Every worker would train independently as rank 0 and they would all write to the\n"
            "       same checkpoint path, corrupting it. Make the platform inject RANK/WORLD_SIZE/\n"
            "       MASTER_ADDR into each worker, or submit this as a single-worker job.",
            file=sys.stderr,
        )
        sys.exit(2)
    if effective_world > 1 and not info["master_addr"] and args.launcher == "plain":
        print(
            "\nerror: world size > 1 but MASTER_ADDR is not set; ranks would each rendezvous with\n"
            "       localhost and never form one group.",
            file=sys.stderr,
        )
        sys.exit(2)
    print("[dist] ------------------------------")
    return info


def choose_launcher(args, info):
    if args.launcher != "auto":
        return args.launcher
    # The platform already assigned this process a rank in a multi-worker job;
    # wrapping it in torchrun would overwrite RANK/WORLD_SIZE and split the job
    # into independent single-process trainers.
    if info["world_size"] is not None and info["world_size"] > 1:
        return "plain"
    if len([g for g in args.gpus.split(",") if g != ""]) > 1:
        return "torchrun"
    return "plain"


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

    info = preflight(args, env)
    launcher = choose_launcher(args, info)

    print(f"[train] modality={name} gpus={args.gpus} launcher={launcher} epochs={args.max_epochs} batch={args.batch_size} workers={args.workers}")
    print(f"[train] train metadata: {lafite_data_dir}/{name}-train.jsonl")
    print(f"[train] validation metadata: {lafite_data_dir}/{name}-val.jsonl")
    print(f"[train] validation scratch (local): {args.local_scratch}")

    if launcher == "torchrun":
        prefix = _torchrun_launcher() + ["--standalone", f"--nproc_per_node={nproc}"]
    else:
        prefix = [args.python]

    cmd = prefix + [
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
