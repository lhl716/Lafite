#!/usr/bin/env python3
"""GPU-free smoke test for the checkpoint save/load fix.

Exercises lib.utils.misc_helper.save_checkpoint and the retrying loader in
medical_experiments/lafite_adapter.py against tiny dummy tensors -- no LafitE
model, dataset, or GPU needed. Run this before committing to a full training
job, so a checkpoint I/O regression shows up in seconds instead of after
1000 epochs.

Usage:
    python3 scripts/medical/test-checkpoint-io.py
"""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from easydict import EasyDict

import lib.utils.misc_helper as misc_helper
from medical_experiments.lafite_adapter import _load_checkpoint_with_retry


def _config(save_path):
    return EasyDict({"save_path": str(save_path), "saver": {"always_save": True}})


def _state(epoch, fill):
    return {
        "epoch": epoch,
        "arch": "dummy",
        "best_metric": 0.5,
        "optimizer": {},
        "state_dict": {"layer.weight": torch.full((8, 8), float(fill))},
    }


def test_round_trip(tmpdir):
    print("[1/3] round-trip save/load ...", end=" ", flush=True)
    config = _config(tmpdir)
    state = _state(epoch=1, fill=1.0)
    misc_helper.save_checkpoint(state, True, config)
    for name in ("ckpt.pth.tar", "ckpt_best.pth.tar", "ckpt_1.pth.tar"):
        path = Path(tmpdir) / name
        assert path.is_file(), f"missing {name}"
        loaded = torch.load(str(path), map_location="cpu")
        assert torch.equal(
            loaded["state_dict"]["layer.weight"], state["state_dict"]["layer.weight"]
        )
    print("OK")


def test_failed_write_does_not_corrupt(tmpdir):
    print(
        "[2/3] a save that dies mid-write must not touch the real checkpoint ...",
        end=" ",
        flush=True,
    )
    config = _config(tmpdir)
    baseline = _state(epoch=10, fill=10.0)
    misc_helper.save_checkpoint(baseline, True, config)
    ckpt_path = Path(tmpdir) / "ckpt.pth.tar"
    before = ckpt_path.read_bytes()

    real_torch_save = torch.save

    def dying_save(obj, f, *a, **kw):
        # Reproduces the original bug's trigger: the process dies after the
        # destination is opened/partially written but before torch.save returns.
        with open(f, "wb") as fh:
            fh.write(b"\x00" * 37)
        raise IOError("simulated interruption mid-write")

    torch.save = dying_save
    try:
        crashed = False
        try:
            misc_helper.save_checkpoint(_state(epoch=11, fill=999.0), True, config)
        except IOError:
            crashed = True
        assert crashed, "save_checkpoint swallowed the write error instead of propagating it"
    finally:
        torch.save = real_torch_save

    after = ckpt_path.read_bytes()
    assert after == before, (
        "ckpt.pth.tar was modified by an interrupted save -- atomic replace did not hold"
    )
    torch.load(str(ckpt_path), map_location="cpu")  # still loadable
    print("OK (baseline untouched, error propagated instead of being silently written)")


def test_loader_retries_through_bad_read(tmpdir):
    print("[3/3] loader retries through a transient EOFError ...", end=" ", flush=True)
    target = Path(tmpdir) / "flaky.pth.tar"
    target.write_bytes(b"")  # empty file: the exact failure signature hit in production

    def fix_it_shortly():
        time.sleep(0.4)
        tmp = str(target) + ".tmp"
        torch.save(_state(epoch=1, fill=2.0), tmp)
        os.replace(tmp, str(target))

    threading.Thread(target=fix_it_shortly, daemon=True).start()
    loaded = _load_checkpoint_with_retry(torch, target, attempts=5, delay_seconds=0.3)
    assert "state_dict" in loaded
    print("OK")


def main():
    with tempfile.TemporaryDirectory(prefix="ckpt-io-smoketest-") as tmpdir:
        test_round_trip(tmpdir)
        test_failed_write_does_not_corrupt(tmpdir)
    with tempfile.TemporaryDirectory(prefix="ckpt-io-smoketest-") as tmpdir2:
        test_loader_retries_through_bad_read(tmpdir2)
    print("\nAll checkpoint I/O smoke tests passed -- safe to kick off the real training run.")


if __name__ == "__main__":
    main()
