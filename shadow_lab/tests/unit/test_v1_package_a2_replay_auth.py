"""
Regression tests for BetterOcr / ShadowOCR V1 Package A2:
Secure Offline-Replay Authorization.

Requirements:
- Default OFF.
- Production source discovery still requires authentic finalization.
- Historical offline replay must be explicitly enabled.
- Allowed root must be canonicalized with Path.resolve(strict=True).
- Input must be a regular finalized MP4, not a special file.
- Check canonical containment with Path.is_relative_to() against approved roots.
- Reject symlinks for intake files.
- Reject files outside approved roots.
- Recheck source metadata and SHA-256 after copying.
- Do not accept arbitrary roots merely because a caller supplies them.
- Load allowed roots from trusted site/deployment config.
- Validate configured root against operator-controlled replay workspace root.

Tests:
1. Legitimate historical MP4 inside approved root -> allowed.
2. Unapproved MP4 -> rejected.
3. Symlink outside root -> rejected.
4. Similar prefix path -> rejected.
5. Path containing literal word 'shadow_lab' outside root -> rejected.
6. Production intake without authentic finalization -> rejected.
7. Replay disabled -> unmarked MP4 rejected.
"""

import os
from pathlib import Path
import pytest
import cv2
import numpy as np

from shadow_identity.shadow_ocr.worker import ShadowWorker


def _create_dummy_mp4(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 64))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(5):
        out.write(frame)
    out.release()


def test_a2_replay_disabled_by_default_rejects_unmarked(tmp_path):
    """7. Replay disabled -> unmarked MP4 rejected."""
    inbox = tmp_path / "inbox"
    storage = tmp_path / "storage"
    out_dir = tmp_path / "out"
    mp4 = inbox / "test.mp4"
    _create_dummy_mp4(mp4)

    # Worker with default settings (offline_replay disabled)
    worker = ShadowWorker(
        inbox_dir=inbox,
        storage_dir=storage,
        output_dir=out_dir,
        size_stability_interval_sec=0.0,
    )
    assert worker.is_file_finalized(mp4) is False


def test_a2_approved_root_allowed(tmp_path):
    """1. Legitimate historical MP4 inside approved root -> allowed."""
    workspace = tmp_path / "workspace"
    approved_root = workspace / "approved_historical"
    approved_root.mkdir(parents=True, exist_ok=True)
    mp4 = approved_root / "historic.mp4"
    _create_dummy_mp4(mp4)

    storage = tmp_path / "storage"
    out_dir = tmp_path / "out"

    worker = ShadowWorker(
        inbox_dir=approved_root,
        storage_dir=storage,
        output_dir=out_dir,
        size_stability_interval_sec=0.0,
        offline_replay_config={
            "enabled": True,
            "allowed_roots": [str(approved_root)],
        },
        operator_replay_workspace_root=workspace,
    )
    assert worker.is_file_finalized(mp4) is True


def test_a2_unapproved_root_rejected(tmp_path):
    """2. Unapproved MP4 -> rejected."""
    workspace = tmp_path / "workspace"
    approved_root = workspace / "approved"
    approved_root.mkdir(parents=True, exist_ok=True)
    unapproved_root = workspace / "unapproved"
    unapproved_root.mkdir(parents=True, exist_ok=True)

    mp4 = unapproved_root / "evil.mp4"
    _create_dummy_mp4(mp4)

    worker = ShadowWorker(
        inbox_dir=unapproved_root,
        storage_dir=tmp_path / "storage",
        output_dir=tmp_path / "out",
        size_stability_interval_sec=0.0,
        offline_replay_config={
            "enabled": True,
            "allowed_roots": [str(approved_root)],
        },
        operator_replay_workspace_root=workspace,
    )
    assert worker.is_file_finalized(mp4) is False


def test_a2_symlink_rejected(tmp_path):
    """3. Symlink outside root or intake symlink -> rejected."""
    workspace = tmp_path / "workspace"
    approved_root = workspace / "approved"
    approved_root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    outside.mkdir(parents=True, exist_ok=True)

    real_mp4 = outside / "real.mp4"
    _create_dummy_mp4(real_mp4)

    symlink_mp4 = approved_root / "symlink.mp4"
    symlink_mp4.symlink_to(real_mp4)

    worker = ShadowWorker(
        inbox_dir=approved_root,
        storage_dir=tmp_path / "storage",
        output_dir=tmp_path / "out",
        size_stability_interval_sec=0.0,
        offline_replay_config={
            "enabled": True,
            "allowed_roots": [str(approved_root)],
        },
        operator_replay_workspace_root=workspace,
    )
    assert worker.is_file_finalized(symlink_mp4) is False


def test_a2_similar_prefix_path_rejected(tmp_path):
    """4. Similar prefix path -> rejected."""
    workspace = tmp_path / "workspace"
    approved_root = workspace / "approved"
    approved_root.mkdir(parents=True, exist_ok=True)
    fake_prefix_root = workspace / "approved_fake"
    fake_prefix_root.mkdir(parents=True, exist_ok=True)

    mp4 = fake_prefix_root / "spoof.mp4"
    _create_dummy_mp4(mp4)

    worker = ShadowWorker(
        inbox_dir=fake_prefix_root,
        storage_dir=tmp_path / "storage",
        output_dir=tmp_path / "out",
        size_stability_interval_sec=0.0,
        offline_replay_config={
            "enabled": True,
            "allowed_roots": [str(approved_root)],
        },
        operator_replay_workspace_root=workspace,
    )
    assert worker.is_file_finalized(mp4) is False


def test_a2_shadow_lab_substring_trick_rejected(tmp_path):
    """5. Path containing literal word 'shadow_lab' outside root -> rejected."""
    workspace = tmp_path / "workspace"
    approved_root = workspace / "approved"
    approved_root.mkdir(parents=True, exist_ok=True)

    trick_dir = tmp_path / "shadow_lab_fake_replay"
    trick_dir.mkdir(parents=True, exist_ok=True)
    mp4 = trick_dir / "trick.mp4"
    _create_dummy_mp4(mp4)

    worker = ShadowWorker(
        inbox_dir=trick_dir,
        storage_dir=tmp_path / "storage",
        output_dir=tmp_path / "out",
        size_stability_interval_sec=0.0,
        offline_replay_config={
            "enabled": True,
            "allowed_roots": [str(approved_root)],
        },
        operator_replay_workspace_root=workspace,
    )
    assert worker.is_file_finalized(mp4) is False


def test_a2_production_intake_without_authentic_finalization_rejected(tmp_path):
    """6. Production intake without authentic finalization -> rejected."""
    prod_inbox = tmp_path / "prod_inbox"
    prod_inbox.mkdir(parents=True, exist_ok=True)
    mp4 = prod_inbox / "prod.mp4"
    _create_dummy_mp4(mp4)

    worker = ShadowWorker(
        inbox_dir=prod_inbox,
        storage_dir=tmp_path / "storage",
        output_dir=tmp_path / "out",
        size_stability_interval_sec=0.0,
        require_finalization_signal=True,
        offline_replay_config={"enabled": False, "allowed_roots": []},
    )
    # No .done or .json finalization signal
    assert worker.is_file_finalized(mp4) is False

    # Now add authentic .done signal
    done_file = prod_inbox / "prod.done"
    done_file.write_text("DONE")
    assert worker.is_file_finalized(mp4) is True
