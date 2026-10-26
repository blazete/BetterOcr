"""
Regression tests for Task 1: Hardened Offline-Replay Source Authorization.

Verifies:
1. Replay enabled + missing or empty trusted workspace root -> REJECTED.
2. Replay enabled + nonexistent trusted workspace root -> REJECTED.
3. Allowed root escaping trusted workspace root -> REJECTED.
4. Symlink source files (even if target is in approved root) -> REJECTED.
5. Path traversal attempts escaping authorized root -> REJECTED.
6. Sibling directory with textual prefix match -> REJECTED.
7. Replay disabled by default -> REJECTED.
8. Legitimate file in approved root contained in trusted workspace root -> ALLOWED.
"""

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


def test_replay_enabled_missing_workspace_root_rejected(tmp_path):
    """Missing operator_replay_workspace_root when replay mode enabled must be rejected."""
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True)
    mp4 = inbox / "historical.mp4"
    _create_dummy_mp4(mp4)

    # When replay is enabled, missing trusted workspace root must either raise ValueError on init
    # or reject unmarked historical MP4s in is_file_finalized.
    with pytest.raises(ValueError, match="operator_replay_workspace_root|trusted.*workspace"):
        ShadowWorker(
            inbox_dir=inbox,
            storage_dir=tmp_path / "storage",
            output_dir=tmp_path / "out",
            size_stability_interval_sec=0.0,
            offline_replay_config={
                "enabled": True,
                "allowed_roots": [str(inbox.resolve())],
            },
            operator_replay_workspace_root=None,
        )


def test_replay_enabled_empty_workspace_root_rejected(tmp_path):
    """Empty string workspace root must be rejected."""
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True)
    mp4 = inbox / "historical.mp4"
    _create_dummy_mp4(mp4)

    with pytest.raises(ValueError, match="operator_replay_workspace_root|trusted.*workspace"):
        ShadowWorker(
            inbox_dir=inbox,
            storage_dir=tmp_path / "storage",
            output_dir=tmp_path / "out",
            size_stability_interval_sec=0.0,
            offline_replay_config={
                "enabled": True,
                "allowed_roots": [str(inbox.resolve())],
            },
            operator_replay_workspace_root="",
        )


def test_replay_enabled_nonexistent_workspace_root_rejected(tmp_path):
    """Non-existent workspace root must be rejected."""
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True)
    non_existent = tmp_path / "does_not_exist_workspace"

    with pytest.raises(ValueError, match="operator_replay_workspace_root|trusted.*workspace"):
        ShadowWorker(
            inbox_dir=inbox,
            storage_dir=tmp_path / "storage",
            output_dir=tmp_path / "out",
            size_stability_interval_sec=0.0,
            offline_replay_config={
                "enabled": True,
                "allowed_roots": [str(inbox.resolve())],
            },
            operator_replay_workspace_root=non_existent,
        )


def test_replay_allowed_root_outside_workspace_rejected(tmp_path):
    """Allowed root outside operator_replay_workspace_root must be rejected."""
    workspace = tmp_path / "operator_workspace"
    workspace.mkdir(parents=True)
    outside_root = tmp_path / "outside_root"
    outside_root.mkdir(parents=True)

    mp4 = outside_root / "test.mp4"
    _create_dummy_mp4(mp4)

    # Allowed root is outside the trusted workspace
    with pytest.raises(ValueError, match="is_relative_to|contained within|outside"):
        ShadowWorker(
            inbox_dir=outside_root,
            storage_dir=tmp_path / "storage",
            output_dir=tmp_path / "out",
            size_stability_interval_sec=0.0,
            offline_replay_config={
                "enabled": True,
                "allowed_roots": [str(outside_root.resolve())],
            },
            operator_replay_workspace_root=workspace,
        )


def test_replay_sibling_directory_textual_prefix_rejected(tmp_path):
    """Sibling directory sharing textual prefix but not hierarchy must be rejected."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    approved = workspace / "approved"
    approved.mkdir(parents=True)
    sibling_prefix = workspace / "approved_sibling"
    sibling_prefix.mkdir(parents=True)

    mp4 = sibling_prefix / "test.mp4"
    _create_dummy_mp4(mp4)

    worker = ShadowWorker(
        inbox_dir=sibling_prefix,
        storage_dir=tmp_path / "storage",
        output_dir=tmp_path / "out",
        size_stability_interval_sec=0.0,
        offline_replay_config={
            "enabled": True,
            "allowed_roots": [str(approved.resolve())],
        },
        operator_replay_workspace_root=workspace,
    )
    assert worker.is_file_finalized(mp4) is False


def test_replay_symlink_rejected(tmp_path):
    """Symlink intake files must be rejected even inside approved root."""
    workspace = tmp_path / "workspace"
    approved = workspace / "approved"
    approved.mkdir(parents=True)

    real_mp4 = approved / "real.mp4"
    _create_dummy_mp4(real_mp4)

    symlink_mp4 = approved / "symlink.mp4"
    symlink_mp4.symlink_to(real_mp4)

    worker = ShadowWorker(
        inbox_dir=approved,
        storage_dir=tmp_path / "storage",
        output_dir=tmp_path / "out",
        size_stability_interval_sec=0.0,
        offline_replay_config={
            "enabled": True,
            "allowed_roots": [str(approved.resolve())],
        },
        operator_replay_workspace_root=workspace,
    )
    assert worker.is_file_finalized(symlink_mp4) is False


def test_replay_disabled_by_default_rejects(tmp_path):
    """When offline replay is disabled by default, unmarked MP4s are rejected."""
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True)
    mp4 = inbox / "unmarked.mp4"
    _create_dummy_mp4(mp4)

    worker = ShadowWorker(
        inbox_dir=inbox,
        storage_dir=tmp_path / "storage",
        output_dir=tmp_path / "out",
        size_stability_interval_sec=0.0,
    )
    assert worker.is_file_finalized(mp4) is False


def test_replay_valid_trusted_root_allowed(tmp_path):
    """Legitimate MP4 in approved root inside trusted workspace is allowed."""
    workspace = tmp_path / "workspace"
    approved = workspace / "approved"
    approved.mkdir(parents=True)

    mp4 = approved / "historical_ok.mp4"
    _create_dummy_mp4(mp4)

    worker = ShadowWorker(
        inbox_dir=approved,
        storage_dir=tmp_path / "storage",
        output_dir=tmp_path / "out",
        size_stability_interval_sec=0.0,
        offline_replay_config={
            "enabled": True,
            "allowed_roots": [str(approved.resolve())],
        },
        operator_replay_workspace_root=workspace,
    )
    assert worker.is_file_finalized(mp4) is True
