"""
Automatic Shadow Worker for Offline Passage Ingestion and Identification.

Autonomously monitors a local read-only inbox for finalized ARDS MP4 recordings,
validates finalization (size stability & readability), stages verified temporary copies,
executes ShadowOCR inference asynchronously, loads schedules/NTES snapshots,
atomically persists decisions & raw evidence, and records job metrics with crash recovery.
Completely decoupled from production ARDS and NDLS live servers.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import time
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2

from shadow_identity.config_loader import SiteRegistry
from shadow_identity.shadow_ocr.cli import atomic_write_json, atomic_write_jsonl, execute_pipeline
from shadow_identity.shadow_ocr.temp_mp4_lifecycle import TempMp4LifecycleManager

_log = logging.getLogger("shadow_ocr.worker")


def compute_file_sha256(path: Path) -> str:
    """Compute full SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class SourceIdentityMismatch(ValueError):
    """A staged or source file no longer matches its immutable discovery identity."""


def verify_staged_source_identity(job: WorkerJob, staged_path: Path) -> None:
    if job.discovered_sha256 is None or job.discovered_size_bytes is None:
        raise SourceIdentityMismatch("DISCOVERY_IDENTITY_MISSING")
    if staged_path.is_symlink() or not staged_path.is_file():
        raise SourceIdentityMismatch("STAGED_SOURCE_UNSAFE")
    actual_size = staged_path.stat().st_size
    if actual_size != job.discovered_size_bytes:
        raise SourceIdentityMismatch("SOURCE_SIZE_CHANGED_SINCE_DISCOVERY")
    if compute_file_sha256(staged_path) != job.discovered_sha256:
        raise SourceIdentityMismatch("SOURCE_CHANGED_SINCE_DISCOVERY")


def validate_worker_runtime_policy(shadow_lab_root: Path, site_id: str) -> None:
    """Require explicit global collection opt-in while keeping site NTES and network off."""
    registry = SiteRegistry(shadow_lab_root=shadow_lab_root)
    defaults = registry.defaults
    if defaults.get("live_shadow") is not True:
        raise RuntimeError("SHADOW_WORKER_DISABLED_LIVE_SHADOW_FALSE")
    if defaults.get("allow_network") is not False or defaults.get("offline_replay", {}).get("enabled") is not False:
        raise RuntimeError("SHADOW_WORKER_NETWORK_OR_REPLAY_POLICY_UNSAFE")

    site = registry.get_site(site_id)
    if site.get("shadow", {}).get("live_shadow") is not False:
        raise RuntimeError("SITE_LIVE_SHADOW_MUST_REMAIN_FALSE")
    if site.get("ntes", {}).get("allow_network") is not False:
        raise RuntimeError("SITE_NTES_NETWORK_MUST_REMAIN_FALSE")
    if site.get("shadow", {}).get("production_write_allowed") is not False:
        raise RuntimeError("SITE_PRODUCTION_WRITES_MUST_REMAIN_FALSE")


@dataclass(frozen=True)
class DiscoveryRecord:
    path: Path
    source_sha256: str
    source_size: int
    discovered_at_epoch: float
    passage_id: str
    site_id: str
    camera_id: str


@dataclass
class WorkerJob:
    job_id: str
    passage_id: str
    site_id: str
    camera_id: str
    original_mp4_path: str
    mp4_sha256: str
    file_size_bytes: int
    status: str  # DISCOVERED, VALIDATED, STAGED, PROCESSING, EVIDENCE_PERSISTED, DECISION_PERSISTED, COMPLETED, RETRY_WAIT, FAILED
    discovered_at_epoch: float
    discovered_sha256: Optional[str] = None
    discovered_size_bytes: Optional[int] = None
    staged_mp4_path: Optional[str] = None
    output_json_path: Optional[str] = None
    raw_jsonl_path: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3
    error_message: Optional[str] = None
    duration_sec: float = 0.0
    lifecycle_job_id: Optional[str] = None
    retry_after_epoch: Optional[float] = None
    resolved_train_number: Optional[str] = None
    resolver_status: Optional[str] = None
    completed_at_epoch: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# B3 Strict State Machine Transitions
LEGAL_TRANSITIONS: Dict[str, Set[str]] = {
    "DISCOVERED": {"VALIDATED", "FAILED"},
    "VALIDATED": {"STAGED", "FAILED"},
    "STAGED": {"PROCESSING", "FAILED"},
    "PROCESSING": {"EVIDENCE_PERSISTED", "RETRY_WAIT", "FAILED", "STAGED", "COMPLETED"},
    "EVIDENCE_PERSISTED": {"DECISION_PERSISTED", "RETRY_WAIT", "FAILED", "STAGED", "COMPLETED"},
    "DECISION_PERSISTED": {"COMPLETED", "FAILED"},
    "RETRY_WAIT": {"STAGED", "FAILED"},
    "COMPLETED": set(),
    "FAILED": set(),
    # Legacy alias support
    "PERSISTED": {"COMPLETED", "FAILED"},
}


def transition_job_status(job: WorkerJob, new_status: str) -> None:
    """Enforces legal state progression per Part 4 B3."""
    allowed = LEGAL_TRANSITIONS.get(job.status, set())
    if new_status not in allowed:
        raise ValueError(f"Illegal transition from {job.status} to {new_status} for job {job.job_id}")
    job.status = new_status


class ShadowWorker:
    """
    Autonomous worker for discovering, staging, and executing ShadowOCR passage jobs.
    Thread-safe and resistant to collisions, concurrent races, and partial writes.
    """

    def __init__(
        self,
        inbox_dir: Path,
        storage_dir: Path,
        output_dir: Path,
        site_id: str = "ndls_pf12_13",
        camera_id: str = "side_a",
        timetable_path: Optional[str] = None,
        ntes_dir: Optional[str] = None,
        ground_truth_dir: Optional[str] = None,
        max_concurrency: int = 2,
        size_stability_interval_sec: float = 1.0,
        cleanup_retention_sec: float = 3600.0,
        screening_fps: float = 5.0,
        max_ocr_budget: int = 25,
        require_finalization_signal: bool = True,
        allow_unmarked_offline_replay: bool = False,
        offline_replay_config: Optional[Dict[str, Any]] = None,
        operator_replay_workspace_root: Optional[Union[str, Path]] = None,
    ):
        self.inbox_dir = Path(inbox_dir).resolve()
        self.storage_dir = Path(storage_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.site_id = site_id
        self.camera_id = camera_id
        self.timetable_path = timetable_path
        self.ntes_dir = Path(ntes_dir).resolve() if ntes_dir else None
        self.ground_truth_dir = Path(ground_truth_dir).resolve() if ground_truth_dir else None
        self.max_concurrency = max_concurrency
        self.size_stability_interval_sec = size_stability_interval_sec
        self.cleanup_retention_sec = cleanup_retention_sec
        self.screening_fps = screening_fps
        self.max_ocr_budget = max_ocr_budget
        self.require_finalization_signal = require_finalization_signal
        self.allow_unmarked_offline_replay = allow_unmarked_offline_replay

        if offline_replay_config is not None:
            self.offline_replay_config = dict(offline_replay_config)
        else:
            self.offline_replay_config = {
                "enabled": allow_unmarked_offline_replay,
                "allowed_roots": [str(self.inbox_dir)] if allow_unmarked_offline_replay else [],
            }

        ws_root = operator_replay_workspace_root or (
            self.offline_replay_config.get("trusted_workspace_root")
            if isinstance(self.offline_replay_config, dict)
            else None
        ) or (
            self.offline_replay_config.get("operator_replay_workspace_root")
            if isinstance(self.offline_replay_config, dict)
            else None
        )

        if self.offline_replay_config.get("enabled") is True:
            if not ws_root or not str(ws_root).strip():
                raise ValueError("offline_replay enabled requires an explicit, non-empty, existing operator_replay_workspace_root")
            ws_path = Path(ws_root)
            if not ws_path.exists():
                raise ValueError(f"operator_replay_workspace_root does not exist: {ws_root}")
            self.operator_replay_workspace_root = ws_path.resolve(strict=True)

            canonical_allowed: List[str] = []
            allowed_roots = self.offline_replay_config.get("allowed_roots", [])
            if not allowed_roots:
                raise ValueError("offline_replay enabled requires at least one allowed_root inside operator_replay_workspace_root")
            for root_str in allowed_roots:
                rp = Path(root_str)
                if not rp.exists():
                    raise ValueError(f"Configured allowed_root does not exist: {root_str}")
                can_r = rp.resolve(strict=True)
                if not can_r.is_relative_to(self.operator_replay_workspace_root):
                    raise ValueError(f"Configured allowed_root {root_str} is outside operator_replay_workspace_root {self.operator_replay_workspace_root}")
                canonical_allowed.append(str(can_r))
            self.offline_replay_config["allowed_roots"] = canonical_allowed
        else:
            self.operator_replay_workspace_root = (
                Path(ws_root).resolve(strict=True)
                if ws_root and str(ws_root).strip() and Path(ws_root).exists()
                else None
            )

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.storage_dir / "worker_jobs.json"
        self.temp_mp4_dir = self.storage_dir / "temp_mp4"
        self.lifecycle_mgr = TempMp4LifecycleManager(
            temp_dir=self.temp_mp4_dir,
            retention_completed_sec=cleanup_retention_sec,
        )
        self._lock = threading.RLock()
        self._discovery_records: Dict[str, DiscoveryRecord] = {}

        self.jobs: Dict[str, WorkerJob] = self._load_state()
        self._recover_interrupted_jobs()

    def _load_state(self) -> Dict[str, WorkerJob]:
        if not self.state_file.is_file():
            return {}
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            from dataclasses import fields
            valid_fields = {f.name for f in fields(WorkerJob)}
            jobs = {}
            for k, v in data.items():
                filtered = {fk: fv for fk, fv in v.items() if fk in valid_fields}
                jobs[k] = WorkerJob(**filtered)
            return jobs
        except Exception as e:
            _log.warning("Could not load worker state, starting empty: %s", e)
            return {}

    def _save_state(self) -> None:
        with self._lock:
            atomic_write_json(self.state_file, {k: v.to_dict() for k, v in self.jobs.items()})

    def _recover_interrupted_jobs(self) -> None:
        """Crash recovery: Reconcile valid persisted artifacts, otherwise clean partial outputs and reset to STAGED."""
        with self._lock:
            for j in self.jobs.values():
                if j.status == "PROCESSING":
                    # Check if valid outputs were already persisted before crash
                    has_decision = False
                    if j.output_json_path and Path(j.output_json_path).is_file():
                        try:
                            with open(j.output_json_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            file_hash = data.get("input_video_sha256") or data.get("passage_metadata", {}).get("video_sha256")
                            if data.get("resolver_decision") and (not file_hash or file_hash == j.mp4_sha256):
                                has_decision = True
                        except Exception:
                            has_decision = False

                    has_raw = bool(j.raw_jsonl_path and Path(j.raw_jsonl_path).is_file())

                    if has_decision and has_raw:
                        _log.info("Interrupted job %s has valid persisted output, marking COMPLETED", j.job_id)
                        j.status = "COMPLETED"
                        continue

                    if has_raw and not has_decision:
                        # B3: If raw evidence exists but decision is missing, recover conservatively without destroying durable evidence
                        _log.warning("Interrupted job %s has raw evidence but missing decision -> recovering to EVIDENCE_PERSISTED", j.job_id)
                        j.status = "EVIDENCE_PERSISTED"
                        if j.output_json_path and Path(j.output_json_path).is_file():
                            Path(j.output_json_path).unlink(missing_ok=True)
                        continue

                    _log.warning("Recovering interrupted job %s -> resetting to STAGED", j.job_id)
                    j.status = "STAGED"
                    # Clean up partial output files if left behind
                    if j.output_json_path and Path(j.output_json_path).is_file():
                        Path(j.output_json_path).unlink(missing_ok=True)
                    if j.raw_jsonl_path and Path(j.raw_jsonl_path).is_file():
                        Path(j.raw_jsonl_path).unlink(missing_ok=True)
            self._save_state()

    def is_file_finalized(self, path: Path) -> bool:
        """
        Pure validation gate for intake files (§A2):
        1. Reject symlinks for intake files.
        2. Regular file check.
        3. Authoritative signal check (.done or .json finalization).
        4. Validated offline replay root containment.
        5. File-size stability over configured interval.
        6. OpenCV readable frame probe.
        """
        if path.is_symlink():
            _log.warning("Rejecting symlink intake file: %s", path)
            return False

        if not path.is_file():
            return False

        # 1. Authoritative Signal Check
        done_marker = path.with_suffix(".done")
        sidecar_json = path.with_suffix(".json")
        has_signal = False
        if done_marker.is_file():
            has_signal = True
        elif sidecar_json.is_file():
            try:
                with open(sidecar_json, "r", encoding="utf-8") as f:
                    s_data = json.load(f)
                    if s_data.get("finalized") is True or s_data.get("status") == "FINALIZED":
                        has_signal = True
            except Exception:
                pass

        if self.require_finalization_signal and not has_signal:
            replay_authorized = False
            if self.offline_replay_config.get("enabled") is True and self.operator_replay_workspace_root is not None:
                try:
                    resolved_path = path.resolve(strict=True)
                    if not resolved_path.is_relative_to(self.operator_replay_workspace_root):
                        replay_authorized = False
                    else:
                        allowed_roots = self.offline_replay_config.get("allowed_roots", [])
                        for root_str in allowed_roots:
                            root_path = Path(root_str)
                            if not root_path.exists():
                                continue
                            canonical_root = root_path.resolve(strict=True)
                            if canonical_root.is_relative_to(self.operator_replay_workspace_root) and resolved_path.is_relative_to(canonical_root):
                                replay_authorized = True
                                break
                except Exception as e:
                    _log.warning("Error validating replay root for %s: %s", path, e)
                    replay_authorized = False

            if replay_authorized:
                _log.info("Allowing authorized offline replay for %s", path.name)
                has_signal = True
            else:
                _log.info("File %s lacks required authoritative finalization signal or replay authorization", path.name)
                return False

        s1 = path.stat().st_size
        if s1 == 0:
            return False

        if self.size_stability_interval_sec > 0:
            time.sleep(self.size_stability_interval_sec)
            s2 = path.stat().st_size
            if s1 != s2:
                _log.info("File %s size changing (%d -> %d), writer active", path.name, s1, s2)
                return False

        # Check OpenCV probe
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            return False
        cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return cnt > 0

    def discover_new_recordings(self) -> List[Path]:
        """Scans inbox for new, finalized MP4s not yet queued."""
        if not self.inbox_dir.exists():
            return []

        candidates = sorted(list(self.inbox_dir.glob("*.mp4")))
        discovered: List[Path] = []
        inbox_resolved = self.inbox_dir.resolve()

        def sanitize(s: str) -> str:
            return re.sub(r"[^a-zA-Z0-9_\-]", "_", s)

        with self._lock:
            for p in candidates:
                if p.is_symlink():
                    _log.warning("Rejecting symlink intake file: %s", p)
                    continue

                # Reject symlinks escaping allowlisted inbox path
                try:
                    p_res = p.resolve(strict=True)
                    if not p_res.is_relative_to(inbox_resolved):
                        _log.warning("Rejecting symlink escaping allowlisted inbox: %s", p)
                        continue
                except Exception:
                    continue

                if not p.is_file():
                    continue

                if not self.is_file_finalized(p):
                    continue

                sha = compute_file_sha256(p_res)
                stat_size = p_res.stat().st_size
                passage_id = f"pas_{p.stem}"
                cam_id = self.camera_id
                job_key = f"{sanitize(self.site_id)}_{sanitize(cam_id)}_{sanitize(passage_id)}_{sha}"

                rec = DiscoveryRecord(
                    path=p_res,
                    source_sha256=sha,
                    source_size=stat_size,
                    discovered_at_epoch=time.time(),
                    passage_id=passage_id,
                    site_id=self.site_id,
                    camera_id=cam_id,
                )
                self._discovery_records[str(p_res)] = rec
                self._discovery_records[str(p)] = rec

                if job_key in self.jobs and self.jobs[job_key].status in ("PERSISTED", "COMPLETED"):
                    continue

                discovered.append(p)

        return discovered

    def stage_job(self, mp4_path: Path) -> WorkerJob:
        """
        Idempotently stages an MP4 and prepares processing job using collision-resistant identity.
        Key structure: {site_id}_{camera_id}_{passage_id}_{full_sha256}
        Guarantees source discovered is exactly the file that gets processed.
        """
        def sanitize(s: str) -> str:
            return re.sub(r"[^a-zA-Z0-9_\-]", "_", s)

        # Check sidecar for canonical passage_id / camera_id if present
        sidecar_json = mp4_path.with_suffix(".json")
        passage_id = f"pas_{mp4_path.stem}"
        cam_id = self.camera_id
        if sidecar_json.is_file():
            try:
                with open(sidecar_json, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
                    passage_id = m_data.get("passage_id") or passage_id
                    cam_id = m_data.get("camera_id") or cam_id
            except Exception:
                pass

        with self._lock:
            # 1. Symlink / regular file checks
            if mp4_path.is_symlink():
                job_key = f"{sanitize(self.site_id)}_{sanitize(cam_id)}_{sanitize(passage_id)}_symlink_rejected"
                job = WorkerJob(
                    job_id=job_key,
                    passage_id=passage_id,
                    site_id=self.site_id,
                    camera_id=cam_id,
                    original_mp4_path=str(mp4_path),
                    mp4_sha256="",
                    file_size_bytes=0,
                    status="FAILED",
                    discovered_at_epoch=time.time(),
                    error_message="SYMLINK_REPLACEMENT_REJECTED",
                )
                self.jobs[job_key] = job
                self._save_state()
                return job

            try:
                resolved_orig = mp4_path.resolve(strict=True)
            except Exception as e:
                job_key = f"{sanitize(self.site_id)}_{sanitize(cam_id)}_{sanitize(passage_id)}_unresolvable_rejected"
                job = WorkerJob(
                    job_id=job_key,
                    passage_id=passage_id,
                    site_id=self.site_id,
                    camera_id=cam_id,
                    original_mp4_path=str(mp4_path),
                    mp4_sha256="",
                    file_size_bytes=0,
                    status="FAILED",
                    discovered_at_epoch=time.time(),
                    error_message=f"SOURCE_UNRESOLVABLE: {e}",
                )
                self.jobs[job_key] = job
                self._save_state()
                return job

            # 2. Retrieve or establish discovery record
            disc_rec = self._discovery_records.get(str(resolved_orig)) or self._discovery_records.get(str(mp4_path))
            if disc_rec is None:
                disc_sha = compute_file_sha256(resolved_orig)
                disc_size = resolved_orig.stat().st_size
                disc_rec = DiscoveryRecord(
                    path=resolved_orig,
                    source_sha256=disc_sha,
                    source_size=disc_size,
                    discovered_at_epoch=time.time(),
                    passage_id=passage_id,
                    site_id=self.site_id,
                    camera_id=cam_id,
                )
                self._discovery_records[str(resolved_orig)] = disc_rec

            discovered_sha256 = disc_rec.source_sha256
            discovered_size = disc_rec.source_size

            # Collision-resistant job key incorporating discovery SHA-256
            job_key = f"{sanitize(self.site_id)}_{sanitize(cam_id)}_{sanitize(passage_id)}_{discovered_sha256}"

            if job_key in self.jobs and self.jobs[job_key].status in ("STAGED", "PROCESSING", "PERSISTED", "COMPLETED", "FAILED"):
                return self.jobs[job_key]

            # Revalidate source finalization immediately before staging
            if not self.is_file_finalized(mp4_path):
                job = WorkerJob(
                    job_id=job_key,
                    passage_id=passage_id,
                    site_id=self.site_id,
                    camera_id=cam_id,
                    original_mp4_path=str(resolved_orig),
                    mp4_sha256=discovered_sha256,
                    file_size_bytes=discovered_size,
                    status="FAILED",
                    discovered_at_epoch=disc_rec.discovered_at_epoch,
                    discovered_sha256=discovered_sha256,
                    discovered_size_bytes=discovered_size,
                    error_message="SOURCE_NOT_FINALIZED_AT_STAGING",
                )
                self.jobs[job_key] = job
                self._save_state()
                return job

            now = time.time()

            # Stage temporary copy in shadow-owned storage
            staged_record = self.lifecycle_mgr.stage_temporary_copy(
                original_mp4_path=mp4_path,
                passage_id=passage_id,
                site_id=self.site_id,
            )

            staged_file = Path(staged_record.temp_mp4_path)
            identity = WorkerJob(
                job_id=job_key, passage_id=passage_id, site_id=self.site_id, camera_id=cam_id,
                original_mp4_path=str(resolved_orig), mp4_sha256=discovered_sha256,
                file_size_bytes=discovered_size, status="STAGED",
                discovered_at_epoch=disc_rec.discovered_at_epoch,
                discovered_sha256=discovered_sha256, discovered_size_bytes=discovered_size,
            )
            try:
                verify_staged_source_identity(identity, staged_file)
            except SourceIdentityMismatch as e:
                staged_file.unlink(missing_ok=True)
                job = WorkerJob(
                    job_id=job_key,
                    passage_id=passage_id,
                    site_id=self.site_id,
                    camera_id=cam_id,
                    original_mp4_path=str(resolved_orig),
                    mp4_sha256=discovered_sha256,
                    file_size_bytes=discovered_size,
                    status="FAILED",
                    discovered_at_epoch=disc_rec.discovered_at_epoch,
                    discovered_sha256=discovered_sha256,
                    discovered_size_bytes=discovered_size,
                    error_message=str(e),
                )
                self.jobs[job_key] = job
                self._save_state()
                return job

            job = WorkerJob(
                job_id=job_key,
                passage_id=passage_id,
                site_id=self.site_id,
                camera_id=cam_id,
                original_mp4_path=str(resolved_orig),
                mp4_sha256=discovered_sha256,
                file_size_bytes=discovered_size,
                status="STAGED",
                discovered_at_epoch=disc_rec.discovered_at_epoch,
                discovered_sha256=discovered_sha256,
                discovered_size_bytes=discovered_size,
                staged_mp4_path=staged_record.temp_mp4_path,
                lifecycle_job_id=staged_record.job_id,
                output_json_path=str((self.output_dir / f"{job_key}_decision.json").resolve()),
                raw_jsonl_path=str((self.output_dir / f"{job_key}_raw.jsonl").resolve()),
            )
            self.jobs[job_key] = job
            self._save_state()
            return job

    def claim_job_for_processing(self, job_key: str) -> Optional[WorkerJob]:
        """
        Atomically claims a job for processing inside _lock.
        Transitions STAGED/RETRY_WAIT -> PROCESSING and increments attempts.
        Returns the claimed WorkerJob or None if not claimable.
        """
        with self._lock:
            job = self.jobs.get(job_key)
            if not job:
                return None
            if job.status not in ("STAGED", "RETRY_WAIT"):
                return None
            now = time.time()
            if job.status == "RETRY_WAIT" and job.retry_after_epoch and now < job.retry_after_epoch:
                return None
            if job.attempts >= job.max_attempts:
                job.status = "FAILED"
                self._save_state()
                return None
            job.status = "PROCESSING"
            job.attempts += 1
            self._save_state()
            return job

    def process_job(self, job: WorkerJob) -> bool:
        """Executes the pipeline for a single staged job with retry and backoff."""
        if job.status in ("PERSISTED", "COMPLETED"):
            return True

        if job.status != "PROCESSING":
            claimed = self.claim_job_for_processing(job.job_id)
            if not claimed:
                return False
            job = claimed

        t0 = time.time()
        try:
            # Revalidate staged copy exists; restage if missing
            staged_p = Path(job.staged_mp4_path) if job.staged_mp4_path else None
            if not staged_p or not staged_p.is_file():
                orig_p = Path(job.original_mp4_path)
                if orig_p.is_symlink() or not orig_p.resolve(strict=True).is_relative_to(self.inbox_dir):
                    raise SourceIdentityMismatch("SOURCE_AUTHORIZATION_FAILED")
                verify_staged_source_identity(job, orig_p)
                if not self.is_file_finalized(orig_p):
                    raise SourceIdentityMismatch("SOURCE_NOT_FINALIZED_AT_RESTAGING")
                staged_record = self.lifecycle_mgr.stage_temporary_copy(
                    original_mp4_path=orig_p,
                    passage_id=job.passage_id,
                    site_id=job.site_id,
                )
                job.staged_mp4_path = staged_record.temp_mp4_path
                staged_p = Path(job.staged_mp4_path)
                job.lifecycle_job_id = staged_record.job_id
                self._save_state()

            # Check if source metadata JSON exists alongside MP4
            orig_p = Path(job.original_mp4_path)
            meta_json = orig_p.with_suffix(".json")
            p_start = None
            p_end = None
            platform = None
            boundaries_path = None
            ntes_path = None
            if meta_json.is_file():
                try:
                    with open(meta_json, "r", encoding="utf-8") as mf:
                        m_data = json.load(mf)
                        p_start = m_data.get("passage_start")
                        p_end = m_data.get("passage_end")
                        platform = m_data.get("platform")
                        boundaries_path = m_data.get("coach_boundaries_path")
                        ntes_path = m_data.get("ntes_snapshot_path")
                except Exception:
                    pass

            # If not in sidecar, search ntes_dir for passage-specific snapshot
            if not ntes_path and self.ntes_dir and self.ntes_dir.is_dir():
                cand_paths = [
                    self.ntes_dir / f"{job.passage_id}_ntes.json",
                    self.ntes_dir / f"{orig_p.stem}_ntes.json",
                    self.ntes_dir / f"{orig_p.stem}.json",
                    self.ntes_dir / f"{job.passage_id}.json",
                ]
                for cp in cand_paths:
                    if cp.is_file():
                        ntes_path = str(cp)
                        break

            # Run ShadowOCR pipeline
            verify_staged_source_identity(job, staged_p)
            res = execute_pipeline(
                mp4_path=job.staged_mp4_path or job.original_mp4_path,
                site_id=job.site_id,
                passage_id=job.passage_id,
                platform=platform,
                passage_start=p_start,
                passage_end=p_end,
                camera_id=job.camera_id,
                coach_boundaries_path=boundaries_path,
                timetable_path=self.timetable_path,
                ntes_snapshot_path=ntes_path,
                raw_observations_path=job.raw_jsonl_path,
                output_path=job.output_json_path,
                screening_fps=self.screening_fps,
                max_ocr_budget=self.max_ocr_budget,
            )

            job.duration_sec = round(time.time() - t0, 3)
            r_dec = res.get("resolver_decision", {})
            job.resolved_train_number = r_dec.get("shadow_final_number")
            job.resolver_status = r_dec.get("shadow_status")
            job.status = "PERSISTED"
            self._save_state()

            # Mark completed in lifecycle manager using exact lifecycle_job_id
            if job.lifecycle_job_id:
                try:
                    self.lifecycle_mgr.mark_completed(job.lifecycle_job_id)
                except Exception as l_err:
                    _log.warning("Could not mark lifecycle job %s completed: %s", job.lifecycle_job_id, l_err)

            job.status = "COMPLETED"
            job.completed_at_epoch = time.time()
            self._save_state()
            return True

        except Exception as e:
            with self._lock:
                if job.attempts < job.max_attempts:
                    job.status = "RETRY_WAIT"
                    delay = min(300.0, 5.0 * (2 ** (job.attempts - 1)))
                    job.retry_after_epoch = time.time() + delay
                    _log.warning("Job %s failed attempt %d/%d; retrying after %0.1fs: %s",
                                 job.job_id, job.attempts, job.max_attempts, delay, e)
                else:
                    job.status = "FAILED"
                    _log.error("Job %s permanently failed after %d attempts: %s",
                               job.job_id, job.attempts, e)
                job.error_message = str(e)
                job.duration_sec = round(time.time() - t0, 3)
                self._save_state()
                if job.status == "FAILED" and job.lifecycle_job_id:
                    try:
                        self.lifecycle_mgr.mark_failed(job.lifecycle_job_id, job.error_message)
                    except Exception as l_err:
                        _log.warning("Could not mark lifecycle job %s failed: %s", job.lifecycle_job_id, l_err)
            return False

    def run_cycle(self) -> Dict[str, Any]:
        """Runs a single discovery and processing cycle."""
        discovered = self.discover_new_recordings()
        staged_jobs = [self.stage_job(p) for p in discovered]

        now = time.time()
        with self._lock:
            ready_keys = [
                j.job_id for j in self.jobs.values()
                if (j.status == "STAGED" or (j.status == "RETRY_WAIT" and (j.retry_after_epoch is None or now >= j.retry_after_epoch)))
                and j.attempts < j.max_attempts
            ]

        processed_count = 0
        failed_count = 0

        def _execute_key(k: str) -> bool:
            claimed = self.claim_job_for_processing(k)
            if not claimed:
                return False
            return self.process_job(claimed)

        # Execute bounded concurrency
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            results = list(executor.map(_execute_key, ready_keys))
            processed_count = sum(1 for r in results if r)
            failed_count = sum(1 for r in results if not r)

        # Run temporary file cleanup
        cleaned_up = self.lifecycle_mgr.run_cleanup_due()

        return {
            "discovered": len(discovered),
            "processed": processed_count,
            "failed": failed_count,
            "cleaned_up_mp4s": len(cleaned_up),
            "total_active_jobs": len(self.jobs),
        }

    def evaluate_against_ground_truth(self) -> List[Dict[str, Any]]:
        """
        Compares all completed jobs against ground truth files in ground_truth_dir.
        Supports both individual {passage_id}_gt.json files and tabular ground_truth.csv.
        """
        evaluations: List[Dict[str, Any]] = []
        if not self.ground_truth_dir or not self.ground_truth_dir.is_dir():
            return evaluations

        gt_map: Dict[str, Dict[str, Any]] = {}

        # 1. Check for ground_truth.csv
        gt_csv = self.ground_truth_dir / "ground_truth.csv"
        if gt_csv.is_file():
            import csv
            try:
                with open(gt_csv, "r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        vid = row.get("video_id") or Path(row.get("filename", "")).stem
                        num = row.get("actual_train_number", "").strip()
                        name = row.get("actual_train_name", "").strip()
                        if vid:
                            gt_map[vid] = {"train_number": num, "train_name": name}
                            gt_map[f"pas_{vid}"] = {"train_number": num, "train_name": name}
            except Exception as e:
                _log.warning("Could not read ground_truth.csv: %s", e)

        # 2. Check for individual JSON files
        for p in self.ground_truth_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    pid = data.get("passage_id") or p.stem.replace("_gt", "")
                    gt_num = data.get("train_number") or data.get("ground_truth_number")
                    gt_name = data.get("train_name", "")
                    if pid and gt_num:
                        gt_map[pid] = {"train_number": str(gt_num).strip(), "train_name": gt_name}
            except Exception:
                pass

        for j in self.jobs.values():
            if j.status == "COMPLETED":
                # Check mapping by passage_id, orig stem, or job_id
                orig_stem = Path(j.original_mp4_path).stem
                gt_entry = gt_map.get(j.passage_id) or gt_map.get(orig_stem) or {}
                expected_num = gt_entry.get("train_number")
                expected_name = gt_entry.get("train_name")
                actual_num = j.resolved_train_number

                # Determine match: handles single numbers (e.g. "15708") and pair notation (e.g. "18103/18104")
                exact_active_match = None
                pair_membership_match = None
                service_family_match = False
                direction_label_quality = "UNLABELED"

                if expected_num:
                    valid_expected_nums = [n.strip() for n in expected_num.replace("-", "/").split("/") if n.strip()]
                    if len(valid_expected_nums) == 1:
                        direction_label_quality = "SINGLE_ACTIVE_SERVICE"
                        if actual_num:
                            exact_active_match = (actual_num == valid_expected_nums[0])
                        else:
                            exact_active_match = False
                    elif len(valid_expected_nums) > 1:
                        direction_label_quality = "UNVERIFIED_ACTIVE_DIRECTION"
                        exact_active_match = None

                    if actual_num:
                        pair_membership_match = (actual_num in valid_expected_nums)
                    else:
                        pair_membership_match = False

                    # Service family match: True ONLY through explicit service pair/family membership;
                    # never compare first 4 digits as an ad-hoc shortcut.
                    service_family_match = bool(pair_membership_match)

                abstained = bool(actual_num is None or j.resolver_status in ("UNKNOWN", "AMBIGUOUS", "NO_NUMBERS"))
                incorrect_provisional = bool(j.resolver_status == "PROVISIONAL" and pair_membership_match is False)
                false_confirmed = bool(j.resolver_status == "CONFIRMED" and pair_membership_match is False)

                evaluations.append({
                    "job_id": j.job_id,
                    "passage_id": j.passage_id,
                    "original_file": Path(j.original_mp4_path).name,
                    "resolved_train_number": actual_num,
                    "ground_truth_number": expected_num,
                    "ground_truth_name": expected_name,
                    "exact_active_service_match": exact_active_match,
                    "pair_membership_match": pair_membership_match,
                    "service_family_match": service_family_match,
                    "exact_match": exact_active_match if exact_active_match is not None else pair_membership_match,  # legacy compatibility
                    "direction_label_quality": direction_label_quality,
                    "abstained": abstained,
                    "incorrect_provisional": incorrect_provisional,
                    "false_confirmed": false_confirmed,
                    "gt_provenance": "ground_truth.csv" if gt_csv.is_file() else "per_passage_json",
                    "status": j.resolver_status,
                    "duration_sec": j.duration_sec,
                    "mp4_sha256": j.mp4_sha256,
                })

        return evaluations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ShadowOCR Automatic Shadow Worker — Autonomous passage discovery, staging, and inference."
    )
    parser.add_argument("--watch-dir", required=True, help="Read-only inbox directory containing new MP4 recordings")
    parser.add_argument("--storage-dir", default="shadow_lab/data/worker_storage", help="Shadow worker internal storage directory")
    parser.add_argument("--output-dir", default="shadow_lab/data/worker_output", help="Output directory for durable JSON and JSONL")
    parser.add_argument("--site-id", default="ndls_pf12_13", help="Site identifier")
    parser.add_argument("--camera-id", default="side_a", help="Camera identifier")
    parser.add_argument("--timetable", default=None, help="Timetable schedule CSV")
    parser.add_argument("--ntes-dir", default=None, help="Directory containing archived NTES snapshots")
    parser.add_argument("--ground-truth-dir", default=None, help="Directory containing independent ground truth JSON files")
    parser.add_argument("--concurrency", type=int, default=2, help="Max parallel inference workers")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Poll interval in seconds (default: 5s)")
    parser.add_argument("--once", action="store_true", help="Run single discovery cycle and exit")

    args = parser.parse_args()

    try:
        validate_worker_runtime_policy(Path(__file__).resolve().parents[3], args.site_id)
    except Exception as exc:
        print(f"[BLOCKED] Shadow worker runtime policy: {exc}", file=sys.stderr)
        return 2

    worker = ShadowWorker(
        inbox_dir=Path(args.watch_dir),
        storage_dir=Path(args.storage_dir),
        output_dir=Path(args.output_dir),
        site_id=args.site_id,
        camera_id=args.camera_id,
        timetable_path=args.timetable,
        ntes_dir=args.ntes_dir,
        ground_truth_dir=args.ground_truth_dir,
        max_concurrency=args.concurrency,
    )

    if args.once:
        stats = worker.run_cycle()
        print(f"[WORKER] Single cycle completed: {json.dumps(stats, indent=2)}")
        evals = worker.evaluate_against_ground_truth()
        if evals:
            print(f"[WORKER] Ground truth evaluation: {json.dumps(evals, indent=2)}")
        return 0

    print(f"[WORKER] Starting continuous polling on {args.watch_dir} (interval {args.poll_interval}s)...")
    try:
        while True:
            stats = worker.run_cycle()
            if stats["discovered"] > 0 or stats["processed"] > 0:
                print(f"[WORKER] Cycle update: {json.dumps(stats)}")
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("[WORKER] Stopped by user.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
