"""
Temporary MP4 Lifecycle and Safe Retention Manager for ShadowOCR.
Strictly isolated to shadow_lab/data/temp_mp4/.
Zero deletion or modification of production ARDS recordings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Dict, List, Optional, Tuple

_log = logging.getLogger("shadow_ocr.temp_mp4_lifecycle")


@dataclass
class TempMp4JobRecord:
    job_id: str
    passage_id: str
    site_id: str
    original_recording_path: str
    temp_mp4_path: str
    file_size_bytes: int
    created_at_epoch: float
    status: str  # "STAGED" | "PROCESSING" | "COMPLETED" | "FAILED" | "DELETED"
    completed_at_epoch: Optional[float] = None
    delete_after_epoch: Optional[float] = None
    deleted_at_epoch: Optional[float] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class TempMp4LifecycleManager:
    """
    Manages safe copying, retention scheduling, and deletion of temporary passage MP4s.
    """

    def __init__(
        self,
        temp_dir: Path,
        manifest_path: Optional[Path] = None,
        retention_completed_sec: float = 3600.0,  # 60 minutes
        retention_failed_sec: float = 86400.0,    # 24 hours
    ):
        self.temp_dir = Path(temp_dir).resolve()
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = (manifest_path or (self.temp_dir / "temp_mp4_manifest.json")).resolve()
        self.retention_completed_sec = retention_completed_sec
        self.retention_failed_sec = retention_failed_sec
        self._jobs: Dict[str, TempMp4JobRecord] = self._load_manifest()

    def _load_manifest(self) -> Dict[str, TempMp4JobRecord]:
        if not self.manifest_path.is_file():
            return {}
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: TempMp4JobRecord(**v) for k, v in data.items()}
        except Exception as e:
            _log.warning("Could not read temp MP4 manifest, initializing fresh: %s", e)
            return {}

    def _save_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in self._jobs.items()}, f, indent=2)
        tmp.replace(self.manifest_path)

    def stage_temporary_copy(
        self,
        original_mp4_path: Path,
        passage_id: str,
        site_id: str,
    ) -> TempMp4JobRecord:
        """
        Creates an isolated temporary copy in shadow_lab/data/temp_mp4/.
        Rejects path escapes and leaves original file untouched.
        """
        orig_p = Path(original_mp4_path).resolve()
        if not orig_p.is_file():
            raise FileNotFoundError(f"Original recording does not exist: {orig_p}")

        # Unique collision-safe temp filename inside shadow temp dir
        now = time.time()
        unique_suffix = hashlib.sha256(f"{now}_{os.getpid()}_{Path(orig_p).name}".encode()).hexdigest()[:8]
        job_id = f"{site_id}_{passage_id}_{int(now)}_{unique_suffix}"
        temp_target = self.temp_dir / f"temp_{job_id}.mp4"

        # Safe copy
        shutil.copy2(orig_p, temp_target)
        size = temp_target.stat().st_size

        # Verify copy SHA-256 matches source SHA-256
        h_orig = hashlib.sha256()
        with open(orig_p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h_orig.update(chunk)
        h_copy = hashlib.sha256()
        with open(temp_target, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h_copy.update(chunk)

        if h_orig.hexdigest() != h_copy.hexdigest():
            temp_target.unlink(missing_ok=True)
            raise ValueError(f"Staged copy SHA-256 mismatch for {orig_p.name}")

        job = TempMp4JobRecord(
            job_id=job_id,
            passage_id=passage_id,
            site_id=site_id,
            original_recording_path=str(orig_p),
            temp_mp4_path=str(temp_target),
            file_size_bytes=size,
            created_at_epoch=now,
            status="STAGED",
        )
        self._jobs[job_id] = job
        self._save_manifest()
        return job

    def mark_completed(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(f"Job not found: {job_id}")

        now = time.time()
        job.status = "COMPLETED"
        job.completed_at_epoch = now
        job.delete_after_epoch = now + self.retention_completed_sec
        self._save_manifest()

    def mark_failed(self, job_id: str, error_message: str) -> None:
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(f"Job not found: {job_id}")

        now = time.time()
        job.status = "FAILED"
        job.completed_at_epoch = now
        job.delete_after_epoch = now + self.retention_failed_sec
        job.error_message = error_message
        self._save_manifest()

    def run_cleanup_due(self, now_epoch: Optional[float] = None) -> List[str]:
        """
        Deletes only eligible expired temporary MP4s strictly located in temp_dir.
        Guarantees original recordings are never touched.
        """
        now = now_epoch if now_epoch is not None else time.time()
        deleted_jobs: List[str] = []

        for job_id, job in list(self._jobs.items()):
            # Only COMPLETED or FAILED jobs can be cleaned up
            if job.status not in ("COMPLETED", "FAILED"):
                continue

            if job.delete_after_epoch is None or now < job.delete_after_epoch:
                continue

            target_path = Path(job.temp_mp4_path).resolve()

            # Safety check: target must be inside temp_dir and never match original
            if self.temp_dir not in target_path.parents and target_path.parent != self.temp_dir:
                _log.error("Safety violation: attempted cleanup outside temp_dir: %s", target_path)
                continue

            if str(target_path) == job.original_recording_path:
                _log.critical("Safety violation: target matches original path! Aborting delete.")
                continue

            if target_path.is_file():
                try:
                    target_path.unlink()
                    job.status = "DELETED"
                    job.deleted_at_epoch = now
                    deleted_jobs.append(job_id)
                except Exception as e:
                    _log.error("Failed to delete temporary MP4 %s: %s", target_path, e)
            else:
                job.status = "DELETED"
                job.deleted_at_epoch = now
                deleted_jobs.append(job_id)

        if deleted_jobs:
            self._save_manifest()

        return deleted_jobs

    def get_statistics(self) -> Dict[str, Any]:
        total_created = len(self._jobs)
        staged = sum(1 for j in self._jobs.values() if j.status in ("STAGED", "PROCESSING"))
        completed = sum(1 for j in self._jobs.values() if j.status == "COMPLETED")
        failed = sum(1 for j in self._jobs.values() if j.status == "FAILED")
        deleted = sum(1 for j in self._jobs.values() if j.status == "DELETED")

        # Storage consumed currently in temp_dir
        consumed_bytes = sum(
            Path(j.temp_mp4_path).stat().st_size
            for j in self._jobs.values()
            if j.status != "DELETED" and Path(j.temp_mp4_path).is_file()
        )

        return {
            "total_jobs": total_created,
            "staged_or_processing": staged,
            "completed_pending_deletion": completed,
            "failed_pending_deletion": failed,
            "deleted": deleted,
            "storage_consumed_mb": round(consumed_bytes / (1024 * 1024), 2),
        }
