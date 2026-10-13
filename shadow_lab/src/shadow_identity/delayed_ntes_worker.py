"""
Shadow-Only Delayed NTES Verification Worker (§4)

Lifecycle:
  passage completes
  -> enqueue passage_id
  -> around +180 seconds reuse approved NTES cache/scraper or snapshot archive
  -> save snapshot strictly under shadow_lab/data/ntes_snapshots/ only
  -> rerun Stage T2
  -> update shadow result & comparison bundle
  -> strictly read-only towards ARDS production (ZERO writes outside shadow_lab)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from shadow_identity.candidate_resolver import resolve_shadow
from shadow_identity.comparison import categorize_comparison
from shadow_identity.config_loader import SiteRegistry
from shadow_identity.models import (
    NameCandidate,
    NtesSnapshot,
    NtesTrainEntry,
    NumberCandidate,
    PassageBundle,
    PassageInput,
    PassageVisualConsensus,
    ShadowDecision,
    TimetableRow,
    VisualObservation,
    VisualOcrEvidence,
)
from shadow_identity.reason_codes import (
    FT_BOARD_NOT_NUMBERED,
    FT_MULTIPLE_LIVE_CANDIDATES,
    FT_NTES_STALE,
    FT_NTES_UNAVAILABLE,
    FT_OCR_NUMBER_FAILURE,
    FT_TIMETABLE_AMBIGUITY,
    FT_TIMETABLE_NTES_CONFLICT,
    FT_TIMETABLE_STALE,
    FT_UNKNOWN,
    FT_VISUAL_NTES_CONFLICT,
    FT_VISUAL_TIMETABLE_CONFLICT,
    RC_AMBIGUOUS_MULTI_CANDIDATE,
    RC_NTES_CORROBORATED_STALE,
    RC_NTES_UNAVAILABLE,
    RC_SCHEDULE_OUTSIDE_WINDOW,
    RC_SCHEDULE_UNAVAILABLE,
    RC_TIMETABLE_NTES_CONFLICT,
    RC_VISUAL_BOARD_NOT_NUMBERED,
    RC_VISUAL_NTES_CONFLICT,
    RC_VISUAL_TIMETABLE_CONFLICT,
)
from shadow_identity.timetable_importer import TimetableImporter

logger = logging.getLogger("shadow_identity.delayed_ntes_worker")


def _to_naive_utc(dt: datetime) -> datetime:
    """Normalize datetime to naive UTC for safe cross-timezone comparisons."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt



@dataclass
class DelayedVerificationTask:
    passage_id: str
    site_id: str
    enqueued_at: str
    passage_end_time: str
    scheduled_verify_at: str
    delay_seconds: float = 180.0
    status: str = "PENDING"  # PENDING, PROCESSED, FAILED
    error_message: Optional[str] = None
    processed_at: Optional[str] = None
    t2_train_number: Optional[str] = None
    t2_status: Optional[str] = None


class DelayedNtesVerificationWorker:
    def __init__(
        self,
        shadow_lab_root: Optional[Path] = None,
        default_delay_seconds: float = 180.0,
        snapshot_provider: Optional[Callable[[str, str], Optional[NtesSnapshot]]] = None,
        queue_file: Optional[Path] = None,
    ):
        if shadow_lab_root is None:
            self.root = Path(__file__).resolve().parents[2]
        else:
            self.root = Path(shadow_lab_root)

        self.default_delay_seconds = default_delay_seconds
        self.root = self.root.resolve()

        if snapshot_provider is not None:
            self.snapshot_provider = snapshot_provider
            self.adapter = None
        else:
            from shadow_identity.ntes_adapter import ArdsNtesAdapter
            self.adapter = ArdsNtesAdapter(shadow_lab_root=self.root)
            self.snapshot_provider = self.adapter.get_snapshot

        self.data_dir = self.root / "data"
        self.passages_dir = self.data_dir / "passages"
        self.ntes_dir = self.data_dir / "ntes_snapshots"
        self.queue_file = Path(queue_file) if queue_file else (self.data_dir / "delayed_verification_queue.json")


        self.passages_dir.mkdir(parents=True, exist_ok=True)
        self.ntes_dir.mkdir(parents=True, exist_ok=True)

        self._tasks: Dict[str, DelayedVerificationTask] = {}
        self._load_queue()

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    def _load_queue(self) -> None:
        if not self.queue_file.exists():
            return
        try:
            with open(self.queue_file, "r", encoding="utf-8") as f:
                raw_list = json.load(f)
                for item in raw_list:
                    t = DelayedVerificationTask(**item)
                    self._tasks[t.passage_id] = t
        except Exception as e:
            logger.warning(f"Could not load delayed verification queue: {e}")

    def _save_queue(self) -> None:
        try:
            with open(self.queue_file, "w", encoding="utf-8") as f:
                json.dump([asdict(t) for t in self._tasks.values()], f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save delayed verification queue: {e}")

    def enqueue_passage(
        self,
        passage_id: str,
        site_id: str,
        passage_end_time: Optional[str] = None,
        delay_seconds: Optional[float] = None,
    ) -> DelayedVerificationTask:
        """
        Enqueue a passage for delayed NTES verification.
        Scheduled verify time is around passage_end_time + delay_seconds.
        """
        now = datetime.now()
        delay_s = delay_seconds if delay_seconds is not None else self.default_delay_seconds

        end_str = passage_end_time or now.isoformat()
        try:
            base_dt = datetime.fromisoformat(end_str)
        except Exception:
            base_dt = now

        sched_dt = base_dt + timedelta(seconds=delay_s)

        task = DelayedVerificationTask(
            passage_id=passage_id,
            site_id=site_id,
            enqueued_at=now.isoformat(),
            passage_end_time=end_str,
            scheduled_verify_at=sched_dt.isoformat(),
            delay_seconds=delay_s,
            status="PENDING",
        )

        with self._lock:
            self._tasks[passage_id] = task
            self._save_queue()

        return task

    def get_pending_tasks(self) -> List[DelayedVerificationTask]:
        with self._lock:
            return [t for t in self._tasks.values() if t.status == "PENDING"]

    def fetch_delayed_snapshot(
        self, site_id: str, passage_end_time: str
    ) -> Optional[NtesSnapshot]:
        """
        Obtain or reuse approved NTES snapshot around +180s.
        Never writes outside shadow_lab.
        """
        if self.snapshot_provider is not None:
            snap = self.snapshot_provider(site_id, passage_end_time)
            if snap and not getattr(snap, "from_cache", False):
                self.save_snapshot_locally(site_id, snap)
            return snap

        # Fallback: inspect locally archived snapshots under shadow_lab
        site_ntes_dir = self.ntes_dir / site_id
        if site_ntes_dir.exists():
            snaps = sorted(site_ntes_dir.glob("*.json"))
            if snaps:
                try:
                    with open(snaps[-1], "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return NtesSnapshot(
                        site_id=data.get("site_id", site_id),
                        station_code=data.get("station_code", ""),
                        requested_at=data.get("requested_at", data.get("fetched_at", "")),
                        fetched_at=data.get("fetched_at", ""),
                        available=data.get("available", True),
                        trains=[NtesTrainEntry(**t) for t in data.get("trains", [])],
                        raw_hash=data.get("raw_hash", data.get("raw_payload_checksum", "")),
                        from_cache=True,
                    )
                except Exception:
                    pass

        return None

    def save_snapshot_locally(self, site_id: str, snap: NtesSnapshot) -> Path:
        """
        Save snapshot strictly within shadow_lab/data/ntes_snapshots/<site_id>/.
        Guarantees ZERO mutation to production ARDS.
        """
        target_dir = self.ntes_dir / site_id
        target_dir.mkdir(parents=True, exist_ok=True)
        fname = f"snap_{snap.fetched_at.replace(':', '-').replace('.', '-') if snap.fetched_at else int(time.time())}.json"
        out_path = target_dir / fname

        resolved_str = str(out_path.resolve())
        root_str = str(self.root.resolve())
        if not resolved_str.startswith(root_str) or "ARDS-main" in resolved_str:
            raise PermissionError(f"Security violation: attempt to write outside shadow_lab: {out_path}")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(snap), f, indent=2)
        return out_path

    def process_passage(
        self,
        passage_id: str,
        snapshot_override: Optional[NtesSnapshot] = None,
    ) -> Optional[PassageBundle]:
        """
        Execute delayed T2 verification:
        1. Load passage bundle.
        2. Acquire +180s snapshot.
        3. Save snapshot locally under shadow_lab.
        4. Rerun Stage T2.
        5. Update bundle, comparison, and save strictly in shadow workspace.
        """
        bundle_file = self.passages_dir / f"{passage_id}.json"
        if not bundle_file.exists():
            logger.error(f"Passage bundle {passage_id} not found at {bundle_file}")
            if passage_id in self._tasks:
                self._tasks[passage_id].status = "FAILED"
                self._tasks[passage_id].error_message = "File not found"
                self._save_queue()
            return None

        with open(bundle_file, "r", encoding="utf-8") as f:
            b_dict = json.load(f)

        site_id = b_dict.get("site_id", "")
        passage_end = b_dict.get("passage", {}).get("end_time") or ""

        # Step 2: Acquire snapshot
        snapshot = snapshot_override
        if snapshot is None:
            snapshot = self.fetch_delayed_snapshot(site_id, passage_end)

        if snapshot is not None:
            if not getattr(snapshot, "from_cache", False):
                self.save_snapshot_locally(site_id, snapshot)
            b_dict["ntes"]["post_passage_snapshot"] = asdict(snapshot)

        # Step 3: Rerun Stage T2
        registry = SiteRegistry(shadow_lab_root=self.root)
        site_cfg = registry.get_site_config(site_id)
        site_aliases = registry.get_site_aliases(site_id)

        sched_cfg = site_cfg.get("schedule", {})
        sched_path_str = sched_cfg.get("path") or site_cfg.get("timetable_file")
        sched_path = Path(sched_path_str)
        if not sched_path.is_absolute():
            sched_path = self.root.parent / sched_path
            if not sched_path.exists():
                sched_path = self.root / sched_path_str

        tt_snapshot = TimetableImporter.import_schedule(
            site_id=site_id,
            station_code=site_cfg.get("station", {}).get("code", site_cfg.get("station_code", "")),
            csv_path=str(sched_path),
            version=sched_cfg.get("version", "v1"),
            effective_from=sched_cfg.get("effective_from"),
        )

        p_info = b_dict.get("passage", {})
        p_input = PassageInput(
            site_id=site_id,
            installation_id=site_id,
            passage_id=passage_id,
            passage_start=p_info.get("start_time") or "",
            passage_end=p_info.get("end_time") or "",
            observed_platform=p_info.get("observed_platform"),
            possible_platforms=p_info.get("possible_platforms", []),
            direction=p_info.get("direction", "unknown"),
        )

        vis_dict = b_dict.get("visual", {})
        num_cands = []
        for c in vis_dict.get("service_numbers", []):
            if isinstance(c, NumberCandidate):
                num_cands.append(c)
            elif isinstance(c, dict):
                n_str = c.get("number") or c.get("train_number") or ""
                coaches = c.get("distinct_coaches") or c.get("coach_count") or 1
                obs = c.get("total_observations") or 1
                conf = c.get("mean_confidence") or c.get("score") or 0.90
                num_cands.append(NumberCandidate(number=n_str, distinct_coaches=coaches, total_observations=obs, mean_confidence=conf))

        name_cands = []
        for c in vis_dict.get("train_name_candidates", []):
            if isinstance(c, NameCandidate):
                name_cands.append(c)
            elif isinstance(c, dict):
                nm_str = c.get("name") or c.get("train_name") or ""
                coaches = c.get("distinct_coaches") or c.get("coach_count") or 1
                obs = c.get("total_observations") or 1
                name_cands.append(NameCandidate(name=nm_str, distinct_coaches=coaches, total_observations=obs))

        consensus = PassageVisualConsensus(
            service_number_candidates=num_cands,
            stable_pair=vis_dict.get("stable_pair", []),
            train_name_candidates=name_cands,
            endpoint_candidates=vis_dict.get("endpoint_candidates", []),
            number_board_type=vis_dict.get("number_board_type", "UNKNOWN"),
            numeric_status=vis_dict.get("numeric_status", "UNKNOWN"),
        )

        # Extract T0/T1 NTES snapshot if available
        ntes_before_dict = b_dict.get("ntes", {}).get("pre_or_near_passage_snapshot")
        ntes_before = None
        if ntes_before_dict:
            ntes_before = NtesSnapshot(
                site_id=ntes_before_dict.get("site_id", site_id),
                station_code=ntes_before_dict.get("station_code", ""),
                requested_at=ntes_before_dict.get("requested_at", ntes_before_dict.get("fetched_at", "")),
                fetched_at=ntes_before_dict.get("fetched_at", ""),
                available=ntes_before_dict.get("available", True),
                trains=[NtesTrainEntry(**t) for t in ntes_before_dict.get("trains", [])],
                raw_hash=ntes_before_dict.get("raw_hash", ntes_before_dict.get("raw_payload_checksum", "")),
                from_cache=ntes_before_dict.get("from_cache", False),
            )

        dec_t2 = resolve_shadow(
            passage=p_input,
            visual_evidence=VisualOcrEvidence(
                passage_id=passage_id,
                per_observation=[],
                passage_consensus=consensus,
            ),
            site_schedule_rows=tt_snapshot.rows,
            ntes_before=ntes_before,
            ntes_after=snapshot,
            site_aliases=site_aliases,
            config=site_cfg,
            stage="T2",
        )

        b_dict["shadow_result"]["T2"] = {
            "train_number": dec_t2.shadow_final_number,
            "status": dec_t2.shadow_status,
            "reason_codes": dec_t2.reason_codes,
            "post_fetch_kind": dec_t2.post_fetch_kind,
        }

        # Step 4: Re-evaluate comparison with production and ground truth
        prod_dict = b_dict.get("production", {})
        gt_dict = b_dict.get("ground_truth", {})

        prod_num = prod_dict.get("train_number")
        prod_conf = prod_dict.get("confidence", "LOW")
        gt_num = gt_dict.get("train_number")

        shadow_num = dec_t2.shadow_final_number
        shadow_status = dec_t2.shadow_status

        if prod_num and shadow_num:
            prod_vs_shadow = "MATCH" if prod_num == shadow_num else "CONFLICT"
        elif shadow_num and not prod_num:
            prod_vs_shadow = "SHADOW_ONLY"
        elif prod_num and not shadow_num:
            prod_vs_shadow = "PRODUCTION_ONLY"
        else:
            prod_vs_shadow = "BOTH_UNKNOWN"

        eval_cat, rev_req = categorize_comparison(
            prod_num=prod_num,
            prod_conf=prod_conf,
            shadow_num=shadow_num,
            shadow_status=shadow_status,
            ground_truth=gt_num,
        )

        # Failure Taxonomy Assignment
        failure_reason = None
        if shadow_status in ("UNKNOWN", "CONFLICT", "AMBIGUOUS") or prod_vs_shadow == "CONFLICT":
            if RC_VISUAL_NTES_CONFLICT in dec_t2.reason_codes:
                failure_reason = FT_VISUAL_NTES_CONFLICT
            elif RC_VISUAL_TIMETABLE_CONFLICT in dec_t2.reason_codes:
                failure_reason = FT_VISUAL_TIMETABLE_CONFLICT
            elif RC_TIMETABLE_NTES_CONFLICT in dec_t2.reason_codes:
                failure_reason = FT_TIMETABLE_NTES_CONFLICT
            elif RC_VISUAL_BOARD_NOT_NUMBERED in dec_t2.reason_codes and not consensus.train_name_candidates:
                failure_reason = FT_BOARD_NOT_NUMBERED
            elif RC_AMBIGUOUS_MULTI_CANDIDATE in dec_t2.reason_codes:
                failure_reason = FT_MULTIPLE_LIVE_CANDIDATES
            elif RC_NTES_UNAVAILABLE in dec_t2.reason_codes:
                failure_reason = FT_NTES_UNAVAILABLE
            elif RC_NTES_CORROBORATED_STALE in dec_t2.reason_codes:
                failure_reason = FT_NTES_STALE
            elif RC_SCHEDULE_OUTSIDE_WINDOW in dec_t2.reason_codes:
                failure_reason = FT_TIMETABLE_AMBIGUITY
            elif RC_SCHEDULE_UNAVAILABLE in dec_t2.reason_codes:
                failure_reason = FT_TIMETABLE_STALE
            else:
                failure_reason = FT_UNKNOWN

        b_dict["comparison"] = {
            "category": eval_cat,
            "production_vs_shadow": prod_vs_shadow,
            "review_required": rev_req,
            "failure_reason": failure_reason,
            "shadow_decisive_stage": "T2" if shadow_num else "T1",
        }

        # Step 5: Save updated bundle
        with open(bundle_file, "w", encoding="utf-8") as f:
            json.dump(b_dict, f, indent=2)

        results_file = self.root / "results" / "passage_results" / f"{passage_id}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(b_dict, f, indent=2)

        # Update task status
        with self._lock:
            if passage_id in self._tasks:
                t = self._tasks[passage_id]
                t.status = "PROCESSED"
                t.processed_at = datetime.now().isoformat()
                t.t2_train_number = shadow_num
                t.t2_status = shadow_status
                self._save_queue()

        return PassageBundle(**b_dict)

    def process_due(self, now_dt: Optional[datetime] = None) -> List[PassageBundle]:
        """
        Process all tasks whose scheduled verify time has arrived (§6).
        Only processes tasks whose scheduled_verify_at <= cur.
        """
        cur = now_dt or datetime.now()
        cur_norm = _to_naive_utc(cur)
        due: List[DelayedVerificationTask] = []
        with self._lock:
            for t in self._tasks.values():
                if t.status == "PENDING":
                    try:
                        s_dt = datetime.fromisoformat(t.scheduled_verify_at)
                        if _to_naive_utc(s_dt) <= cur_norm:
                            due.append(t)
                    except Exception as e:
                        logger.warning(f"Error parsing scheduled_verify_at for task {t.passage_id}: {e}")

        results: List[PassageBundle] = []
        for task in due:
            res = self.process_passage(task.passage_id)
            if res:
                results.append(res)
        return results

    def process_all_pending(self) -> List[PassageBundle]:
        """
        Immediately process all pending verification tasks.
        WARNING: Do not use for the real operational trial, as it executes checks before +180s.
        """
        pending = self.get_pending_tasks()
        results: List[PassageBundle] = []
        for task in pending:
            res = self.process_passage(task.passage_id)
            if res:
                results.append(res)
        return results

    def run_worker_loop(
        self,
        poll_interval_s: float = 5.0,
        stop_event: Optional[threading.Event] = None,
        max_iterations: Optional[int] = None,
    ) -> int:
        """
        Periodic worker loop (§6).
        Periodically calls process_due() and only processes tasks whose
        scheduled_verify_at has passed.
        Never calls process_all_pending() to avoid executing checks before +180 seconds.
        """
        logger.info(f"Starting DelayedNtesVerificationWorker loop (interval: {poll_interval_s}s)")
        iterations = 0
        total_processed = 0

        while stop_event is None or not stop_event.is_set():
            try:
                processed = self.process_due()
                total_processed += len(processed)
            except Exception as e:
                logger.error(f"Error in delayed worker loop cycle: {e}")

            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break

            if stop_event is not None:
                stop_event.wait(poll_interval_s)
            else:
                time.sleep(poll_interval_s)

        logger.info(f"DelayedNtesVerificationWorker loop stopped. Total processed: {total_processed}")
        return total_processed

    def start_service(self, poll_interval_s: float = 5.0) -> None:
        """Start the worker loop service in a background daemon thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._stop_event = threading.Event()
            self._thread = threading.Thread(
                target=self.run_worker_loop,
                args=(poll_interval_s, self._stop_event),
                name="delayed-ntes-worker",
                daemon=True,
            )
            self._thread.start()

    def stop_service(self, timeout_s: float = 5.0) -> None:
        """Stop background worker service."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            if hasattr(self, "_stop_event") and self._stop_event:
                self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout_s)
