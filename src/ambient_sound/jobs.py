"""Thread-safe persistent background job manager for long sequence runs."""

from __future__ import annotations

import ctypes
import os
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ambient_sound.models import JobRecord

if TYPE_CHECKING:
    from ambient_sound.session import SessionManager


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _pid_is_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if sys.platform == "win32":
        process_query_limited_info = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_info, False, int(pid))
        if not handle:
            # ERROR_ACCESS_DENIED (5) means process exists under another privilege
            return ctypes.windll.kernel32.GetLastError() == 5
        try:
            exit_code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            return bool(ok and exit_code.value == 259)
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


class JobManager:
    """Persist and coordinate long-running background jobs on disk."""

    def __init__(self, workspace_dir: Path | str) -> None:
        self.workspace_dir = Path(workspace_dir).expanduser().resolve()
        self.jobs_dir = self.workspace_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cancel_events: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self.reconcile_stale_jobs()

    def reconcile_stale_jobs(self) -> int:
        """Reconcile any queued/running jobs whose owning process has exited."""
        if not self.jobs_dir.exists():
            return 0
        reconciled = 0
        for p in self.jobs_dir.glob("job_*.json"):
            try:
                before = self._load_job_raw(p.stem)
                if before.status in {"queued", "running"}:
                    after = self.get_job(p.stem, reconcile_stale=True)
                    if after.status != before.status:
                        reconciled += 1
            except Exception:
                continue
        return reconciled

    def _assert_output_dir_free(self, output_dir: str | Path | None) -> str | None:
        if output_dir is None or not str(output_dir).strip():
            return None
        resolved = str(Path(output_dir).expanduser().resolve())
        for job in self.list_jobs(limit=100, reconcile_stale=True):
            if job.status in {"queued", "running"} and job.output_dir == resolved:
                raise ValueError(
                    f"output_dir '{resolved}' is already in use by active job '{job.job_id}'."
                )
        return resolved

    def _job_path(self, job_id: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", str(job_id)):
            raise ValueError(f"Invalid job_id '{job_id}'.")
        return self.jobs_dir / f"{job_id}.json"

    def _save_job(self, record: JobRecord) -> None:
        with self._lock:
            self.jobs_dir.mkdir(parents=True, exist_ok=True)
            path = self._job_path(record.job_id)
            tmp = path.with_name(f".{record.job_id}.{uuid.uuid4().hex[:6]}.tmp")
            tmp.write_text(record.model_dump_json(indent=2), encoding="utf-8")
            os.replace(tmp, path)

    def _load_job_raw(self, job_id: str) -> JobRecord:
        path = self._job_path(job_id)
        if not path.exists():
            raise FileNotFoundError(f"Job '{job_id}' not found.")
        return JobRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def _launch_background_job(
        self,
        *,
        kind: str,
        initial_total: int,
        summary_items: list[dict[str, Any]],
        runner: Callable[[Callable[..., None], Callable[[], bool]], dict[str, Any]],
        output_dir: str | None = None,
    ) -> JobRecord:
        """Create, persist, and launch a cancellable background job thread."""
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        now = _utc_now()
        record = JobRecord(
            job_id=job_id,
            kind=kind,
            status="queued",
            owner_pid=os.getpid(),
            created_at=now,
            updated_at=now,
            total_items=int(initial_total),
            completed_items=0,
            current_item_index=None,
            current_stage="queued",
            progress_pct=0.0,
            item_progress_pct=0.0,
            phase="queued",
            bytes_written=0,
            bytes_total=0,
            elapsed_s=0.0,
            eta_s=None,
            output_dir=output_dir,
            cancel_requested=False,
            items_summary=list(summary_items),
        )
        cancel_ev = threading.Event()
        with self._lock:
            self._cancel_events[job_id] = cancel_ev
            self._save_job(record)

        def _worker() -> None:
            started_wall = time.monotonic()
            with self._lock:
                cur = self._load_job_raw(job_id)
                if cur.cancel_requested or cancel_ev.is_set():
                    cur.status = "cancelled"
                    cur.current_stage = "cancelled"
                    cur.phase = "cancelled"
                    cur.updated_at = _utc_now()
                    self._save_job(cur)
                    return
                cur.status = "running"
                cur.current_stage = "starting"
                cur.phase = "starting"
                cur.updated_at = _utc_now()
                self._save_job(cur)

            def _progress_cb(
                idx: int,
                total: int,
                sid: str,
                stage: str,
                completed_count: int,
                take_id: str | None = None,
                *,
                item_progress_pct: float | None = None,
                phase: str = "",
                bytes_written: int = 0,
                bytes_total: int = 0,
                item_meta: dict[str, Any] | None = None,
            ) -> None:
                with self._lock:
                    rec = self._load_job_raw(job_id)
                    elapsed = max(0.0, time.monotonic() - started_wall)
                    rec.current_item_index = idx
                    rec.current_stage = f"{sid}: {stage}"
                    rec.completed_items = completed_count
                    rec.phase = phase or rec.phase
                    rec.bytes_written = int(bytes_written)
                    rec.bytes_total = int(bytes_total)
                    rec.elapsed_s = round(elapsed, 1)
                    if item_progress_pct is not None:
                        ipct = round(max(0.0, min(100.0, float(item_progress_pct))), 1)
                    elif completed_count > idx:
                        ipct = 100.0
                    else:
                        ipct = 0.0
                    rec.item_progress_pct = ipct

                    if rec.kind != "process_sequence":
                        rec.total_items = max(int(total), 0)
                        if idx >= 0:
                            while len(rec.items_summary) <= idx:
                                rec.items_summary.append(
                                    {
                                        "label": sid,
                                        "kind": "item",
                                        "state": "pending",
                                        "detail": "",
                                        "session_id": None,
                                        "take_id": None,
                                        "output_path": None,
                                        "size_bytes": None,
                                    }
                                )
                            entry = rec.items_summary[idx]
                            entry["label"] = sid
                            if item_meta is not None:
                                for k, v in item_meta.items():
                                    entry[k] = v
                            else:
                                entry["state"] = "done" if completed_count > idx else "running"
                                entry["detail"] = stage
                                if take_id is not None:
                                    entry["take_id"] = take_id
                    elif 0 <= idx < len(rec.items_summary):
                        rec.items_summary[idx]["status"] = stage
                        rec.items_summary[idx]["state"] = (
                            "done" if stage == "done" else "running"
                        )
                        if take_id is not None:
                            rec.items_summary[idx]["take_id"] = take_id

                    denom = float(max(total, rec.total_items, 1))
                    in_item = (ipct / 100.0) if completed_count <= idx else 0.0
                    frac = float(completed_count) + in_item
                    pct = round((frac / denom) * 100.0, 2)
                    rec.progress_pct = (
                        min(99.9, pct) if completed_count < int(denom) else min(100.0, pct)
                    )
                    if frac >= 0.01 and elapsed > 0.2 and frac < denom:
                        rec.eta_s = round(max(0.0, elapsed * ((denom - frac) / frac)), 1)
                    elif completed_count >= int(denom):
                        rec.eta_s = 0.0
                    rec.updated_at = _utc_now()
                    self._save_job(rec)

            def _is_cancelled() -> bool:
                if cancel_ev.is_set():
                    return True
                with self._lock:
                    try:
                        rec = self._load_job_raw(job_id)
                        return bool(rec.cancel_requested)
                    except FileNotFoundError:
                        return True

            try:
                result = runner(_progress_cb, _is_cancelled)
            except Exception as exc:
                from ambient_sound.session import SequenceCancelledError

                with self._lock:
                    rec = self._load_job_raw(job_id)
                    rec.elapsed_s = round(max(0.0, time.monotonic() - started_wall), 1)
                    if isinstance(exc, SequenceCancelledError) or cancel_ev.is_set():
                        rec.status = "cancelled"
                        rec.current_stage = "cancelled"
                        rec.phase = "cancelled"
                        rec.error = str(exc)
                    else:
                        rec.status = "failed"
                        rec.current_stage = "failed"
                        rec.phase = "failed"
                        rec.error = f"{type(exc).__name__}: {exc}"
                    rec.updated_at = _utc_now()
                    self._save_job(rec)
                return

            with self._lock:
                rec = self._load_job_raw(job_id)
                rec.status = "completed"
                rec.current_stage = "completed"
                rec.phase = "done"
                rec.completed_items = max(rec.total_items, rec.completed_items)
                rec.item_progress_pct = 100.0
                rec.progress_pct = 100.0
                rec.elapsed_s = round(max(0.0, time.monotonic() - started_wall), 1)
                rec.eta_s = 0.0
                rec.result = result
                rec.updated_at = _utc_now()
                self._save_job(rec)

        t = threading.Thread(
            target=_worker,
            name=f"ambientsound-{job_id}",
            daemon=True,
        )
        with self._lock:
            self._threads[job_id] = t
        t.start()
        return record

    def start_sequence_job(
        self,
        session_mgr: SessionManager,
        *,
        items: list[dict[str, Any]],
        shared_prefix_steps: list[dict[str, Any]] | None = None,
        shared_suffix_steps: list[dict[str, Any]] | None = None,
        outer_fade_in_s: float = 0.0,
        outer_fade_out_s: float = 0.0,
        seam_context_s: float = 8.0,
        verify_seams: bool = True,
        clean_sections: bool = True,
        export_format: str = "wav",
        discard_take_audio_after_export: bool = False,
        max_sample_step_excess_db: float = 18.0,
        output_dir: str | Path | None = None,
        compile_clean_targets: list[str] | None = None,
        compile_clean_tiers: list[str] | None = None,
        include_combined_full: bool = False,
        clean_fade_in_s: float = 2.0,
        clean_fade_out_s: float = 3.0,
        filename_prefix: str = "",
    ) -> JobRecord:
        """Create and launch a background `process_sequence` job."""
        resolved_out = self._assert_output_dir_free(output_dir)
        summary_items = [
            {
                "session_id": str(raw.get("session_id", "")),
                "status": "queued",
                "state": "pending",
                "take_id": None,
            }
            for raw in items
        ]

        def _run(
            progress_cb: Callable[..., None],
            cancel_check: Callable[[], bool],
        ) -> dict[str, Any]:
            return session_mgr.process_sequence(
                items=items,
                shared_prefix_steps=shared_prefix_steps,
                shared_suffix_steps=shared_suffix_steps,
                outer_fade_in_s=outer_fade_in_s,
                outer_fade_out_s=outer_fade_out_s,
                seam_context_s=seam_context_s,
                verify_seams_flag=verify_seams,
                clean_sections_flag=clean_sections,
                export_format=export_format,
                discard_take_audio_after_export=discard_take_audio_after_export,
                max_sample_step_excess_db=max_sample_step_excess_db,
                output_dir=output_dir,
                compile_clean_targets=compile_clean_targets,
                compile_clean_tiers=compile_clean_tiers,
                include_combined_full=include_combined_full,
                clean_fade_in_s=clean_fade_in_s,
                clean_fade_out_s=clean_fade_out_s,
                filename_prefix=filename_prefix,
                progress_callback=progress_cb,
                cancel_check=cancel_check,
            )

        return self._launch_background_job(
            kind="process_sequence",
            initial_total=len(items),
            summary_items=summary_items,
            runner=_run,
            output_dir=resolved_out,
        )

    def start_compile_clean_sections_job(
        self,
        session_mgr: SessionManager,
        *,
        sequence: list[tuple[str, str]],
        output_dir: str | Path,
        targets: list[str] | None = None,
        tiers: list[str] | None = None,
        include_full_masters: bool = False,
        include_combined_full: bool = False,
        filename_prefix: str = "",
        audio_format: str = "wav",
        fade_in_s: float = 2.0,
        fade_out_s: float = 3.0,
        max_edits_per_hour: float = 2.0,
        max_edited_pct: float = 1.0,
        include_low_severity: bool = False,
        round_to_file_local_minutes: bool = False,
        distinct_regions: bool = False,
        max_files: int | None = None,
    ) -> JobRecord:
        """Create and launch a background `compile_clean_sections` job."""
        if not sequence:
            raise ValueError("sequence must contain at least one (session_id, take_id) pair.")
        resolved_out = self._assert_output_dir_free(output_dir)

        def _run(
            progress_cb: Callable[..., None],
            cancel_check: Callable[[], bool],
        ) -> dict[str, Any]:
            return session_mgr.compile_clean_sections(
                sequence=sequence,
                output_dir=output_dir,
                targets=targets,
                tiers=tiers,
                include_full_masters=include_full_masters,
                include_combined_full=include_combined_full,
                filename_prefix=filename_prefix,
                audio_format=audio_format,
                fade_in_s=fade_in_s,
                fade_out_s=fade_out_s,
                max_edits_per_hour=max_edits_per_hour,
                max_edited_pct=max_edited_pct,
                include_low_severity=include_low_severity,
                round_to_file_local_minutes=round_to_file_local_minutes,
                distinct_regions=distinct_regions,
                max_files=max_files,
                progress_callback=progress_cb,
                cancel_check=cancel_check,
            )

        return self._launch_background_job(
            kind="compile_clean_sections",
            initial_total=0,
            summary_items=[],
            runner=_run,
            output_dir=resolved_out,
        )

    def wait_for_job(
        self,
        job_id: str,
        timeout_s: float = 30.0,
        poll_interval_s: float = 1.0,
    ) -> JobRecord:
        """Wait up to timeout_s (<=55 s) for a job to finish or progress, returning latest state."""
        cap = max(0.0, min(55.0, float(timeout_s)))
        step = max(0.1, min(5.0, float(poll_interval_s)))
        deadline = time.monotonic() + cap
        rec = self.get_job(job_id)
        while rec.status in {"queued", "running"}:
            rem = deadline - time.monotonic()
            if rem <= 0.0:
                break
            time.sleep(min(step, rem))
            rec = self.get_job(job_id)
        return rec

    def get_job(self, job_id: str, *, reconcile_stale: bool = True) -> JobRecord:
        """Fetch latest state of a job, reconciling interrupted runs safely across processes."""
        with self._lock:
            rec = self._load_job_raw(job_id)
            if reconcile_stale and rec.status in {"queued", "running"}:
                th = self._threads.get(job_id)
                same_process = rec.owner_pid is None or rec.owner_pid == os.getpid()
                dead_here = same_process and (th is None or not th.is_alive())
                dead_remote = (not same_process) and (not _pid_is_alive(rec.owner_pid))
                if dead_here or dead_remote:
                    rec.status = "cancelled" if rec.cancel_requested else "failed"
                    rec.current_stage = rec.status
                    if not rec.error:
                        rec.error = "interrupted by server restart"
                    rec.updated_at = _utc_now()
                    self._save_job(rec)
            return rec

    def list_jobs(
        self,
        status: str | None = None,
        limit: int = 20,
        *,
        reconcile_stale: bool = True,
    ) -> list[JobRecord]:
        """List persisted jobs newest-first, optionally filtering by status."""
        if not self.jobs_dir.exists():
            return []
        records: list[JobRecord] = []
        for p in self.jobs_dir.glob("job_*.json"):
            try:
                rec = self.get_job(p.stem, reconcile_stale=reconcile_stale)
            except Exception:
                continue
            if status and rec.status != status:
                continue
            records.append(rec)
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[: max(1, int(limit))]

    def cancel_job(self, job_id: str) -> JobRecord:
        """Request cooperative cancellation of a queued or running job."""
        with self._lock:
            rec = self._load_job_raw(job_id)
            if rec.status in {"completed", "failed", "cancelled"}:
                return rec
            rec.cancel_requested = True
            ev = self._cancel_events.get(job_id)
            if ev is not None:
                ev.set()
            th = self._threads.get(job_id)
            if th is None or not th.is_alive():
                rec.status = "cancelled"
                rec.current_stage = "cancelled"
            rec.updated_at = _utc_now()
            self._save_job(rec)
            return rec
