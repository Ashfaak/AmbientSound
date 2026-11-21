"""Non-destructive session, take DAG, and loudness-matched A/B test management."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from scipy import signal

from ambient_sound import __version__
from ambient_sound.analysis import (
    _LOCALIZED_DISTRACTION_KINDS,
    SensitivityLevel,
    StreamingLUFS,
    StreamingTruePeak,
    _finite_or_none,
    analyze_audio,
    analyze_file,
    chunked_true_peak_linear,
    compute_clean_sections,
    event_kind_counts,
    extract_localized_distraction_spans,
    extract_localized_edit_spans,
    summarize_events,
)
from ambient_sound.config import Settings, ensure_path_allowed
from ambient_sound.dsp import (
    apply_recipe,
    apply_recipe_streaming,
    apply_recipe_with_seam_padding,
    compute_residual,
)
from ambient_sound.engines import compute_comparison_guardrails
from ambient_sound.io_audio import (
    CompilationCancelledError,
    compile_audio_span,
    copy_file_with_sha256,
    export_audio,
    ingest_audio_to_wav,
    iter_audio_blocks,
    load_audio,
    load_audio_slice,
    probe_audio,
    save_wav,
    sha256_file,
)
from ambient_sound.models import (
    ABTestRecord,
    ABVerdict,
    AudioMetrics,
    DistractionEvent,
    FilterStep,
    ProcessingRecipe,
    SessionSummary,
    StepTelemetry,
    TakeRecord,
)
from ambient_sound.visuals import (
    render_diff_spectrogram_png,
    render_file_diff_spectrogram_png,
    render_file_spectrogram_png,
    render_seam_spectrogram_png,
    render_spectrogram_png,
)

# Files longer than 120 seconds automatically stream on disk instead of loading into RAM.
STREAMING_THRESHOLD_S = 120.0


_MAX_INLINE_EVENTS = 200


class SequenceCancelledError(RuntimeError):
    """Raised cleanly when a background sequence job receives cooperative cancellation."""


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip()).strip("-_").lower()
    return cleaned[:32] or "session"


def _patched_spans_from_telemetry(step_metrics: list[StepTelemetry]) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for st in step_metrics:
        for item in st.details.get("patched_spans") or []:
            try:
                spans.append((float(item["target_start_s"]), float(item["target_end_s"])))
            except (KeyError, TypeError, ValueError):
                continue
    return spans


def _exclude_events_for_take(
    events: list[DistractionEvent], duration_s: float
) -> list[tuple[float, float]]:
    """Extract localized distraction intervals (skipping global tones and texture ticks)."""
    out: list[tuple[float, float]] = []
    loc_kinds = {
        "slow_pass",
        "low_freq_rumble",
        "transient_distraction",
        "clipping",
        "synthetic_beep",
        "click_train",
        "crackle_burst",
        "human_voice",
        "dog_bark",
        "variable_rpm_engine",
        "channel_biased_noise",
        "wind_noise",
    }
    for ev in events:
        if ev.end_s <= ev.start_s:
            continue
        is_local_drone = ev.kind == "machine_drone" and (
            duration_s > 0.0 and (ev.end_s - ev.start_s) < 0.5 * duration_s
        )
        if ev.kind in loc_kinds or is_local_drone:
            out.append((max(0.0, float(ev.start_s)), float(ev.end_s)))
    return out


def _next_prefixed_id(existing_ids: list[str], prefix: str) -> str:
    max_num = 0 if prefix == "ab_" else -1
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for eid in existing_ids:
        m = pattern.match(eid)
        if m:
            max_num = max(max_num, int(m.group(1)))
    return f"{prefix}{max_num + 1:03d}"


class SessionManager:
    """Manages immutable audio takes, recipes, spectrograms, and A/B feedback on disk."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.sessions_root = self.settings.workspace_dir / "sessions"
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self._locks_guard = threading.Lock()
        self._session_locks: dict[str, threading.RLock] = {}

    def _get_session_lock(self, session_id: str) -> threading.RLock:
        with self._locks_guard:
            lk = self._session_locks.get(session_id)
            if lk is None:
                lk = threading.RLock()
                self._session_locks[session_id] = lk
            return lk

    def _validate_step_paths(self, steps: list[FilterStep]) -> list[FilterStep]:
        out: list[FilterStep] = []
        for st in steps:
            if st.filter_name == "declip_from_backup":
                raw_bk = st.params.get("backup_audio_path")
                if raw_bk is None or not str(raw_bk).strip():
                    raise ValueError("declip_from_backup requires 'backup_audio_path'.")
                allowed_bk = ensure_path_allowed(str(raw_bk), self.settings)
                params = dict(st.params)
                params["backup_audio_path"] = str(allowed_bk)
                out.append(FilterStep(filter_name=st.filter_name, params=params))
            else:
                out.append(st)
        return out

    def _session_dir(self, session_id: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", session_id):
            raise ValueError(f"Invalid session_id '{session_id}'.")
        return self.sessions_root / session_id

    def _take_dir(self, session_id: str, take_id: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", take_id):
            raise ValueError(f"Invalid take_id '{take_id}'.")
        return self._session_dir(session_id) / "takes" / take_id

    def _session_json_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "session.json"

    def _save_session(self, summary: SessionSummary) -> None:
        with self._get_session_lock(summary.session_id):
            sdir = self._session_dir(summary.session_id)
            sdir.mkdir(parents=True, exist_ok=True)
            path = self._session_json_path(summary.session_id)
            tmp = path.with_name(f".session.{uuid4().hex[:6]}.tmp")
            tmp.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
            os.replace(tmp, path)

    def get_session(self, session_id: str) -> SessionSummary:
        """Load an existing session summary from disk."""
        meta_path = self._session_json_path(session_id)
        if not meta_path.exists():
            raise FileNotFoundError(f"Session '{session_id}' not found.")
        return SessionSummary.model_validate_json(meta_path.read_text(encoding="utf-8"))

    def list_sessions(self) -> list[SessionSummary]:
        """List all sessions sorted newest-first."""
        summaries: list[SessionSummary] = []
        if not self.sessions_root.exists():
            return summaries
        for entry in self.sessions_root.iterdir():
            meta_path = entry / "session.json"
            if not (entry.is_dir() and meta_path.exists()):
                continue
            try:
                summaries.append(
                    SessionSummary.model_validate_json(meta_path.read_text(encoding="utf-8"))
                )
            except (FileNotFoundError, OSError):
                continue
        summaries.sort(key=lambda s: s.created_at, reverse=True)
        return summaries

    def get_take(self, session_id: str, take_id: str) -> TakeRecord:
        """Return a specific TakeRecord within a session."""
        summary = self.get_session(session_id)
        for take in summary.takes:
            if take.take_id == take_id:
                return take
        raise FileNotFoundError(f"Take '{take_id}' not found in session '{session_id}'.")

    def _write_take_events(
        self, take_dir: Path, events: list[DistractionEvent]
    ) -> tuple[Path, dict[str, int], list[DistractionEvent]]:
        events_file = take_dir / "events.json"
        events_file.write_text(
            json.dumps([e.model_dump() for e in events], indent=2),
            encoding="utf-8",
        )
        counts = event_kind_counts(events)
        bounded, _, _, _ = summarize_events(events, max_events=_MAX_INLINE_EVENTS)
        return events_file, counts, bounded

    def load_take_events(self, session_id: str, take_id: str) -> list[DistractionEvent]:
        """Load the full event list for a take from events.json (or inline fallback)."""
        take = self.get_take(session_id, take_id)
        if take.events_path:
            ep = Path(take.events_path)
            if ep.exists():
                raw = json.loads(ep.read_text(encoding="utf-8"))
                return [DistractionEvent.model_validate(item) for item in raw]
        return list(take.events)

    def create_session(
        self, source_path: str | Path, name: str = "", copy_source: bool = True
    ) -> SessionSummary:
        """Ingest an audio recording into immutable take_000 in bounded RAM."""
        src = ensure_path_allowed(source_path, self.settings)

        session_name = name.strip() or src.stem
        session_id = f"{_slugify(session_name)}_{uuid4().hex[:6]}"
        sdir = self._session_dir(session_id)
        take_dir = sdir / "takes" / "take_000"
        take_dir.mkdir(parents=True, exist_ok=True)
        (sdir / "ab_tests").mkdir(parents=True, exist_ok=True)

        audio_wav = take_dir / "audio.wav"
        linked_in_place = False
        if not copy_source and src.suffix.lower() in {".wav", ".wave"}:
            try:
                src_info = probe_audio(src)
                if src_info.format_name.upper() in {"WAV", "RF64", "WAVEX"}:
                    audio_wav = src
                    linked_in_place = True
            except Exception:
                linked_in_place = False
        if not linked_in_place:
            ingest_audio_to_wav(src, audio_wav)

        metrics, events = analyze_file(audio_wav)
        spec_png = render_file_spectrogram_png(
            audio_wav,
            take_dir / "spectrogram.png",
            title=f"{session_name} — take_000 (Original)",
            events=events,
        )
        events_file, counts, bounded_events = self._write_take_events(take_dir, events)

        take_000 = TakeRecord(
            take_id="take_000",
            session_id=session_id,
            label="Original",
            parent_take_id=None,
            created_at=_utc_now_iso(),
            audio_path=str(audio_wav),
            residual_path=None,
            spectrogram_path=str(spec_png),
            diff_spectrogram_path=None,
            events_path=str(events_file),
            sha256=sha256_file(audio_wav),
            source_offset_s=0.0,
            timeline_modified=False,
            recipe=None,
            metrics=metrics,
            event_counts=counts,
            events=bounded_events,
            step_metrics=[],
        )
        summary = SessionSummary(
            session_id=session_id,
            name=session_name,
            created_at=_utc_now_iso(),
            source_path=str(src),
            takes=[take_000],
            ab_tests=[],
        )
        self._save_session(summary)
        return summary

    def analyze_take(
        self,
        session_id: str,
        take_id: str,
        sensitivity: SensitivityLevel = "medium",
        min_event_s: float = 0.12,
        max_event_s: float = 45.0,
        freq_range_hz: tuple[float, float] | None = None,
        event_kinds: list[str] | None = None,
    ) -> tuple[AudioMetrics, list[DistractionEvent]]:
        """Compute fresh metrics and distraction events for an entire take."""
        take = self.get_take(session_id, take_id)
        return analyze_file(
            take.audio_path,
            sensitivity=sensitivity,
            min_event_s=min_event_s,
            max_event_s=max_event_s,
            freq_range_hz=freq_range_hz,
            event_kinds=event_kinds,
        )

    def analyze_region(
        self,
        session_id: str,
        take_id: str,
        start_s: float = 0.0,
        end_s: float | None = None,
        sensitivity: SensitivityLevel = "medium",
        min_event_s: float = 0.12,
        max_event_s: float = 45.0,
        freq_range_hz: tuple[float, float] | None = None,
        event_kinds: list[str] | None = None,
    ) -> tuple[AudioMetrics, list[DistractionEvent]]:
        """Analyze a specific [start_s, end_s] time window of a take with custom sensitivity."""
        take = self.get_take(session_id, take_id)
        return analyze_file(
            take.audio_path,
            start_s=start_s,
            end_s=end_s,
            sensitivity=sensitivity,
            min_event_s=min_event_s,
            max_event_s=max_event_s,
            freq_range_hz=freq_range_hz,
            event_kinds=event_kinds,
        )

    def render_take_visuals(
        self,
        session_id: str,
        take_id: str,
        compare_to_take_id: str = "",
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> dict[str, str]:
        """Render full-file overview or zoomed [start_s, end_s] spectrogram & diff PNGs."""
        take = self.get_take(session_id, take_id)
        take_events = self.load_take_events(session_id, take_id)
        take_dir = self._take_dir(session_id, take_id)
        take_dir.mkdir(parents=True, exist_ok=True)
        windowed = start_s is not None or end_s is not None
        tag = ""
        if windowed:
            s0 = 0.0 if start_s is None else float(start_s)
            s1 = take.metrics.duration_s if end_s is None else float(end_s)
            tag = f"_{s0:.2f}_{s1:.2f}"

        spec_file = take_dir / f"spectrogram{tag}.png"
        title_sfx = (
            f" [{start_s or 0.0:.1f}s-{end_s or take.metrics.duration_s:.1f}s]"
            if windowed
            else ""
        )
        spec_path = render_file_spectrogram_png(
            take.audio_path,
            spec_file,
            title=f"{take.take_id} ({take.label}){title_sfx}",
            events=take_events,
            start_s=start_s,
            end_s=end_s,
        )
        result: dict[str, str] = {"spectrogram_path": str(spec_path)}

        if compare_to_take_id.strip():
            ref_take = self.get_take(session_id, compare_to_take_id.strip())
            before_off = (
                round(take.source_offset_s - ref_take.source_offset_s, 4)
                if not take.timeline_modified
                and not ref_take.timeline_modified
                and take.source_offset_s >= ref_take.source_offset_s
                else 0.0
            )
            diff_file = take_dir / f"diff_vs_{ref_take.take_id}{tag}.png"
            diff_path = render_file_diff_spectrogram_png(
                ref_take.audio_path,
                take.audio_path,
                diff_file,
                title=f"{ref_take.take_id} -> {take.take_id} ({take.label}){title_sfx}",
                start_s=start_s,
                end_s=end_s,
                loudness_match_residual=True,
                patched_spans=_patched_spans_from_telemetry(take.step_metrics) or None,
                before_offset_s=before_off,
            )
            result["diff_spectrogram_path"] = str(diff_path)
        elif (
            not windowed
            and take.diff_spectrogram_path
            and Path(take.diff_spectrogram_path).exists()
        ):
            result["diff_spectrogram_path"] = take.diff_spectrogram_path
        return result

    def render_ab_diff_visual(
        self,
        session_id: str,
        ab_id: str,
        side: Literal["a", "b", "ab"] = "a",
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> Path:
        """Render timeline-aligned A/B/Original 3-panel diff PNG (with .meta.json bounds)."""
        ab = self.get_ab_test(session_id, ab_id)
        take_a = self.get_take(session_id, ab.take_a_id)
        take_b = self.get_take(session_id, ab.take_b_id)
        orig_id = ab.original_take_id or "take_000"
        take_o = self.get_take(session_id, orig_id)

        blind_hide = bool(ab.blind and ab.verdict is None)
        if side == "a":
            before_t, after_t = take_o, take_a
            lbl = "Candidate A" if blind_hide else f"{take_a.take_id} ({take_a.label})"
            title = f"{take_o.take_id} -> {lbl}"
        elif side == "b":
            before_t, after_t = take_o, take_b
            lbl = "Candidate B" if blind_hide else f"{take_b.take_id} ({take_b.label})"
            title = f"{take_o.take_id} -> {lbl}"
        else:
            before_t, after_t = take_a, take_b
            title = (
                "Candidate A vs Candidate B"
                if blind_hide
                else f"{take_a.take_id} -> {take_b.take_id} (Candidate A vs B)"
            )

        before_off = (
            round(after_t.source_offset_s - before_t.source_offset_s, 4)
            if not after_t.timeline_modified
            and not before_t.timeline_modified
            and after_t.source_offset_s >= before_t.source_offset_s
            else (ab.original_offset_s if side in {"a", "b"} else 0.0)
        )

        windowed = start_s is not None or end_s is not None
        tag = ""
        sfx = ""
        if windowed:
            s0 = max(0.0, float(start_s or 0.0))
            s1 = after_t.metrics.duration_s if end_s is None else float(end_s)
            tag = f"_{s0:.2f}_{s1:.2f}"
            sfx = f" [{s0:.1f}s-{s1:.1f}s]"

        ab_dir = self._session_dir(session_id) / "ab_tests"
        ab_dir.mkdir(parents=True, exist_ok=True)
        out_png = ab_dir / f"{ab_id}_diff_{side}{tag}.png"
        meta_file = Path(f"{out_png}.meta.json")
        if out_png.exists() and meta_file.exists():
            return out_png

        patched_spans = _patched_spans_from_telemetry(after_t.step_metrics)
        if not patched_spans and after_t.parent_take_id:
            try:
                p_take = self.get_take(session_id, after_t.parent_take_id)
                p_spans = _patched_spans_from_telemetry(p_take.step_metrics)
                if p_spans and not after_t.timeline_modified:
                    shift = max(0.0, after_t.source_offset_s - p_take.source_offset_s)
                    dur_after = after_t.metrics.duration_s
                    patched_spans = [
                        (max(0.0, ps - shift), min(dur_after, pe - shift))
                        for ps, pe in p_spans
                        if pe - shift > 0.0 and ps - shift < dur_after
                    ]
            except FileNotFoundError:
                pass

        return render_file_diff_spectrogram_png(
            before_t.audio_path,
            after_t.audio_path,
            out_png,
            title=title + sfx,
            start_s=start_s,
            end_s=end_s,
            loudness_match_residual=True,
            patched_spans=patched_spans or None,
            before_offset_s=before_off,
        )

    def process_take(
        self,
        session_id: str,
        parent_take_id: str,
        steps: list[FilterStep] | list[dict[str, Any]],
        label: str = "",
        notes: str = "",
        prev_audio_path: str | Path | None = None,
        next_audio_path: str | Path | None = None,
        seam_context_s: float = 8.0,
    ) -> TakeRecord:
        """Apply an ordered filter chain to a parent take, producing a new immutable child take."""
        raw_steps = [
            s if isinstance(s, FilterStep) else FilterStep.model_validate(s) for s in steps
        ]
        parsed_steps = self._validate_step_paths(raw_steps)
        prev_p = (
            ensure_path_allowed(prev_audio_path, self.settings)
            if prev_audio_path is not None and str(prev_audio_path).strip()
            else None
        )
        next_p = (
            ensure_path_allowed(next_audio_path, self.settings)
            if next_audio_path is not None and str(next_audio_path).strip()
            else None
        )
        changes_length = any(
            st.filter_name in {"trim_range", "cut_regions"} for st in parsed_steps
        )
        if changes_length and (prev_p is not None or next_p is not None):
            raise ValueError(
                "Boundary seam padding (prev_audio_path / next_audio_path) cannot be combined "
                "with trim_range or cut_regions."
            )
        seam_info: dict[str, Any] | None = (
            {
                "prev_audio_path": str(prev_p) if prev_p else None,
                "next_audio_path": str(next_p) if next_p else None,
                "seam_context_s": float(seam_context_s),
            }
            if (prev_p is not None or next_p is not None)
            else None
        )

        with self._get_session_lock(session_id):
            summary = self.get_session(session_id)
            parent = self.get_take(session_id, parent_take_id)
            cur_offset = parent.source_offset_s
            timeline_mod = parent.timeline_modified
            for st in parsed_steps:
                if st.filter_name == "trim_range":
                    cur_offset += max(0.0, float(st.params.get("start_s", 0.0)))
                elif st.filter_name == "cut_regions":
                    timeline_mod = True

            existing_take_ids = [t.take_id for t in summary.takes]
            takes_dir = self._session_dir(session_id) / "takes"
            if takes_dir.exists():
                existing_take_ids.extend(p.name for p in takes_dir.iterdir() if p.is_dir())
            new_take_id = _next_prefixed_id(existing_take_ids, "take_")

            take_label = label.strip() or f"Processed ({len(parsed_steps)} steps)"
            take_dir = takes_dir / new_take_id
            take_dir.mkdir(parents=True, exist_ok=True)
            audio_path = take_dir / "audio.wav"

            recipe = ProcessingRecipe(
                parent_take_id=parent_take_id,
                steps=parsed_steps,
                notes=notes,
                seam_padding=seam_info,
            )
            (take_dir / "recipe.json").write_text(
                recipe.model_dump_json(indent=2), encoding="utf-8"
            )

            parent_info = probe_audio(parent.audio_path)
            parent_events = self.load_take_events(session_id, parent_take_id)
            exclude_events = _exclude_events_for_take(parent_events, parent_info.duration_s)
            residual_wav: Path | None = None
            diff_png: Path | None = None
            step_metrics: list[StepTelemetry] = []

            if parent_info.duration_s <= STREAMING_THRESHOLD_S:
                parent_audio, sr = load_audio(parent.audio_path)
                processed, step_metrics = apply_recipe_with_seam_padding(
                    parent_audio,
                    sr,
                    parsed_steps,
                    prev_audio_path=prev_p,
                    next_audio_path=next_p,
                    seam_context_s=seam_context_s,
                    exclude_events=exclude_events,
                )
                save_wav(audio_path, processed, sr, subtype="FLOAT")

                metrics, events = analyze_audio(processed, sr)
                spec_png = render_spectrogram_png(
                    processed,
                    sr,
                    take_dir / "spectrogram.png",
                    title=f"{summary.name} — {new_take_id} ({take_label})",
                    events=events,
                )
                if not changes_length and processed.shape == parent_audio.shape:
                    residual = compute_residual(parent_audio, processed)
                    residual_wav = save_wav(
                        take_dir / "residual.wav", residual, sr, subtype="FLOAT"
                    )
                    diff_png = render_diff_spectrogram_png(
                        parent_audio,
                        processed,
                        sr,
                        take_dir / "diff_spectrogram.png",
                        title=f"{parent_take_id} -> {new_take_id} ({take_label})",
                        loudness_match_residual=True,
                        patched_spans=_patched_spans_from_telemetry(step_metrics) or None,
                    )
            else:
                out_info = apply_recipe_streaming(
                    parent.audio_path,
                    audio_path,
                    parsed_steps,
                    subtype="PCM_24",
                    telemetry_out=step_metrics,
                    prev_audio_path=prev_p,
                    next_audio_path=next_p,
                    seam_context_s=seam_context_s,
                    exclude_events=exclude_events,
                )
                metrics, events = analyze_file(audio_path)
                spec_png = render_file_spectrogram_png(
                    audio_path,
                    take_dir / "spectrogram.png",
                    title=f"{summary.name} — {new_take_id} ({take_label})",
                    events=events,
                )
                if (
                    not changes_length
                    and out_info.frames == parent_info.frames
                    and out_info.channels == parent_info.channels
                ):
                    diff_png = render_file_diff_spectrogram_png(
                        parent.audio_path,
                        audio_path,
                        take_dir / "diff_spectrogram.png",
                        title=f"{parent_take_id} -> {new_take_id} ({take_label})",
                        loudness_match_residual=True,
                        patched_spans=_patched_spans_from_telemetry(step_metrics) or None,
                    )

            events_file, counts, bounded_events = self._write_take_events(take_dir, events)

            bed_idxs = [i for i, st in enumerate(parsed_steps) if st.filter_name == "bed_shape"]
            if (
                bed_idxs
                and metrics.true_peak_dbtp is not None
                and metrics.true_peak_dbtp > -1.05
            ):
                last_bed = bed_idxs[-1]
                has_post_limit = any(
                    st.filter_name in {"true_peak_limiter", "loudness_normalize"}
                    for st in parsed_steps[last_bed + 1 :]
                )
                if not has_post_limit:
                    warn_msg = (
                        f"true_peak_dbtp={metrics.true_peak_dbtp:.2f} dBTP near/above -1.0 "
                        "ceiling after bed_shape; place true_peak_limiter after bed_shape."
                    )
                    for sm in step_metrics:
                        if sm.step_index == last_bed:
                            sm.details["headroom_warning"] = warn_msg
                    recipe.notes = f"{recipe.notes} | {warn_msg}".strip(" |")
                    (take_dir / "recipe.json").write_text(
                        recipe.model_dump_json(indent=2), encoding="utf-8"
                    )

            record = TakeRecord(
                take_id=new_take_id,
                session_id=session_id,
                label=take_label,
                parent_take_id=parent_take_id,
                created_at=_utc_now_iso(),
                audio_path=str(audio_path),
                residual_path=str(residual_wav) if residual_wav else None,
                spectrogram_path=str(spec_png),
                diff_spectrogram_path=str(diff_png) if diff_png else None,
                events_path=str(events_file),
                sha256=sha256_file(audio_path),
                source_offset_s=round(cur_offset, 4),
                timeline_modified=timeline_mod,
                seam_padding=seam_info,
                recipe=recipe,
                metrics=metrics,
                event_counts=counts,
                events=bounded_events,
                step_metrics=step_metrics,
            )
            summary.takes.append(record)
            self._save_session(summary)
            return record

    def load_take_audio_slice(
        self,
        session_id: str,
        take_id: str,
        start_s: float = 0.0,
        end_s: float | None = None,
    ) -> tuple[np.ndarray, int]:
        """Load a [start_s, end_s] slice of a take's audio in bounded RAM."""
        take = self.get_take(session_id, take_id)
        return load_audio_slice(take.audio_path, start_s=start_s, end_s=end_s, dtype="float64")

    def can_stream_residual(self, session_id: str, take_id: str) -> bool:
        """Return True if a take has a saved residual or matches parent shape for sliced diff."""
        take = self.get_take(session_id, take_id)
        if take.residual_path and Path(take.residual_path).exists():
            return True
        if not take.parent_take_id:
            return False
        parent = self.get_take(session_id, take.parent_take_id)
        t_info = probe_audio(take.audio_path)
        p_info = probe_audio(parent.audio_path)
        return t_info.frames == p_info.frames and t_info.channels == p_info.channels

    def load_take_residual_slice(
        self,
        session_id: str,
        take_id: str,
        start_s: float = 0.0,
        end_s: float | None = None,
    ) -> tuple[np.ndarray, int]:
        """Return loudness-normalized residual slice when gain shifts exceed 1.5 dB."""
        take = self.get_take(session_id, take_id)
        if take.parent_take_id:
            parent = self.get_take(session_id, take.parent_take_id)
            p_info = probe_audio(parent.audio_path)
            t_info = probe_audio(take.audio_path)
            if p_info.frames == t_info.frames and p_info.channels == t_info.channels:
                p_rms = parent.metrics.rms_dbfs
                t_rms = take.metrics.rms_dbfs
                if (
                    p_rms is not None
                    and t_rms is not None
                    and math.isfinite(p_rms)
                    and math.isfinite(t_rms)
                    and abs(t_rms - p_rms) > 1.5
                ):
                    parent_slice, sr = load_audio_slice(
                        parent.audio_path, start_s=start_s, end_s=end_s, dtype="float64"
                    )
                    take_slice, _ = load_audio_slice(
                        take.audio_path, start_s=start_s, end_s=end_s, dtype="float64"
                    )
                    n = min(parent_slice.shape[0], take_slice.shape[0])
                    scale = 10.0 ** ((p_rms - t_rms) / 20.0)
                    return parent_slice[:n] - take_slice[:n] * scale, sr

        if take.residual_path and Path(take.residual_path).exists():
            return load_audio_slice(
                take.residual_path, start_s=start_s, end_s=end_s, dtype="float64"
            )
        if not take.parent_take_id:
            raise FileNotFoundError(f"Residual not available for {take_id}.")
        parent = self.get_take(session_id, take.parent_take_id)
        p_info = probe_audio(parent.audio_path)
        t_info = probe_audio(take.audio_path)
        if p_info.frames != t_info.frames or p_info.channels != t_info.channels:
            raise FileNotFoundError(
                f"Residual unavailable for shape-changing take {take_id}."
            )
        parent_slice, sr = load_audio_slice(
            parent.audio_path, start_s=start_s, end_s=end_s, dtype="float64"
        )
        take_slice, _ = load_audio_slice(
            take.audio_path, start_s=start_s, end_s=end_s, dtype="float64"
        )
        n = min(parent_slice.shape[0], take_slice.shape[0])
        return parent_slice[:n] - take_slice[:n], sr

    def _resolve_original_take_id(
        self,
        summary: SessionSummary,
        take_a: TakeRecord,
        take_b: TakeRecord,
        explicit_original_id: str | None,
    ) -> tuple[str, float]:
        def _align_offset(ref: TakeRecord) -> float:
            if (
                not take_a.timeline_modified
                and not take_b.timeline_modified
                and not ref.timeline_modified
                and abs(take_a.source_offset_s - take_b.source_offset_s) <= 0.05
                and take_a.source_offset_s >= ref.source_offset_s
            ):
                return round(take_a.source_offset_s - ref.source_offset_s, 4)
            return 0.0

        if explicit_original_id and explicit_original_id.strip():
            ref = self.get_take(summary.session_id, explicit_original_id.strip())
            return ref.take_id, _align_offset(ref)

        take_map = {t.take_id: t for t in summary.takes}

        def _lineage(start_tid: str) -> list[TakeRecord]:
            chain: list[TakeRecord] = []
            seen: set[str] = set()
            cur: str | None = start_tid
            while cur and cur in take_map and cur not in seen:
                seen.add(cur)
                rec = take_map[cur]
                chain.append(rec)
                cur = rec.parent_take_id
            chain.reverse()
            return chain

        def _matches_both(cand: TakeRecord) -> bool:
            return (
                abs(cand.source_offset_s - take_a.source_offset_s) <= 0.05
                and abs(cand.source_offset_s - take_b.source_offset_s) <= 0.05
                and cand.timeline_modified == take_a.timeline_modified
                and cand.timeline_modified == take_b.timeline_modified
                and abs(cand.metrics.duration_s - take_a.metrics.duration_s) <= 0.1
                and abs(cand.metrics.duration_s - take_b.metrics.duration_s) <= 0.1
            )

        chain_a = _lineage(take_a.take_id)
        chain_b = _lineage(take_b.take_id)
        set_b = {t.take_id for t in chain_b}

        # Earliest shared common ancestor matching both candidate timelines
        for anc in chain_a:
            if anc.take_id in set_b and _matches_both(anc):
                return anc.take_id, 0.0

        # Otherwise deepest shared common ancestor (never another branch) with time offset
        lca: TakeRecord | None = None
        for anc in chain_a:
            if anc.take_id in set_b:
                lca = anc
        if lca is None:
            lca = take_map.get("take_000", chain_a[0])
        return lca.take_id, _align_offset(lca)

    def create_ab_test(
        self,
        session_id: str,
        take_a_id: str,
        take_b_id: str,
        focus_start_s: float | None = None,
        focus_end_s: float | None = None,
        blind: bool = False,
        notes: str = "",
        original_take_id: str | None = None,
        match_original: bool = False,
    ) -> ABTestRecord:
        """Create an A/B test matching candidate loudness by default (optional original match)."""
        with self._get_session_lock(session_id):
            summary = self.get_session(session_id)
            take_a = self.get_take(session_id, take_a_id)
            take_b = self.get_take(session_id, take_b_id)
            resolved_orig_id, resolved_orig_offset = self._resolve_original_take_id(
                summary, take_a, take_b, original_take_id
            )
            take_o = self.get_take(session_id, resolved_orig_id)

            candidate_lufs = [
                v
                for v in (take_a.metrics.integrated_lufs, take_b.metrics.integrated_lufs)
                if v is not None
            ]
            if match_original and take_o.metrics.integrated_lufs is not None:
                candidate_lufs.append(take_o.metrics.integrated_lufs)

            if candidate_lufs:
                target_lufs = min(candidate_lufs)

                def _attenuation_gain(lufs: float | None) -> float:
                    if lufs is None:
                        return 0.0
                    return round(min(0.0, target_lufs - lufs), 2)

                gain_a = _attenuation_gain(take_a.metrics.integrated_lufs)
                gain_b = _attenuation_gain(take_b.metrics.integrated_lufs)
                gain_o = _attenuation_gain(take_o.metrics.integrated_lufs)
            else:
                gain_a = gain_b = gain_o = 0.0

            ab_id = _next_prefixed_id([a.ab_id for a in summary.ab_tests], "ab_")
            record = ABTestRecord(
                ab_id=ab_id,
                session_id=session_id,
                created_at=_utc_now_iso(),
                take_a_id=take_a_id,
                take_b_id=take_b_id,
                original_take_id=resolved_orig_id,
                original_offset_s=resolved_orig_offset,
                focus_start_s=focus_start_s,
                focus_end_s=focus_end_s,
                blind=blind,
                match_original=match_original,
                notes=notes,
                gain_db_a=gain_a,
                gain_db_b=gain_b,
                gain_db_original=gain_o,
                verdict=None,
            )
            ab_path = self._session_dir(session_id) / "ab_tests" / f"{ab_id}.json"
            ab_path.parent.mkdir(parents=True, exist_ok=True)
            ab_tmp = ab_path.with_name(f".{ab_id}.{uuid4().hex[:6]}.tmp")
            ab_tmp.write_text(record.model_dump_json(indent=2), encoding="utf-8")
            os.replace(ab_tmp, ab_path)

            summary.ab_tests.append(record)
            self._save_session(summary)
            return record

    def get_ab_test(self, session_id: str, ab_id: str) -> ABTestRecord:
        """Retrieve a specific A/B test record."""
        summary = self.get_session(session_id)
        for ab in summary.ab_tests:
            if ab.ab_id == ab_id:
                return ab
        raise FileNotFoundError(f"A/B test '{ab_id}' not found in session '{session_id}'.")

    def list_ab_tests(self, session_id: str) -> list[ABTestRecord]:
        """List all A/B tests in a session."""
        return self.get_session(session_id).ab_tests

    def submit_ab_verdict(
        self,
        session_id: str,
        ab_id: str,
        winner: Literal["a", "b", "tie", "original"],
        artifact_tags: list[str] | None = None,
        comment: str = "",
    ) -> ABTestRecord:
        """Record human listening verdict & feedback and append to session feedback log."""
        with self._get_session_lock(session_id):
            summary = self.get_session(session_id)
            verdict = ABVerdict(
                winner=winner,
                artifact_tags=[t.strip() for t in (artifact_tags or []) if t.strip()],
                comment=comment.strip(),
                submitted_at=_utc_now_iso(),
            )

            target_ab: ABTestRecord | None = None
            for idx, ab in enumerate(summary.ab_tests):
                if ab.ab_id == ab_id:
                    ab.verdict = verdict
                    summary.ab_tests[idx] = ab
                    target_ab = ab
                    break
            if target_ab is None:
                raise FileNotFoundError(f"A/B test '{ab_id}' not found in session '{session_id}'.")

            ab_path = self._session_dir(session_id) / "ab_tests" / f"{ab_id}.json"
            ab_tmp = ab_path.with_name(f".{ab_id}.{uuid4().hex[:6]}.tmp")
            ab_tmp.write_text(target_ab.model_dump_json(indent=2), encoding="utf-8")
            os.replace(ab_tmp, ab_path)
            self._save_session(summary)

            log_path = self._session_dir(session_id) / "session_feedback_log.md"
            tags_str = ", ".join(verdict.artifact_tags) if verdict.artifact_tags else "none"
            entry = (
                f"## {target_ab.ab_id} ({verdict.submitted_at})\n"
                f"- **A:** `{target_ab.take_a_id}` | **B:** `{target_ab.take_b_id}` "
                f"| **Blind:** `{target_ab.blind}`\n"
                f"- **Winner:** `{verdict.winner}`\n"
                f"- **Artifact Tags:** {tags_str}\n"
                f"- **Comment:** {verdict.comment or '(none)'}\n\n"
            )
            if not log_path.exists():
                log_path.write_text(
                    f"# Feedback Log — {summary.name} (`{summary.session_id}`)\n\n{entry}",
                    encoding="utf-8",
                )
            else:
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(entry)

            return target_ab

    def _protected_export_paths(self, summary: SessionSummary, take: TakeRecord) -> set[Path]:
        protected: set[Path] = {Path(take.audio_path).resolve()}
        if summary.source_path:
            protected.add(Path(summary.source_path).resolve())
        for t in summary.takes:
            if t.take_id == "take_000":
                protected.add(Path(t.audio_path).resolve())
        return protected

    def export_take(
        self,
        session_id: str,
        take_id: str,
        output_path: str | Path,
        audio_format: str = "wav",
    ) -> Path:
        """Export a take to WAV, FLAC, MP3, or OGG inside allowed roots."""
        dst = ensure_path_allowed(output_path, self.settings)
        summary = self.get_session(session_id)
        take = self.get_take(session_id, take_id)
        if dst.resolve() in self._protected_export_paths(summary, take):
            raise ValueError("Refusing to overwrite immutable source or take audio.")
        return export_audio(
            take.audio_path, dst, take.metrics.sample_rate, audio_format=audio_format
        )

    def export_take_detailed(
        self,
        session_id: str,
        take_id: str,
        output_path: str | Path,
        audio_format: str = "wav",
        discard_take_audio: bool = False,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Export a take with verified file/loudness metadata and optional intermediate cleanup."""
        dst = ensure_path_allowed(output_path, self.settings)
        summary = self.get_session(session_id)
        take = self.get_take(session_id, take_id)
        if dst.resolve() in self._protected_export_paths(summary, take):
            raise ValueError("Refusing to overwrite immutable source or take audio.")

        fmt = audio_format.strip().lower()
        src_info = probe_audio(take.audio_path)
        has_cached_metrics = (
            take.metrics.integrated_lufs is not None
            and take.metrics.true_peak_dbtp is not None
        )
        if (
            fmt == "wav"
            and str(src_info.format_name).upper() in {"WAV", "RF64"}
            and has_cached_metrics
        ):
            try:
                written_bytes, digest = copy_file_with_sha256(
                    take.audio_path,
                    dst,
                    on_progress=on_progress,
                    cancel_check=cancel_check,
                )
            except CompilationCancelledError as exc:
                raise SequenceCancelledError(str(exc)) from exc
            out_p = dst
            info = probe_audio(out_p)
            size_bytes = int(written_bytes)
            out_lufs = take.metrics.integrated_lufs
            out_tp = take.metrics.true_peak_dbtp
            metrics_source = "cached_take_metrics"
        else:
            out_p = self.export_take(session_id, take_id, dst, audio_format=audio_format)
            info = probe_audio(out_p)
            size_bytes = out_p.stat().st_size if out_p.exists() else 0
            digest = sha256_file(out_p)
            lufs_acc = StreamingLUFS(info.sample_rate, info.channels)
            tp_acc = StreamingTruePeak(info.channels)
            processed_frames = 0
            for _, blk in iter_audio_blocks(out_p, block_seconds=30.0, dtype="float64"):
                if cancel_check is not None and cancel_check():
                    raise SequenceCancelledError("Export measurement cancelled.")
                lufs_acc.update(blk)
                tp_acc.update(blk)
                processed_frames += blk.shape[0]
                if on_progress is not None and info.frames > 0:
                    on_progress(processed_frames, info.frames)
            out_lufs = _finite_or_none(lufs_acc.finalize(), digits=3)
            out_tp = _finite_or_none(
                20.0 * math.log10(max(tp_acc.peak_lin, 1e-12)), digits=3
            )
            metrics_source = "measured_output"

        discarded = False
        if (
            discard_take_audio
            and take_id != "take_000"
            and out_p.exists()
            and size_bytes > 0
        ):
            with self._get_session_lock(session_id):
                summary = self.get_session(session_id)
                sdir_res = self._session_dir(session_id).resolve()
                exp_res = out_p.resolve()
                for t in summary.takes:
                    if t.take_id != take_id:
                        continue
                    cur_audio = Path(t.audio_path)
                    if (
                        cur_audio.exists()
                        and cur_audio.resolve() != exp_res
                        and sdir_res in cur_audio.resolve().parents
                    ):
                        cur_audio.unlink()
                        t.audio_path = str(out_p)
                        discarded = True
                    if t.residual_path:
                        res_p = Path(t.residual_path)
                        if res_p.exists() and sdir_res in res_p.resolve().parents:
                            res_p.unlink()
                        t.residual_path = None
                if discarded:
                    self._save_session(summary)

        src_file = Path(summary.source_path).name if summary.source_path else summary.name
        return {
            "session_id": session_id,
            "take_id": take_id,
            "source_file": src_file,
            "format": fmt,
            "container": str(info.format_name).upper(),
            "output_path": str(out_p),
            "size_bytes": int(size_bytes),
            "duration_s": _finite_or_none(info.duration_s, digits=3),
            "sample_rate": int(info.sample_rate),
            "channels": int(info.channels),
            "subtype": info.subtype,
            "integrated_lufs": out_lufs,
            "true_peak_dbtp": out_tp,
            "metrics_source": metrics_source,
            "sha256": digest,
            "take_audio_discarded": discarded,
        }

    def compare_takes(
        self,
        session_id: str,
        take_id: str,
        compare_to_take_id: str,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> dict[str, Any]:
        """Compare processed vs reference take metrics, band deltas, and bird guardrails."""
        take_proc = self.get_take(session_id, take_id)
        take_ref = self.get_take(session_id, compare_to_take_id)

        def _delta(a: float | None, b: float | None) -> float | None:
            if a is None or b is None:
                return None
            return _finite_or_none(float(a) - float(b), digits=4)

        if start_s is None and end_s is None:
            m_proc = take_proc.metrics
            m_ref = take_ref.metrics
            dur_cap = min(
                30.0,
                max(0.0, float(m_proc.duration_s)),
                max(0.0, float(m_ref.duration_s)),
            )
            ref_off = (
                max(0.0, take_proc.source_offset_s - take_ref.source_offset_s)
                if not take_proc.timeline_modified and not take_ref.timeline_modified
                else 0.0
            )
            proc_slice, sr_p = load_audio_slice(
                take_proc.audio_path, start_s=0.0, end_s=dur_cap, dtype="float64"
            )
            ref_slice, sr_r = load_audio_slice(
                take_ref.audio_path, start_s=ref_off, end_s=ref_off + dur_cap, dtype="float64"
            )
        else:
            s_t = max(0.0, float(start_s or 0.0))
            m_proc, _ = analyze_file(take_proc.audio_path, start_s=s_t, end_s=end_s)
            m_ref, _ = analyze_file(take_ref.audio_path, start_s=s_t, end_s=end_s)
            proc_slice, sr_p = load_audio_slice(
                take_proc.audio_path, start_s=s_t, end_s=end_s, dtype="float64"
            )
            ref_slice, sr_r = load_audio_slice(
                take_ref.audio_path, start_s=s_t, end_s=end_s, dtype="float64"
            )

        guardrails = (
            compute_comparison_guardrails(ref_slice, proc_slice, sr_p)
            if sr_p == sr_r and proc_slice.shape[0] >= 256 and ref_slice.shape[0] >= 256
            else {"bird_retention_db": None, "musical_noise_index": None, "guardrail_warnings": []}
        )

        deltas = {
            "lufs_delta_db": _delta(m_proc.integrated_lufs, m_ref.integrated_lufs),
            "rms_delta_db": _delta(m_proc.rms_dbfs, m_ref.rms_dbfs),
            "true_peak_delta_db": _delta(m_proc.true_peak_dbtp, m_ref.true_peak_dbtp),
            "sharpness_delta_acum": _delta(m_proc.sharpness_acum, m_ref.sharpness_acum),
            "ndsi_delta": _delta(m_proc.ndsi, m_ref.ndsi),
            "pleasantness_delta": _delta(m_proc.pleasantness_score, m_ref.pleasantness_score),
            "spectral_slope_delta": _delta(
                m_proc.spectral_slope_db_per_oct, m_ref.spectral_slope_db_per_oct
            ),
        }

        band_deltas: dict[str, dict[str, float | None]] = {}
        common_bands = [k for k in m_proc.band_metrics if k in m_ref.band_metrics]
        for bname in common_bands:
            bp = m_proc.band_metrics[bname]
            br = m_ref.band_metrics[bname]
            band_deltas[bname] = {
                "rms_delta_db": _delta(bp.get("rms_dbfs"), br.get("rms_dbfs")),
                "bed_p10_delta_db": _delta(bp.get("bed_p10_dbfs"), br.get("bed_p10_dbfs")),
                "peak_excess_delta_db": _delta(bp.get("peak_excess_db"), br.get("peak_excess_db")),
            }

        return {
            "take_id": take_id,
            "compare_to_take_id": compare_to_take_id,
            "deltas": deltas,
            "band_deltas_db": band_deltas,
            **guardrails,
        }

    def ab_ui_url(self, session_id: str, ab_id: str) -> str:
        """Build the local A/B studio URL for a test."""
        return f"{self.settings.ui_base_url}/?session={session_id}&ab={ab_id}"

    def delete_take(
        self, session_id: str, take_id: str, cascade: bool = False
    ) -> dict[str, Any]:
        """Delete a non-root take (and optional descendant subtree) plus orphaned A/B tests."""
        if take_id == "take_000":
            raise ValueError(
                "Cannot delete immutable root 'take_000'. Use delete_session to remove the session."
            )
        with self._get_session_lock(session_id):
            summary = self.get_session(session_id)
            take_ids = {t.take_id for t in summary.takes}
            if take_id not in take_ids:
                raise FileNotFoundError(f"Take '{take_id}' not found in session '{session_id}'.")

            children_by_parent: dict[str, list[str]] = {}
            for t in summary.takes:
                if t.parent_take_id:
                    children_by_parent.setdefault(t.parent_take_id, []).append(t.take_id)

            descendants: list[str] = []
            queue = list(children_by_parent.get(take_id, []))
            while queue:
                cid = queue.pop(0)
                if cid not in descendants:
                    descendants.append(cid)
                    queue.extend(children_by_parent.get(cid, []))

            if descendants and not cascade:
                raise ValueError(
                    f"Take '{take_id}' has child take(s) {descendants}. "
                    "Pass cascade=True to delete the take and its descendants."
                )

            to_delete = {take_id, *descendants}
            sdir = self._session_dir(session_id)
            for tid in sorted(to_delete):
                tdir = sdir / "takes" / tid
                if tdir.exists():
                    shutil.rmtree(tdir, ignore_errors=True)

            summary.takes = [t for t in summary.takes if t.take_id not in to_delete]
            deleted_ab_ids: list[str] = []
            kept_abs: list[ABTestRecord] = []
            for ab in summary.ab_tests:
                if (
                    ab.take_a_id in to_delete
                    or ab.take_b_id in to_delete
                    or ab.original_take_id in to_delete
                ):
                    deleted_ab_ids.append(ab.ab_id)
                    ab_file = sdir / "ab_tests" / f"{ab.ab_id}.json"
                    if ab_file.exists():
                        ab_file.unlink()
                else:
                    kept_abs.append(ab)
            summary.ab_tests = kept_abs
            self._save_session(summary)

            return {
                "session_id": session_id,
                "deleted_take_ids": sorted(to_delete),
                "deleted_ab_ids": deleted_ab_ids,
                "remaining_takes": len(summary.takes),
            }

    def delete_session(self, session_id: str) -> dict[str, Any]:
        """Permanently delete a session directory and all its takes and A/B tests."""
        with self._get_session_lock(session_id):
            self.get_session(session_id)
            sdir = self._session_dir(session_id)
            shutil.rmtree(sdir)
            return {"session_id": session_id, "deleted": True}

    def find_clean_sections(
        self,
        sequence: list[tuple[str, str]],
        max_edits_per_hour: float = 2.0,
        max_edited_pct: float = 1.0,
        include_low_severity: bool = False,
        round_to_file_local_minutes: bool = False,
        exclude_spans: list[tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        """Find largest untouched and minimal-edit ASMR stretches across one or more takes."""
        if not sequence:
            raise ValueError("Pass either (session_id, take_id) or a non-empty sequence.")
        segments: list[dict[str, Any]] = []
        for sid, tid in sequence:
            summary = self.get_session(sid)
            take = self.get_take(sid, tid)
            events = self.load_take_events(sid, tid)
            edit_spans = extract_localized_edit_spans(take.recipe)
            distraction_spans = extract_localized_distraction_spans(
                events, include_low_severity=include_low_severity
            )
            src_name = Path(summary.source_path).name if summary.source_path else summary.name
            segments.append(
                {
                    "label": src_name,
                    "session_id": sid,
                    "take_id": tid,
                    "duration_s": take.metrics.duration_s,
                    "edit_spans": edit_spans,
                    "distraction_spans": distraction_spans,
                }
            )
        return compute_clean_sections(
            segments,
            max_edits_per_hour=max_edits_per_hour,
            max_edited_pct=max_edited_pct,
            round_to_file_local_minutes=round_to_file_local_minutes,
            exclude_spans=exclude_spans,
        )

    def compile_clean_sections(
        self,
        sequence: list[tuple[str, str]],
        output_dir: str | Path,
        *,
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
        progress_callback: Callable[..., None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Compile clean YouTube ASMR sections (splitting/stitching across files) to output_dir."""
        if not sequence:
            raise ValueError("compile_clean_sections requires at least one (session_id, take_id).")
        out_dir = ensure_path_allowed(output_dir, self.settings)
        sess_root_res = self.sessions_root.resolve()
        if out_dir.resolve() == sess_root_res or sess_root_res in out_dir.resolve().parents:
            raise ValueError("output_dir cannot reside inside internal sessions storage.")
        fmt = audio_format.strip().lower()
        if fmt not in {"wav", "flac", "mp3", "ogg"}:
            raise ValueError(
                f"Unsupported export format '{audio_format}'. Use wav, flac, mp3, or ogg."
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        clean_dir = out_dir / "clean_sections"
        clean_dir.mkdir(parents=True, exist_ok=True)

        seg_inputs: list[tuple[str | Path, float]] = []
        seg_meta: list[dict[str, Any]] = []
        protected: set[Path] = set()
        analysis_segments: list[dict[str, Any]] = []
        global_edits: list[tuple[float, float]] = []
        global_distractions: list[tuple[float, float, str, str]] = []
        seg_boundaries: list[tuple[float, str, str]] = []
        cum_t = 0.0

        for idx, (sid, tid) in enumerate(sequence):
            summary = self.get_session(sid)
            take = self.get_take(sid, tid)
            if summary.source_path:
                protected.add(Path(summary.source_path).resolve())
            for st_t in summary.takes:
                if st_t.take_id == "take_000":
                    protected.add(Path(st_t.audio_path).resolve())
            protected.add(Path(take.audio_path).resolve())

            dur_s = float(take.metrics.duration_s)
            seg_inputs.append((take.audio_path, dur_s))
            src_name = Path(summary.source_path).name if summary.source_path else summary.name
            src_stem = (
                Path(summary.source_path).stem
                if summary.source_path
                else _slugify(summary.name)
            )
            events = self.load_take_events(sid, tid)
            edit_spans = extract_localized_edit_spans(take.recipe)
            distraction_spans = extract_localized_distraction_spans(
                events, include_low_severity=include_low_severity
            )
            seg_meta.append(
                {
                    "session_id": sid,
                    "take_id": tid,
                    "stem": src_stem,
                    "name": summary.name,
                    "source_file": src_name,
                }
            )
            analysis_segments.append(
                {
                    "label": src_name,
                    "session_id": sid,
                    "take_id": tid,
                    "duration_s": dur_s,
                    "edit_spans": edit_spans,
                    "distraction_spans": distraction_spans,
                }
            )
            for es, ee in edit_spans:
                cs = max(0.0, min(dur_s, float(es)))
                ce = max(0.0, min(dur_s, float(ee)))
                if ce > cs:
                    global_edits.append((cum_t + cs, cum_t + ce))
            for ev in events:
                if (
                    ev.kind in _LOCALIZED_DISTRACTION_KINDS
                    and float(ev.end_s) > float(ev.start_s)
                    and (include_low_severity or ev.severity != "low")
                ):
                    cs = max(0.0, min(dur_s, float(ev.start_s)))
                    ce = max(0.0, min(dur_s, float(ev.end_s)))
                    if ce > cs:
                        global_distractions.append(
                            (cum_t + cs, cum_t + ce, ev.kind, ev.severity)
                        )
            if idx > 0:
                seg_boundaries.append((cum_t, seg_meta[idx - 1]["source_file"], src_name))
            cum_t += dur_s

        explicit_prefix = _slugify(filename_prefix) if filename_prefix.strip() else ""
        sec_prefix = explicit_prefix or _slugify(seg_meta[0]["name"])

        clean_report = compute_clean_sections(
            analysis_segments,
            max_edits_per_hour=max_edits_per_hour,
            max_edited_pct=max_edited_pct,
            round_to_file_local_minutes=round_to_file_local_minutes,
        )
        asmr_targets: list[dict[str, Any]] = list(clean_report.get("asmr_targets") or [])

        def _parse_min(raw_t: str) -> float | None:
            s = str(raw_t).strip().lower()
            if s.endswith("min"):
                s = s[:-3].strip()
                return float(s) if s else None
            if s.endswith("m"):
                s = s[:-1].strip()
                return float(s) if s else None
            if s.endswith("hr"):
                s = s[:-2].strip()
                return float(s) * 60.0 if s else None
            if s.endswith("h"):
                s = s[:-1].strip()
                return float(s) * 60.0 if s else None
            try:
                return float(s)
            except ValueError:
                return None

        tier_filter = {str(t).strip().lower() for t in tiers if str(t).strip()} if tiers else None
        chosen: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []

        def _match_target(
            cand_list: list[dict[str, Any]],
            tok: str,
            rep_obj: dict[str, Any],
        ) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
            if tok.lower() == "largest":
                out_c: list[dict[str, Any]] = []
                for key_name in (
                    "largest_untouched_asmr_target",
                    "largest_minimal_edit_asmr_target",
                ):
                    c = rep_obj.get(key_name)
                    if not c:
                        continue
                    if tier_filter and c.get("tier", "").lower() not in tier_filter:
                        continue
                    copy_c = dict(c)
                    copy_c["_requested_target"] = "largest"
                    out_c.append(copy_c)
                if not out_c:
                    return [], {
                        "target": tok,
                        "reason": "No eligible target matched 'largest' criteria.",
                    }
                return out_c, None
            by_min: dict[int, dict[str, Any]] = {int(t["duration_min"]): t for t in cand_list}
            mins = _parse_min(tok)
            cand = (
                by_min.get(int(round(mins)))
                if mins is not None
                else next(
                    (
                        t
                        for t in cand_list
                        if str(t.get("duration_label", "")).lower() == tok.lower()
                    ),
                    None,
                )
            )
            if cand is None:
                return [], {
                    "target": tok,
                    "reason": "Duration unavailable or overlaps previously selected regions.",
                }
            if tier_filter and cand.get("tier", "").lower() not in tier_filter:
                return [], {
                    "target": tok,
                    "reason": f"Tier '{cand.get('tier')}' excluded by tiers filter.",
                }
            copy_c = dict(cand)
            copy_c["_requested_target"] = tok
            return [copy_c], None

        if distinct_regions:
            used_spans: list[tuple[float, float]] = []
            seen_durs: set[int] = set()
            if targets:
                parsed_reqs: list[tuple[int, str, float]] = []
                for req_i, raw_t in enumerate(targets):
                    tok = str(raw_t).strip()
                    if not tok:
                        continue
                    m_val = _parse_min(tok)
                    sort_k = float("inf") if tok.lower() == "largest" else (m_val or 0.0)
                    parsed_reqs.append((req_i, tok, sort_k))
                parsed_reqs.sort(key=lambda item: (-item[2], item[0]))
                selected_by_req: list[tuple[int, dict[str, Any]]] = []
                for req_i, tok, _ in parsed_reqs:
                    if tok.lower() == "largest":
                        sub_selected: list[dict[str, Any]] = []
                        for key_name in (
                            "largest_minimal_edit_asmr_target",
                            "largest_untouched_asmr_target",
                        ):
                            sub_rep = (
                                compute_clean_sections(
                                    analysis_segments,
                                    max_edits_per_hour=max_edits_per_hour,
                                    max_edited_pct=max_edited_pct,
                                    round_to_file_local_minutes=round_to_file_local_minutes,
                                    exclude_spans=used_spans,
                                )
                                if used_spans
                                else clean_report
                            )
                            c = sub_rep.get(key_name)
                            if c and (
                                not tier_filter or c.get("tier", "").lower() in tier_filter
                            ):
                                d_m = int(c["duration_min"])
                                if d_m not in seen_durs:
                                    copy_c = dict(c)
                                    copy_c["_requested_target"] = "largest"
                                    sub_selected.append(copy_c)
                                    seen_durs.add(d_m)
                                    used_spans.append((float(c["start_s"]), float(c["end_s"])))
                        if not sub_selected:
                            skipped.append(
                                {
                                    "target": tok,
                                    "reason": "No eligible target matched 'largest' criteria.",
                                }
                            )
                        else:
                            for sc in sub_selected:
                                selected_by_req.append((req_i, sc))
                        continue
                    rep_i = (
                        compute_clean_sections(
                            analysis_segments,
                            max_edits_per_hour=max_edits_per_hour,
                            max_edited_pct=max_edited_pct,
                            round_to_file_local_minutes=round_to_file_local_minutes,
                            exclude_spans=used_spans,
                        )
                        if used_spans
                        else clean_report
                    )
                    matched, skip_info = _match_target(
                        list(rep_i.get("asmr_targets") or []), tok, rep_i
                    )
                    if skip_info is not None:
                        skipped.append(skip_info)
                        continue
                    for cand in matched:
                        d_m = int(cand["duration_min"])
                        if d_m in seen_durs:
                            continue
                        selected_by_req.append((req_i, cand))
                        seen_durs.add(d_m)
                        used_spans.append((float(cand["start_s"]), float(cand["end_s"])))
                selected_by_req.sort(key=lambda item: item[0])
                chosen = [c for _, c in selected_by_req]
            else:
                allowed_tiers = tier_filter or {"untouched", "minimal_edits"}
                for cand in reversed(asmr_targets):
                    d_m = int(cand["duration_min"])
                    rep_i = (
                        compute_clean_sections(
                            analysis_segments,
                            max_edits_per_hour=max_edits_per_hour,
                            max_edited_pct=max_edited_pct,
                            round_to_file_local_minutes=round_to_file_local_minutes,
                            exclude_spans=used_spans,
                        )
                        if used_spans
                        else clean_report
                    )
                    match = next(
                        (
                            t
                            for t in (rep_i.get("asmr_targets") or [])
                            if int(t["duration_min"]) == d_m
                        ),
                        None,
                    )
                    if match is None or str(match.get("tier", "")).lower() not in allowed_tiers:
                        skipped.append(
                            {
                                "target": str(cand.get("duration_label", "")),
                                "reason": "No distinct stretch meeting tier thresholds.",
                            }
                        )
                        continue
                    copy_c = dict(match)
                    copy_c["_requested_target"] = str(copy_c.get("duration_label", ""))
                    chosen.append(copy_c)
                    used_spans.append((float(match["start_s"]), float(match["end_s"])))
                chosen.reverse()
        elif targets:
            seen_durs = set()
            for raw_t in targets:
                tok = str(raw_t).strip()
                if not tok:
                    continue
                matched, skip_info = _match_target(asmr_targets, tok, clean_report)
                if skip_info is not None:
                    skipped.append(skip_info)
                    continue
                for cand in matched:
                    d_m = int(cand["duration_min"])
                    if d_m not in seen_durs:
                        chosen.append(cand)
                        seen_durs.add(d_m)
        else:
            allowed_tiers = tier_filter or {"untouched", "minimal_edits"}
            for cand in asmr_targets:
                if str(cand.get("tier", "")).lower() in allowed_tiers:
                    copy_c = dict(cand)
                    copy_c["_requested_target"] = str(cand.get("duration_label", ""))
                    chosen.append(copy_c)
                else:
                    skipped.append(
                        {
                            "target": str(cand.get("duration_label", "")),
                            "reason": f"Tier '{cand.get('tier')}' outside {sorted(allowed_tiers)}.",
                        }
                    )

        if max_files is not None and int(max_files) > 0:
            chosen = chosen[: int(max_files)]

        total_jobs = (
            (len(sequence) if include_full_masters else 0)
            + (1 if include_combined_full else 0)
            + len(chosen)
        )
        done_jobs = 0

        def _emit_job_progress(
            idx: int,
            sid_lbl: str,
            detail: str,
            completed_count: int,
            *,
            item_pct: float | None = None,
            phase: str = "",
            b_written: int = 0,
            b_total: int = 0,
            meta: dict[str, Any] | None = None,
        ) -> None:
            if progress_callback is None:
                return
            progress_callback(
                idx,
                total_jobs,
                sid_lbl,
                detail,
                completed_count,
                meta.get("take_id") if meta else None,
                item_progress_pct=item_pct,
                phase=phase,
                bytes_written=b_written,
                bytes_total=b_total,
                item_meta=meta,
            )

        def _compile_with_metrics(
            item_idx: int,
            item_label: str,
            item_kind: str,
            start_s: float,
            end_s: float,
            dst_p: Path,
            f_in: float,
            f_out: float,
            est_bytes: int,
        ) -> dict[str, Any]:
            if dst_p.resolve() in protected:
                raise ValueError(f"Refusing to overwrite protected audio path: {dst_p}")
            sr0 = probe_audio(seg_inputs[0][0]).sample_rate
            ch0 = probe_audio(seg_inputs[0][0]).channels
            lufs_acc = StreamingLUFS(sr0, ch0)
            tp_acc = StreamingTruePeak(ch0)
            last_hb = [0.0]

            def _on_blk(blk: np.ndarray) -> None:
                lufs_acc.update(blk)
                tp_acc.update(blk)

            def _on_prog(written_f: int, total_f: int) -> None:
                now_t = time.monotonic()
                if now_t - last_hb[0] < 2.0 and written_f < total_f:
                    return
                last_hb[0] = now_t
                pct = (written_f / float(max(total_f, 1))) * 100.0
                bw = int(round((written_f / float(max(total_f, 1))) * est_bytes))
                _emit_job_progress(
                    item_idx,
                    item_label,
                    f"compile {dst_p.name} ({pct:.0f}%)",
                    done_jobs,
                    item_pct=pct,
                    phase="write_and_measure",
                    b_written=bw,
                    b_total=est_bytes,
                    meta={
                        "label": item_label,
                        "kind": item_kind,
                        "state": "running",
                        "detail": f"compiling {dst_p.name}",
                        "session_id": None,
                        "take_id": None,
                        "output_path": str(dst_p),
                        "size_bytes": bw,
                    },
                )

            try:
                c_info = compile_audio_span(
                    seg_inputs,
                    start_s,
                    end_s,
                    dst_p,
                    audio_format=fmt,
                    subtype="PCM_24",
                    fade_in_s=f_in,
                    fade_out_s=f_out,
                    on_block=_on_blk,
                    on_progress=_on_prog,
                    cancel_check=cancel_check,
                )
            except CompilationCancelledError as exc:
                raise SequenceCancelledError(str(exc)) from exc

            for sl in c_info.get("source_slices") or []:
                s_idx = int(sl.get("segment_index", 0))
                if 0 <= s_idx < len(seg_meta):
                    meta = seg_meta[s_idx]
                    sl["file"] = meta["source_file"]
                    sl["source_file"] = meta["source_file"]
                    sl["session_id"] = meta["session_id"]
                    sl["take_id"] = meta["take_id"]

            c_info["integrated_lufs"] = _finite_or_none(lufs_acc.finalize(), digits=3)
            c_info["true_peak_dbtp"] = _finite_or_none(
                20.0 * math.log10(max(tp_acc.peak_lin, 1e-12)), digits=3
            )
            c_info["peak_measured_on"] = "output"
            return c_info

        full_masters: list[dict[str, Any]] = []
        if include_full_masters:
            used_names: set[str] = set()
            for idx, (sid, tid) in enumerate(sequence):
                if cancel_check and cancel_check():
                    raise SequenceCancelledError("Clean-section compilation job cancelled.")
                meta = seg_meta[idx]
                stem = meta["stem"]
                if stem in used_names:
                    stem = f"{stem}_{idx + 1}"
                used_names.add(stem)
                master_name = (
                    f"{explicit_prefix}_{stem}_master.{fmt}"
                    if explicit_prefix
                    else f"{stem}_master.{fmt}"
                )
                dst_m = out_dir / master_name
                item_idx = done_jobs
                src_size = (
                    Path(seg_inputs[idx][0]).stat().st_size
                    if Path(seg_inputs[idx][0]).exists()
                    else 0
                )
                last_m_hb = [0.0]

                def _on_master_prog(
                    done_u: int,
                    tot_u: int,
                    *,
                    _i: int = item_idx,
                    _sid: str = sid,
                    _tid: str = tid,
                    _dst: Path = dst_m,
                    _hb: list[float] = last_m_hb,
                    _sz: int = src_size,
                    _done: int = done_jobs,
                ) -> None:
                    now_t = time.monotonic()
                    if now_t - _hb[0] < 2.0 and done_u < tot_u:
                        return
                    _hb[0] = now_t
                    pct = (done_u / float(max(tot_u, 1))) * 100.0
                    bw = int(round((done_u / float(max(tot_u, 1))) * _sz)) if _sz else int(done_u)
                    bt = _sz or int(tot_u)
                    _emit_job_progress(
                        _i,
                        _sid,
                        f"export {_dst.name} ({pct:.0f}%)",
                        _done,
                        item_pct=pct,
                        phase="copy_and_hash",
                        b_written=bw,
                        b_total=bt,
                        meta={
                            "label": _sid,
                            "kind": "master",
                            "state": "running",
                            "detail": f"exporting {_dst.name}",
                            "session_id": _sid,
                            "take_id": _tid,
                            "output_path": str(_dst),
                            "size_bytes": bw,
                        },
                    )

                _emit_job_progress(
                    item_idx,
                    sid,
                    f"export {dst_m.name}",
                    done_jobs,
                    item_pct=0.0,
                    phase="copy_and_hash",
                    b_written=0,
                    b_total=src_size,
                    meta={
                        "label": sid,
                        "kind": "master",
                        "state": "running",
                        "detail": f"exporting {dst_m.name}",
                        "session_id": sid,
                        "take_id": tid,
                        "output_path": str(dst_m),
                        "size_bytes": 0,
                    },
                )
                exp_info = self.export_take_detailed(
                    sid,
                    tid,
                    dst_m,
                    audio_format=fmt,
                    on_progress=_on_master_prog,
                    cancel_check=cancel_check,
                )
                exp_info["prefix_applied"] = bool(explicit_prefix)
                full_masters.append(exp_info)
                done_jobs += 1
                _emit_job_progress(
                    item_idx,
                    sid,
                    f"exported {dst_m.name}",
                    done_jobs,
                    item_pct=100.0,
                    phase="done",
                    b_written=int(exp_info.get("size_bytes") or 0),
                    b_total=int(exp_info.get("size_bytes") or 0),
                    meta={
                        "label": sid,
                        "kind": "master",
                        "state": "done",
                        "detail": f"exported {dst_m.name}",
                        "session_id": sid,
                        "take_id": tid,
                        "output_path": str(dst_m),
                        "size_bytes": int(exp_info.get("size_bytes") or 0),
                    },
                )

        sr_ref = probe_audio(seg_inputs[0][0]).sample_rate
        ch_ref = probe_audio(seg_inputs[0][0]).channels
        bps_est = 3 if fmt == "wav" else 2

        combined_full: dict[str, Any] | None = None
        if include_combined_full:
            if cancel_check and cancel_check():
                raise SequenceCancelledError("Clean-section compilation job cancelled.")
            total_seq_s = sum(d for _, d in seg_inputs)
            dst_c = out_dir / f"{sec_prefix}_full_combined.{fmt}"
            item_idx = done_jobs
            est_c = int(round(total_seq_s * sr_ref * ch_ref * bps_est))
            _emit_job_progress(
                item_idx,
                sec_prefix,
                f"compile {dst_c.name}",
                done_jobs,
                item_pct=0.0,
                phase="write_and_measure",
                b_written=0,
                b_total=est_c,
                meta={
                    "label": dst_c.name,
                    "kind": "combined",
                    "state": "running",
                    "detail": f"compiling {dst_c.name}",
                    "session_id": None,
                    "take_id": None,
                    "output_path": str(dst_c),
                    "size_bytes": 0,
                },
            )
            combined_full = _compile_with_metrics(
                item_idx, dst_c.name, "combined", 0.0, total_seq_s, dst_c, 0.0, 0.0, est_c
            )
            done_jobs += 1
            _emit_job_progress(
                item_idx,
                sec_prefix,
                f"compiled {dst_c.name}",
                done_jobs,
                item_pct=100.0,
                phase="done",
                b_written=int(combined_full.get("size_bytes") or 0),
                b_total=int(combined_full.get("size_bytes") or 0),
                meta={
                    "label": dst_c.name,
                    "kind": "combined",
                    "state": "done",
                    "detail": f"compiled {dst_c.name}",
                    "session_id": None,
                    "take_id": None,
                    "output_path": str(dst_c),
                    "size_bytes": int(combined_full.get("size_bytes") or 0),
                },
            )

        compiled_sections: list[dict[str, Any]] = []
        for cand in chosen:
            if cancel_check and cancel_check():
                raise SequenceCancelledError("Clean-section compilation job cancelled.")
            dur_slug = str(cand["duration_label"]).replace(" ", "").replace(".", "p")
            tier_slug = str(cand["tier"]).replace(" ", "_")
            dst_s = clean_dir / f"{sec_prefix}_clean_{dur_slug}_{tier_slug}.{fmt}"
            c_start = float(cand["start_s"])
            c_end = float(cand["end_s"])
            item_idx = done_jobs
            est_s = int(round((c_end - c_start) * sr_ref * ch_ref * bps_est))
            _emit_job_progress(
                item_idx,
                dur_slug,
                f"compile {dst_s.name}",
                done_jobs,
                item_pct=0.0,
                phase="write_and_measure",
                b_written=0,
                b_total=est_s,
                meta={
                    "label": dur_slug,
                    "kind": "section",
                    "state": "running",
                    "detail": f"compiling {dst_s.name}",
                    "session_id": None,
                    "take_id": None,
                    "output_path": str(dst_s),
                    "size_bytes": 0,
                },
            )
            c_info = _compile_with_metrics(
                item_idx,
                dur_slug,
                "section",
                c_start,
                c_end,
                dst_s,
                fade_in_s,
                fade_out_s,
                est_s,
            )

            sec_seams = [
                {
                    "t_in_file_s": round(b_t - c_start, 3),
                    "from_file": f_a,
                    "to_file": f_b,
                }
                for b_t, f_a, f_b in seg_boundaries
                if c_start + 1e-6 < b_t < c_end - 1e-6
            ]
            sec_events: list[dict[str, Any]] = []
            for es, ee in global_edits:
                if ee > c_start + 1e-6 and es < c_end - 1e-6:
                    sec_events.append(
                        {
                            "t_in_file_s": round(max(0.0, es - c_start), 3),
                            "end_in_file_s": round(min(c_end - c_start, ee - c_start), 3),
                            "kind": "localized_edit",
                            "severity": "edit",
                            "action": "repaired",
                        }
                    )
            for ds, de, kind, sev in global_distractions:
                if de > c_start + 1e-6 and ds < c_end - 1e-6:
                    sec_events.append(
                        {
                            "t_in_file_s": round(max(0.0, ds - c_start), 3),
                            "end_in_file_s": round(min(c_end - c_start, de - c_start), 3),
                            "kind": kind,
                            "severity": sev,
                            "action": "residual",
                        }
                    )
            sec_events.sort(key=lambda e: (e["t_in_file_s"], e["action"]))

            req_tok = str(cand.get("_requested_target") or cand.get("duration_label", ""))
            sel_reason = (
                f"Cleanest eligible stretch matching '{req_tok}' "
                f"({cand['tier']}, {cand['edited_pct']}% edited, "
                f"{cand['edit_count']} edits)."
            )
            entry = {
                "file_name": dst_s.name,
                "duration_label": cand["duration_label"],
                "duration_min": cand["duration_min"],
                "tier": cand["tier"],
                "eligible": cand.get("eligible", cand["tier"] in {"untouched", "minimal_edits"}),
                "violations": list(cand.get("violations") or []),
                "requested_target": req_tok,
                "selection_reason": sel_reason,
                "start_s": cand["start_s"],
                "end_s": cand["end_s"],
                "start_hms": cand["start_hms"],
                "end_hms": cand["end_hms"],
                "start_file": cand["start_file"],
                "start_local_s": cand.get("start_local_s"),
                "start_local_hms": cand["start_local_hms"],
                "end_file": cand["end_file"],
                "end_local_s": cand.get("end_local_s"),
                "end_local_hms": cand["end_local_hms"],
                "fade_in_s": float(fade_in_s),
                "fade_out_s": float(fade_out_s),
                "edit_count": cand["edit_count"],
                "raw_edit_span_count": cand.get("raw_edit_span_count"),
                "distraction_count": cand["distraction_count"],
                "residual_distraction_count": cand.get(
                    "residual_distraction_count", cand["distraction_count"]
                ),
                "edited_s": cand.get("edited_s"),
                "edited_pct": cand["edited_pct"],
                "residual_distraction_s": cand.get("residual_distraction_s"),
                "residual_distraction_pct": cand.get("residual_distraction_pct"),
                "spans_multiple_files": len(c_info["source_slices"]) > 1,
                "seams": sec_seams,
                "events": sec_events,
                **c_info,
            }
            compiled_sections.append(entry)
            done_jobs += 1
            _emit_job_progress(
                item_idx,
                dur_slug,
                f"compiled {dst_s.name}",
                done_jobs,
                item_pct=100.0,
                phase="done",
                b_written=int(entry.get("size_bytes") or 0),
                b_total=int(entry.get("size_bytes") or 0),
                meta={
                    "label": dur_slug,
                    "kind": "section",
                    "state": "done",
                    "detail": f"compiled {dst_s.name}",
                    "session_id": None,
                    "take_id": None,
                    "output_path": str(dst_s),
                    "size_bytes": int(entry.get("size_bytes") or 0),
                },
            )

        total_bytes = (
            sum(int(m.get("size_bytes") or 0) for m in full_masters)
            + (int(combined_full.get("size_bytes") or 0) if combined_full else 0)
            + sum(int(s.get("size_bytes") or 0) for s in compiled_sections)
        )
        warnings: list[str] = []
        rf64_files = [
            s["file_name"] for s in compiled_sections if s.get("container") == "RF64"
        ]
        if combined_full and combined_full.get("container") == "RF64":
            rf64_files.append(Path(combined_full["output_path"]).name)
        rf64_files.extend(
            Path(m["output_path"]).name for m in full_masters if m.get("container") == "RF64"
        )
        if rf64_files:
            rf64_list = ", ".join(rf64_files)
            warnings.append(
                f"RF64 container auto-selected for >=3.8 GB WAV output(s): {rf64_list}. "
                "Use format='flac' for smaller lossless files with universal player support."
            )

        manifest_path = out_dir / "clean_sections_manifest.json"
        payload = {
            "tool_version": __version__,
            "created_at": _utc_now_iso(),
            "output_dir": str(out_dir),
            "clean_sections_dir": str(clean_dir),
            "manifest_path": str(manifest_path),
            "format": fmt,
            "parameters": {
                "targets": targets,
                "tiers": tiers,
                "include_full_masters": include_full_masters,
                "include_combined_full": include_combined_full,
                "filename_prefix": filename_prefix,
                "format": fmt,
                "fade_in_s": float(fade_in_s),
                "fade_out_s": float(fade_out_s),
                "max_edits_per_hour": float(max_edits_per_hour),
                "max_edited_pct": float(max_edited_pct),
                "include_low_severity": bool(include_low_severity),
                "round_to_file_local_minutes": bool(round_to_file_local_minutes),
                "distinct_regions": bool(distinct_regions),
                "max_files": max_files,
            },
            "sequence": [
                {
                    "session_id": m["session_id"],
                    "take_id": m["take_id"],
                    "source_file": m["source_file"],
                }
                for m in seg_meta
            ],
            "masters_count": len(full_masters),
            "compiled_count": len(compiled_sections),
            "total_bytes": int(total_bytes),
            "warnings": warnings,
            "full_masters": full_masters,
            "combined_full": combined_full,
            "compiled_sections": compiled_sections,
            "skipped_targets": skipped,
            "clean_sections_report": clean_report,
        }
        tmp_m = manifest_path.with_name(f".manifest.{uuid4().hex[:6]}.tmp")
        tmp_m.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(tmp_m, manifest_path)
        return payload

    def verify_seams(
        self,
        sequence: list[tuple[str, str]],
        window_s: float = 10.0,
        max_sample_step_excess_db: float = 18.0,
    ) -> dict[str, Any]:
        """Measure LUFS, 2 s RMS, sample step, and 40-8000 Hz spectral continuity across joins."""
        if len(sequence) < 2:
            raise ValueError(
                "verify_seams requires at least two consecutive (session_id, take_id) pairs."
            )
        win_s = max(1.0, min(30.0, float(window_s)))
        seams: list[dict[str, Any]] = []

        for idx in range(len(sequence) - 1):
            sid_a, tid_a = sequence[idx]
            sid_b, tid_b = sequence[idx + 1]
            sum_a = self.get_session(sid_a)
            sum_b = self.get_session(sid_b)
            take_a = self.get_take(sid_a, tid_a)
            take_b = self.get_take(sid_b, tid_b)
            info_a = probe_audio(take_a.audio_path)
            info_b = probe_audio(take_b.audio_path)
            if info_a.sample_rate != info_b.sample_rate or info_a.channels != info_b.channels:
                raise ValueError(
                    f"Format mismatch at seam {sid_a}/{tid_a} ({info_a.sample_rate}Hz/"
                    f"{info_a.channels}ch) -> {sid_b}/{tid_b} ({info_b.sample_rate}Hz/"
                    f"{info_b.channels}ch)."
                )
            sr = info_a.sample_rate
            ch = info_a.channels
            end_arr, _ = load_audio_slice(
                take_a.audio_path,
                start_s=max(0.0, info_a.duration_s - win_s),
                end_s=info_a.duration_s,
                dtype="float64",
            )
            start_arr, _ = load_audio_slice(
                take_b.audio_path,
                start_s=0.0,
                end_s=min(win_s, info_b.duration_s),
                dtype="float64",
            )

            lufs_end: float | None = None
            lufs_start: float | None = None
            if end_arr.shape[0] >= int(0.5 * sr):
                acc_e = StreamingLUFS(sr, ch)
                acc_e.update(end_arr)
                lufs_end = acc_e.finalize()
            if start_arr.shape[0] >= int(0.5 * sr):
                acc_s = StreamingLUFS(sr, ch)
                acc_s.update(start_arr)
                lufs_start = acc_s.finalize()
            lufs_delta = (
                _finite_or_none(abs(lufs_start - lufs_end), digits=3)
                if lufs_end is not None and lufs_start is not None
                else None
            )

            n_2s = min(int(2.0 * sr), end_arr.shape[0], start_arr.shape[0])
            rms_delta_2s: float | None = None
            rms_e_db: float | None = None
            rms_s_db: float | None = None
            if n_2s >= 64:
                re_lin = float(np.sqrt(np.mean(end_arr[-n_2s:] ** 2)))
                rs_lin = float(np.sqrt(np.mean(start_arr[:n_2s] ** 2)))
                if re_lin > 1e-12 and rs_lin > 1e-12:
                    rms_e_db = _finite_or_none(20.0 * math.log10(re_lin), digits=3)
                    rms_s_db = _finite_or_none(20.0 * math.log10(rs_lin), digits=3)
                    if rms_e_db is not None and rms_s_db is not None:
                        rms_delta_2s = _finite_or_none(abs(rms_s_db - rms_e_db), digits=3)

            sample_step_dbfs: float | None = None
            sample_step_excess_db: float | None = None
            if end_arr.shape[0] > 0 and start_arr.shape[0] > 0:
                step_lin = float(np.max(np.abs(start_arr[0] - end_arr[-1])))
                sample_step_dbfs = _finite_or_none(
                    20.0 * math.log10(max(step_lin, 1e-6)), digits=2
                )
                d_parts = []
                if end_arr.shape[0] >= 8:
                    n_e = min(end_arr.shape[0], int(2.0 * sr))
                    d_parts.append(np.diff(end_arr[-n_e:], axis=0))
                if start_arr.shape[0] >= 8:
                    n_s = min(start_arr.shape[0], int(2.0 * sr))
                    d_parts.append(np.diff(start_arr[:n_s], axis=0))
                if d_parts:
                    cat_d = np.vstack(d_parts)
                    local_diff_rms = float(np.sqrt(np.mean(cat_d * cat_d)))
                    if local_diff_rms > 1e-12:
                        sample_step_excess_db = _finite_or_none(
                            20.0 * math.log10(max(step_lin, 1e-12) / local_diff_rms),
                            digits=2,
                        )

            spectral_diff_db: float | None = None
            mono_e = np.mean(end_arr, axis=1) if end_arr.size else np.zeros(0)
            mono_s = np.mean(start_arr, axis=1) if start_arr.size else np.zeros(0)
            nperseg = min(4096, mono_e.size, mono_s.size)
            if nperseg >= 64:
                freqs, p_e = signal.welch(mono_e, fs=sr, nperseg=nperseg)
                _, p_s = signal.welch(mono_s, fs=sr, nperseg=nperseg)
                f_mask = (freqs >= 40.0) & (freqs <= min(8000.0, sr * 0.45))
                if np.any(f_mask):
                    pe_db = 10.0 * np.log10(np.maximum(p_e[f_mask], 1e-24))
                    ps_db = 10.0 * np.log10(np.maximum(p_s[f_mask], 1e-24))
                    spectral_diff_db = _finite_or_none(float(np.mean(np.abs(ps_db - pe_db))), 3)

            lbl_a = Path(sum_a.source_path).name if sum_a.source_path else sum_a.name
            lbl_b = Path(sum_b.source_path).name if sum_b.source_path else sum_b.name
            seam_png = (
                self._session_dir(sid_b)
                / "seams"
                / f"seam_{idx:02d}_{sid_a}_{tid_a}_to_{sid_b}_{tid_b}.png"
            )
            render_seam_spectrogram_png(
                take_a.audio_path,
                take_b.audio_path,
                seam_png,
                window_s=win_s,
                title=f"Seam {lbl_a} ({tid_a}) -> {lbl_b} ({tid_b})",
            )

            step_ok = (
                sample_step_dbfs is None
                or sample_step_dbfs <= -30.0
                or (
                    sample_step_dbfs <= -6.0
                    and (
                        sample_step_excess_db is None
                        or sample_step_excess_db <= max_sample_step_excess_db
                    )
                )
            )
            passed = bool(
                (lufs_delta is None or lufs_delta <= 1.5)
                and (rms_delta_2s is None or rms_delta_2s <= 2.0)
                and (spectral_diff_db is None or spectral_diff_db <= 3.0)
                and step_ok
            )
            seams.append(
                {
                    "pair_index": idx,
                    "prev": {"session_id": sid_a, "take_id": tid_a, "label": lbl_a},
                    "next": {"session_id": sid_b, "take_id": tid_b, "label": lbl_b},
                    "window_s": win_s,
                    "prev_end_lufs": lufs_end,
                    "next_start_lufs": lufs_start,
                    "lufs_delta_db": lufs_delta,
                    "prev_rms_2s_dbfs": rms_e_db,
                    "next_rms_2s_dbfs": rms_s_db,
                    "rms_delta_2s_db": rms_delta_2s,
                    "sample_step_dbfs": sample_step_dbfs,
                    "sample_step_excess_db": sample_step_excess_db,
                    "spectral_diff_db": spectral_diff_db,
                    "passed": passed,
                    "spectrogram_path": str(seam_png),
                }
            )

        return {
            "pair_count": len(seams),
            "all_passed": all(s["passed"] for s in seams),
            "thresholds": {
                "max_lufs_delta_db": 1.5,
                "max_rms_delta_2s_db": 2.0,
                "max_spectral_diff_db": 3.0,
                "max_sample_step_dbfs": -30.0,
                "max_sample_step_excess_db": float(max_sample_step_excess_db),
            },
            "seams": seams,
        }

    def calibrate_sequence(
        self,
        sequence: list[tuple[str, str]],
        shared_steps_before_gain: list[FilterStep] | list[dict[str, Any]],
        target_lufs: float = -14.5,
        ceiling_dbtp: float = -1.0,
        sample_windows_per_file: int = 5,
        window_s: float = 20.0,
        shared_steps_after_gain: list[FilterStep] | list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Solve one shared master gain across a continuous split sequence on clean slices."""
        if not sequence:
            raise ValueError("calibrate_sequence requires at least one (session_id, take_id) pair.")
        before_steps = self._validate_step_paths(
            [
                s if isinstance(s, FilterStep) else FilterStep.model_validate(s)
                for s in shared_steps_before_gain
            ]
        )
        after_steps = self._validate_step_paths(
            [
                s if isinstance(s, FilterStep) else FilterStep.model_validate(s)
                for s in (shared_steps_after_gain or [])
            ]
        )
        k_win = max(1, min(12, int(sample_windows_per_file)))
        win_dur = max(2.0, min(45.0, float(window_s)))

        pre_slices: list[tuple[int, int, np.ndarray]] = []
        file_meta: list[dict[str, Any]] = []
        seq_acc: StreamingLUFS | None = None
        expected_sr: int | None = None
        expected_ch: int | None = None

        for f_idx, (sid, tid) in enumerate(sequence):
            summary = self.get_session(sid)
            take = self.get_take(sid, tid)
            sr_f = take.metrics.sample_rate
            ch_f = take.metrics.channels
            if expected_sr is None:
                expected_sr, expected_ch = sr_f, ch_f
            elif sr_f != expected_sr or ch_f != expected_ch:
                raise ValueError(
                    f"Format mismatch in calibrate_sequence for {sid}/{tid} "
                    f"({sr_f}Hz/{ch_f}ch vs {expected_sr}Hz/{expected_ch}ch)."
                )
            events = self.load_take_events(sid, tid)
            dur = take.metrics.duration_s
            ex_ev = _exclude_events_for_take(events, dur)
            windows = _pick_clean_interior_windows(dur, win_dur, k_win, ex_ev)
            f_acc: StreamingLUFS | None = None

            for w0, w1 in windows:
                arr, sr = load_audio_slice(take.audio_path, start_s=w0, end_s=w1, dtype="float64")
                if arr.shape[0] < int(0.5 * sr):
                    continue
                pre = apply_recipe(arr, sr, before_steps) if before_steps else arr
                if seq_acc is None:
                    seq_acc = StreamingLUFS(sr, ch_f)
                if f_acc is None:
                    f_acc = StreamingLUFS(sr, ch_f)
                seq_acc.update(pre)
                f_acc.update(pre)
                pre_slices.append((f_idx, sr, pre.astype(np.float32)))
                sr_f = sr

            base_lufs = f_acc.finalize() if f_acc is not None else None
            lbl = Path(summary.source_path).name if summary.source_path else summary.name
            file_meta.append(
                {
                    "session_id": sid,
                    "take_id": tid,
                    "label": lbl,
                    "windows_sampled": len(windows),
                    "baseline_lufs": base_lufs,
                }
            )

        seq_base_lufs = seq_acc.finalize() if seq_acc is not None else None
        if seq_base_lufs is None or not math.isfinite(seq_base_lufs):
            for meta in file_meta:
                meta["projected_lufs"] = None
            return {
                "target_lufs": round(float(target_lufs), 2),
                "ceiling_dbtp": round(float(ceiling_dbtp), 2),
                "sequence_baseline_lufs": None,
                "recommended_gain_db": 0.0,
                "projected_sequence_lufs": None,
                "projected_true_peak_dbtp": None,
                "files": file_meta,
            }

        gain_db = max(-48.0, min(48.0, float(target_lufs) - seq_base_lufs))

        def _evaluate_gain(g_val: float) -> tuple[float | None, float | None, list[float | None]]:
            g_lin = 10.0 ** (g_val / 20.0)
            g_acc: StreamingLUFS | None = None
            f_accs: dict[int, StreamingLUFS] = {}
            max_tp = 0.0
            for f_idx, sr, pre32 in pre_slices:
                scaled = pre32.astype(np.float64) * g_lin
                post = apply_recipe(scaled, sr, after_steps) if after_steps else scaled
                if g_acc is None:
                    g_acc = StreamingLUFS(sr, post.shape[1])
                if f_idx not in f_accs:
                    f_accs[f_idx] = StreamingLUFS(sr, post.shape[1])
                g_acc.update(post)
                f_accs[f_idx].update(post)
                max_tp = max(max_tp, chunked_true_peak_linear(post))
            proj_g = g_acc.finalize() if g_acc is not None else None
            tp_db = _finite_or_none(20.0 * math.log10(max(max_tp, 1e-12)), digits=3)
            per_f = [
                f_accs[idx].finalize() if idx in f_accs else None for idx in range(len(file_meta))
            ]
            return proj_g, tp_db, per_f

        proj_seq_lufs, proj_tp_db, per_f_lufs = _evaluate_gain(gain_db)
        if after_steps and proj_seq_lufs is not None and math.isfinite(proj_seq_lufs):
            for _ in range(2):
                deficit = float(target_lufs) - proj_seq_lufs
                if abs(deficit) <= 0.15:
                    break
                gain_db = max(-48.0, min(48.0, gain_db + deficit))
                proj_seq_lufs, proj_tp_db, per_f_lufs = _evaluate_gain(gain_db)
        elif (
            not after_steps
            and proj_tp_db is not None
            and proj_tp_db > float(ceiling_dbtp) + 0.05
        ):
            gain_db = max(-48.0, gain_db - (proj_tp_db - float(ceiling_dbtp)))
            proj_seq_lufs, proj_tp_db, per_f_lufs = _evaluate_gain(gain_db)

        for idx, meta in enumerate(file_meta):
            meta["projected_lufs"] = per_f_lufs[idx]
            meta.pop("sample_rate", None)
            meta.pop("channels", None)

        return {
            "target_lufs": round(float(target_lufs), 2),
            "ceiling_dbtp": round(float(ceiling_dbtp), 2),
            "sequence_baseline_lufs": _finite_or_none(seq_base_lufs, digits=3),
            "recommended_gain_db": _finite_or_none(gain_db, digits=3) or 0.0,
            "projected_sequence_lufs": _finite_or_none(proj_seq_lufs, digits=3),
            "projected_true_peak_dbtp": proj_tp_db,
            "files": file_meta,
        }

    def process_sequence(
        self,
        items: list[dict[str, Any]],
        shared_prefix_steps: list[FilterStep] | list[dict[str, Any]] | None = None,
        shared_suffix_steps: list[FilterStep] | list[dict[str, Any]] | None = None,
        outer_fade_in_s: float = 0.0,
        outer_fade_out_s: float = 0.0,
        seam_context_s: float = 8.0,
        verify_seams_flag: bool = True,
        clean_sections_flag: bool = True,
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
        progress_callback: Callable[..., None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Batch-process an ordered sequence with neighbor seam padding, export, and QC."""
        if not items:
            raise ValueError("process_sequence requires at least one item.")

        seq_out_dir: Path | None = None
        if output_dir is not None and str(output_dir).strip():
            seq_out_dir = ensure_path_allowed(output_dir, self.settings)
            sess_root_res = self.sessions_root.resolve()
            if (
                seq_out_dir.resolve() == sess_root_res
                or sess_root_res in seq_out_dir.resolve().parents
            ):
                raise ValueError("output_dir cannot reside inside internal sessions storage.")
            seq_out_dir.mkdir(parents=True, exist_ok=True)

        def _parse_and_validate(raw_list: list[Any]) -> list[FilterStep]:
            parsed = [
                s if isinstance(s, FilterStep) else FilterStep.model_validate(s) for s in raw_list
            ]
            return self._validate_step_paths(parsed)

        prefix = _parse_and_validate(list(shared_prefix_steps or []))
        suffix = _parse_and_validate(list(shared_suffix_steps or []))

        fmt_ext = export_format.strip().lower()
        explicit_prefix = _slugify(filename_prefix) if filename_prefix.strip() else ""
        used_master_names: set[str] = set()
        validated: list[dict[str, Any]] = []
        parent_paths: list[str] = []
        for idx, raw in enumerate(items):
            sid = str(raw.get("session_id", "")).strip()
            if not sid:
                raise ValueError(f"Sequence item {idx} missing 'session_id'.")
            ptid = str(raw.get("parent_take_id") or "take_000").strip()
            sum_i = self.get_session(sid)
            ptake = self.get_take(sid, ptid)
            exp_raw = raw.get("export_path")
            if exp_raw is not None and str(exp_raw).strip():
                exp_path = ensure_path_allowed(exp_raw, self.settings)
            elif seq_out_dir is not None:
                stem = (
                    Path(sum_i.source_path).stem
                    if sum_i.source_path
                    else _slugify(sum_i.name)
                )
                if stem in used_master_names:
                    stem = f"{stem}_{idx + 1}"
                used_master_names.add(stem)
                master_name = (
                    f"{explicit_prefix}_{stem}_master.{fmt_ext}"
                    if explicit_prefix
                    else f"{stem}_master.{fmt_ext}"
                )
                exp_path = seq_out_dir / master_name
            else:
                exp_path = None
            if discard_take_audio_after_export and exp_path is None:
                raise ValueError(
                    f"Sequence item {idx} ({sid}) must specify 'export_path' or 'output_dir' "
                    "when discard_take_audio_after_export=True."
                )
            local = _parse_and_validate(list(raw.get("local_steps") or []))
            validated.append(
                {
                    "session_id": sid,
                    "parent_take_id": ptid,
                    "local_steps": local,
                    "label": str(raw.get("label") or "").strip(),
                    "notes": str(raw.get("notes") or "").strip(),
                    "export_path": exp_path,
                }
            )
            parent_paths.append(ptake.audio_path)

        n_items = len(validated)
        results: list[dict[str, Any]] = []
        processed_pairs: list[tuple[str, str]] = []
        discard_candidates: list[tuple[str, str, Path]] = []

        for idx, item in enumerate(validated):
            if cancel_check and cancel_check():
                raise SequenceCancelledError("Sequence job cancelled.")
            sid = item["session_id"]
            if progress_callback is not None:
                progress_callback(idx, n_items, sid, "processing", idx, None)
            ptid = item["parent_take_id"]
            steps: list[Any] = [*prefix, *item["local_steps"], *suffix]
            if (idx == 0 and float(outer_fade_in_s) > 0.0) or (
                idx == n_items - 1 and float(outer_fade_out_s) > 0.0
            ):
                steps.append(
                    {
                        "filter_name": "fade_edges",
                        "params": {
                            "fade_in_s": float(outer_fade_in_s) if idx == 0 else 0.0,
                            "fade_out_s": float(outer_fade_out_s) if idx == n_items - 1 else 0.0,
                            "curve": "cosine",
                        },
                    }
                )

            prev_p = parent_paths[idx - 1] if idx > 0 else None
            next_p = parent_paths[idx + 1] if idx < n_items - 1 else None
            rec = self.process_take(
                session_id=sid,
                parent_take_id=ptid,
                steps=steps,
                label=item["label"],
                notes=item["notes"],
                prev_audio_path=prev_p,
                next_audio_path=next_p,
                seam_context_s=seam_context_s,
            )
            exported_file: str | None = None
            if item["export_path"] is not None:
                out_p = self.export_take(
                    sid, rec.take_id, item["export_path"], audio_format=export_format
                )
                exported_file = str(out_p)
                if discard_take_audio_after_export:
                    discard_candidates.append((sid, rec.take_id, out_p))

            processed_pairs.append((sid, rec.take_id))
            results.append(
                {
                    "session_id": sid,
                    "take_id": rec.take_id,
                    "label": rec.label,
                    "duration_s": rec.metrics.duration_s,
                    "integrated_lufs": rec.metrics.integrated_lufs,
                    "true_peak_dbtp": rec.metrics.true_peak_dbtp,
                    "rumble_ratio_db": rec.metrics.rumble_ratio_db,
                    "export_path": exported_file,
                }
            )
            if progress_callback is not None:
                progress_callback(idx, n_items, sid, "done", idx + 1, rec.take_id)

        seams_report: dict[str, Any] | None = None
        if verify_seams_flag and len(processed_pairs) >= 2:
            seams_report = self.verify_seams(
                processed_pairs, max_sample_step_excess_db=max_sample_step_excess_db
            )

        clean_report: dict[str, Any] | None = None
        if clean_sections_flag:
            clean_report = self.find_clean_sections(processed_pairs)

        compiled_clean: dict[str, Any] | None = None
        if seq_out_dir is not None and (
            compile_clean_targets or compile_clean_tiers or include_combined_full
        ):
            compiled_clean = self.compile_clean_sections(
                processed_pairs,
                seq_out_dir,
                targets=compile_clean_targets,
                tiers=compile_clean_tiers,
                include_full_masters=False,
                include_combined_full=include_combined_full,
                filename_prefix=filename_prefix,
                audio_format=export_format,
                fade_in_s=clean_fade_in_s,
                fade_out_s=clean_fade_out_s,
                cancel_check=cancel_check,
            )

        for sid, tid, exp_p in discard_candidates:
            if not exp_p.exists() or exp_p.stat().st_size <= 0:
                continue
            with self._get_session_lock(sid):
                summary = self.get_session(sid)
                sdir_res = self._session_dir(sid).resolve()
                for t in summary.takes:
                    if t.take_id != tid:
                        continue
                    cur_audio = Path(t.audio_path)
                    if (
                        cur_audio.exists()
                        and cur_audio.resolve() != exp_p.resolve()
                        and sdir_res in cur_audio.resolve().parents
                    ):
                        cur_audio.unlink()
                        t.audio_path = str(exp_p)
                self._save_session(summary)

        return {
            "count": len(results),
            "output_dir": str(seq_out_dir) if seq_out_dir else None,
            "items": results,
            "seams": seams_report,
            "clean_sections": clean_report,
            "compiled_clean_sections": compiled_clean,
        }


def _pick_clean_interior_windows(
    dur_s: float,
    win_s: float,
    k: int,
    exclude_events: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Choose k interior [start, end] slices avoiding localized distraction intervals."""
    if dur_s <= win_s + 0.5:
        return [(0.0, dur_s)]
    margin = min(dur_s * 0.05, max(0.0, dur_s - win_s) * 0.25)
    lo = margin
    hi = max(lo, dur_s - margin - win_s)
    if hi <= lo + 0.1:
        return [(lo, min(dur_s, lo + win_s))]

    def _overlaps(a: float, b: float) -> bool:
        return any(ee > a - 1.0 and es < b + 1.0 for es, ee in exclude_events)

    step = (hi - lo) / max(1, k - 1) if k > 1 else 0.0
    chosen: list[tuple[float, float]] = []
    for i in range(k):
        base = lo + (i * step if k > 1 else 0.5 * (hi - lo))
        picked: tuple[float, float] | None = None
        for shift in (0.0, 15.0, -15.0, 30.0, -30.0, 60.0, -60.0):
            s = max(lo, min(hi, base + shift))
            if not _overlaps(s, s + win_s):
                picked = (round(s, 3), round(s + win_s, 3))
                break
        if picked is None:
            s = max(lo, min(hi, base))
            picked = (round(s, 3), round(s + win_s, 3))
        chosen.append(picked)
    return chosen
