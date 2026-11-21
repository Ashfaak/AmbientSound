"""Non-destructive session, take DAG, and loudness-matched A/B test management."""

from __future__ import annotations

import json
import math
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from scipy import signal

from ambient_sound.analysis import (
    SensitivityLevel,
    StreamingLUFS,
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
from ambient_sound.io_audio import (
    export_audio,
    ingest_audio_to_wav,
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
    loc_kinds = {"slow_pass", "low_freq_rumble", "transient_distraction", "clipping"}
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
        sdir = self._session_dir(summary.session_id)
        sdir.mkdir(parents=True, exist_ok=True)
        self._session_json_path(summary.session_id).write_text(
            summary.model_dump_json(indent=2),
            encoding="utf-8",
        )

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
            if entry.is_dir() and meta_path.exists():
                summaries.append(
                    SessionSummary.model_validate_json(meta_path.read_text(encoding="utf-8"))
                )
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

        if side == "a":
            before_t, after_t = take_o, take_a
            title = f"{take_o.take_id} -> {take_a.take_id} ({take_a.label})"
        elif side == "b":
            before_t, after_t = take_o, take_b
            title = f"{take_o.take_id} -> {take_b.take_id} ({take_b.label})"
        else:
            before_t, after_t = take_a, take_b
            title = f"{take_a.take_id} -> {take_b.take_id} (Candidate A vs B)"

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
        summary = self.get_session(session_id)
        parent = self.get_take(session_id, parent_take_id)
        parsed_steps = [
            s if isinstance(s, FilterStep) else FilterStep.model_validate(s) for s in steps
        ]
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

        recipe = ProcessingRecipe(parent_take_id=parent_take_id, steps=parsed_steps, notes=notes)
        (take_dir / "recipe.json").write_text(recipe.model_dump_json(indent=2), encoding="utf-8")

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
                residual_wav = save_wav(take_dir / "residual.wav", residual, sr, subtype="FLOAT")
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
            if not changes_length and out_info.frames == parent_info.frames:
                diff_png = render_file_diff_spectrogram_png(
                    parent.audio_path,
                    audio_path,
                    take_dir / "diff_spectrogram.png",
                    title=f"{parent_take_id} -> {new_take_id} ({take_label})",
                    loudness_match_residual=True,
                    patched_spans=_patched_spans_from_telemetry(step_metrics) or None,
                )

        events_file, counts, bounded_events = self._write_take_events(take_dir, events)

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
        """Return True if a take has a saved residual or matches parent frames for sliced diff."""
        take = self.get_take(session_id, take_id)
        if take.residual_path and Path(take.residual_path).exists():
            return True
        if not take.parent_take_id:
            return False
        parent = self.get_take(session_id, take.parent_take_id)
        return probe_audio(take.audio_path).frames == probe_audio(parent.audio_path).frames

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
            if p_info.frames == t_info.frames:
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
        if p_info.frames != t_info.frames:
            raise FileNotFoundError(
                f"Residual unavailable for length-changing take {take_id}."
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
        ab_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")

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
        ab_path.write_text(target_ab.model_dump_json(indent=2), encoding="utf-8")
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

    def export_take(
        self,
        session_id: str,
        take_id: str,
        output_path: str | Path,
        audio_format: str = "wav",
    ) -> Path:
        """Export a take to WAV, FLAC, MP3, or OGG inside allowed roots."""
        dst = ensure_path_allowed(output_path, self.settings)
        take = self.get_take(session_id, take_id)
        return export_audio(
            take.audio_path, dst, take.metrics.sample_rate, audio_format=audio_format
        )

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
        )

    def verify_seams(
        self,
        sequence: list[tuple[str, str]],
        window_s: float = 10.0,
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
            if end_arr.shape[0] > 0 and start_arr.shape[0] > 0:
                step_lin = float(np.max(np.abs(start_arr[0] - end_arr[-1])))
                sample_step_dbfs = _finite_or_none(
                    20.0 * math.log10(max(step_lin, 1e-6)), digits=2
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

            passed = bool(
                (lufs_delta is None or lufs_delta <= 1.5)
                and (rms_delta_2s is None or rms_delta_2s <= 2.0)
                and (spectral_diff_db is None or spectral_diff_db <= 3.0)
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
                    "spectral_diff_db": spectral_diff_db,
                    "passed": passed,
                    "spectrogram_path": str(seam_png),
                }
            )

        return {
            "pair_count": len(seams),
            "all_passed": all(s["passed"] for s in seams),
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
        before_steps = [
            s if isinstance(s, FilterStep) else FilterStep.model_validate(s)
            for s in shared_steps_before_gain
        ]
        after_steps = [
            s if isinstance(s, FilterStep) else FilterStep.model_validate(s)
            for s in (shared_steps_after_gain or [])
        ]
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
    ) -> dict[str, Any]:
        """Batch-process an ordered sequence with neighbor seam padding, export, and QC."""
        if not items:
            raise ValueError("process_sequence requires at least one item.")

        validated: list[dict[str, Any]] = []
        parent_paths: list[str] = []
        for idx, raw in enumerate(items):
            sid = str(raw.get("session_id", "")).strip()
            if not sid:
                raise ValueError(f"Sequence item {idx} missing 'session_id'.")
            ptid = str(raw.get("parent_take_id") or "take_000").strip()
            self.get_session(sid)
            ptake = self.get_take(sid, ptid)
            exp_raw = raw.get("export_path")
            exp_path = (
                ensure_path_allowed(exp_raw, self.settings)
                if exp_raw is not None and str(exp_raw).strip()
                else None
            )
            if discard_take_audio_after_export and exp_path is None:
                raise ValueError(
                    f"Sequence item {idx} ({sid}) must specify 'export_path' when "
                    "discard_take_audio_after_export=True."
                )
            local = list(raw.get("local_steps") or [])
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

        prefix = list(shared_prefix_steps or [])
        suffix = list(shared_suffix_steps or [])
        n_items = len(validated)
        results: list[dict[str, Any]] = []
        processed_pairs: list[tuple[str, str]] = []
        discard_candidates: list[tuple[str, str, Path]] = []

        for idx, item in enumerate(validated):
            sid = item["session_id"]
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

        seams_report: dict[str, Any] | None = None
        if verify_seams_flag and len(processed_pairs) >= 2:
            seams_report = self.verify_seams(processed_pairs)

        clean_report: dict[str, Any] | None = None
        if clean_sections_flag:
            clean_report = self.find_clean_sections(processed_pairs)

        for sid, tid, exp_p in discard_candidates:
            if not exp_p.exists() or exp_p.stat().st_size <= 0:
                continue
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
            "items": results,
            "seams": seams_report,
            "clean_sections": clean_report,
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


def read_take_recipe(take: TakeRecord) -> dict[str, Any] | None:
    """Read recipe JSON for a take if present."""
    if take.recipe is not None:
        return take.recipe.model_dump()
    recipe_file = Path(take.audio_path).parent / "recipe.json"
    if recipe_file.exists():
        return json.loads(recipe_file.read_text(encoding="utf-8"))
    return None
