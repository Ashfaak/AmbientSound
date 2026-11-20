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

from ambient_sound.analysis import (
    SensitivityLevel,
    analyze_audio,
    analyze_file,
    event_kind_counts,
    summarize_events,
)
from ambient_sound.config import Settings, ensure_path_allowed
from ambient_sound.dsp import (
    apply_recipe_streaming,
    apply_recipe_with_telemetry,
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

    def create_session(self, source_path: str | Path, name: str = "") -> SessionSummary:
        """Ingest an audio recording into immutable take_000 in bounded RAM."""
        src = ensure_path_allowed(source_path, self.settings)

        session_name = name.strip() or src.stem
        session_id = f"{_slugify(session_name)}_{uuid4().hex[:6]}"
        sdir = self._session_dir(session_id)
        take_dir = sdir / "takes" / "take_000"
        take_dir.mkdir(parents=True, exist_ok=True)
        (sdir / "ab_tests").mkdir(parents=True, exist_ok=True)

        audio_wav = take_dir / "audio.wav"
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
        take_dir = Path(take.audio_path).parent
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
            )
            result["diff_spectrogram_path"] = str(diff_path)
        elif (
            not windowed
            and take.diff_spectrogram_path
            and Path(take.diff_spectrogram_path).exists()
        ):
            result["diff_spectrogram_path"] = take.diff_spectrogram_path
        return result

    def process_take(
        self,
        session_id: str,
        parent_take_id: str,
        steps: list[FilterStep] | list[dict[str, Any]],
        label: str = "",
        notes: str = "",
    ) -> TakeRecord:
        """Apply an ordered filter chain to a parent take, producing a new immutable child take."""
        summary = self.get_session(session_id)
        parent = self.get_take(session_id, parent_take_id)
        parsed_steps = [
            s if isinstance(s, FilterStep) else FilterStep.model_validate(s) for s in steps
        ]
        changes_length = any(
            st.filter_name in {"trim_range", "cut_regions"} for st in parsed_steps
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
        residual_wav: Path | None = None
        diff_png: Path | None = None
        step_metrics: list[StepTelemetry] = []

        if parent_info.duration_s <= STREAMING_THRESHOLD_S:
            parent_audio, sr = load_audio(parent.audio_path)
            processed, step_metrics = apply_recipe_with_telemetry(
                parent_audio, sr, parsed_steps
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
    ) -> str:
        if explicit_original_id and explicit_original_id.strip():
            ref = self.get_take(summary.session_id, explicit_original_id.strip())
            return ref.take_id

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

        # Earliest shared ancestor matching both candidate timelines
        for anc in chain_a:
            if anc.take_id in set_b and _matches_both(anc):
                return anc.take_id

        # Earliest ancestor of A (e.g., trimmed preview root) matching timeline & duration
        for anc in chain_a:
            if anc.take_id not in {take_a.take_id, take_b.take_id} and _matches_both(anc):
                return anc.take_id

        return "take_000" if "take_000" in take_map else chain_a[0].take_id

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
        resolved_orig_id = self._resolve_original_take_id(
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


def read_take_recipe(take: TakeRecord) -> dict[str, Any] | None:
    """Read recipe JSON for a take if present."""
    if take.recipe is not None:
        return take.recipe.model_dump()
    recipe_file = Path(take.audio_path).parent / "recipe.json"
    if recipe_file.exists():
        return json.loads(recipe_file.read_text(encoding="utf-8"))
    return None
