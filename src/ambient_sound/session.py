"""Non-destructive session, take DAG, and loudness-matched A/B test management."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from ambient_sound.analysis import SensitivityLevel, analyze_audio, analyze_file
from ambient_sound.config import Settings, ensure_path_allowed
from ambient_sound.dsp import apply_recipe, apply_recipe_streaming, compute_residual
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


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip()).strip("-_").lower()
    return cleaned[:32] or "session"


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
            sha256=sha256_file(audio_wav),
            recipe=None,
            metrics=metrics,
            events=events,
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
    ) -> tuple[AudioMetrics, list[DistractionEvent]]:
        """Compute fresh metrics and distraction events for an entire take."""
        take = self.get_take(session_id, take_id)
        return analyze_file(
            take.audio_path,
            sensitivity=sensitivity,
            min_event_s=min_event_s,
            max_event_s=max_event_s,
            freq_range_hz=freq_range_hz,
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
            events=take.events,
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

        new_take_id = f"take_{len(summary.takes):03d}"
        take_label = label.strip() or f"Processed ({len(parsed_steps)} steps)"
        take_dir = self._session_dir(session_id) / "takes" / new_take_id
        take_dir.mkdir(parents=True, exist_ok=True)
        audio_path = take_dir / "audio.wav"

        recipe = ProcessingRecipe(parent_take_id=parent_take_id, steps=parsed_steps, notes=notes)
        (take_dir / "recipe.json").write_text(recipe.model_dump_json(indent=2), encoding="utf-8")

        parent_info = probe_audio(parent.audio_path)
        residual_wav: Path | None = None
        diff_png: Path | None = None

        if parent_info.duration_s <= STREAMING_THRESHOLD_S:
            parent_audio, sr = load_audio(parent.audio_path)
            processed = apply_recipe(parent_audio, sr, parsed_steps)
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
                )
        else:
            out_info = apply_recipe_streaming(
                parent.audio_path,
                audio_path,
                parsed_steps,
                subtype="PCM_24",
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
            sha256=sha256_file(audio_path),
            recipe=recipe,
            metrics=metrics,
            events=events,
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
        """Return sliced residual (from disk if present, else computed on the fly)."""
        take = self.get_take(session_id, take_id)
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

    def create_ab_test(
        self,
        session_id: str,
        take_a_id: str,
        take_b_id: str,
        focus_start_s: float | None = None,
        focus_end_s: float | None = None,
        blind: bool = False,
        notes: str = "",
        original_take_id: str = "take_000",
    ) -> ABTestRecord:
        """Create a loudness-matched A/B test comparing two candidate takes and the original."""
        summary = self.get_session(session_id)
        take_a = self.get_take(session_id, take_a_id)
        take_b = self.get_take(session_id, take_b_id)
        take_o = self.get_take(session_id, original_take_id)

        lufs_values = [
            v
            for v in (
                take_a.metrics.integrated_lufs,
                take_b.metrics.integrated_lufs,
                take_o.metrics.integrated_lufs,
            )
            if v is not None
        ]
        if lufs_values:
            target_lufs = min(lufs_values)

            def _attenuation_gain(lufs: float | None) -> float:
                if lufs is None:
                    return 0.0
                return round(min(0.0, target_lufs - lufs), 2)

            gain_a = _attenuation_gain(take_a.metrics.integrated_lufs)
            gain_b = _attenuation_gain(take_b.metrics.integrated_lufs)
            gain_o = _attenuation_gain(take_o.metrics.integrated_lufs)
        else:
            gain_a = gain_b = gain_o = 0.0

        ab_id = f"ab_{len(summary.ab_tests) + 1:03d}"
        record = ABTestRecord(
            ab_id=ab_id,
            session_id=session_id,
            created_at=_utc_now_iso(),
            take_a_id=take_a_id,
            take_b_id=take_b_id,
            original_take_id=original_take_id,
            focus_start_s=focus_start_s,
            focus_end_s=focus_end_s,
            blind=blind,
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


def read_take_recipe(take: TakeRecord) -> dict[str, Any] | None:
    """Read recipe JSON for a take if present."""
    if take.recipe is not None:
        return take.recipe.model_dump()
    recipe_file = Path(take.audio_path).parent / "recipe.json"
    if recipe_file.exists():
        return json.loads(recipe_file.read_text(encoding="utf-8"))
    return None
