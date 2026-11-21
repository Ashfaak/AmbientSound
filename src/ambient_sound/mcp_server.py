"""Model Context Protocol (MCP) stdio server for AmbientSound."""

from __future__ import annotations

import logging
import platform
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from ambient_sound import __version__
from ambient_sound.analysis import (
    SensitivityLevel,
    compute_clean_sections,
    event_kind_counts,
    extract_localized_distraction_spans,
    extract_localized_edit_spans,
    summarize_events,
)
from ambient_sound.config import Settings
from ambient_sound.dsp import list_filter_specs
from ambient_sound.engines import assess_do_no_harm
from ambient_sound.jobs import JobManager
from ambient_sound.models import AudioMetrics, DistractionEvent, TakeRecord
from ambient_sound.session import STREAMING_THRESHOLD_S, SessionManager

# Strictly keep stdout clean for MCP JSON-RPC frames.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ambient_sound.mcp")


@contextmanager
def _mcp_errors() -> Iterator[None]:
    try:
        yield
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise ToolError(str(exc)) from exc


def _parse_freq_range(freq_range_hz: list[float] | None) -> tuple[float, float] | None:
    if not freq_range_hz:
        return None
    if len(freq_range_hz) != 2:
        raise ValueError("freq_range_hz must contain exactly two floats: [low_hz, high_hz].")
    return float(freq_range_hz[0]), float(freq_range_hz[1])


def _compact_events_payload(
    events: list[DistractionEvent],
    event_kinds: list[str] | None,
    max_events: int,
    include_events: bool,
    stored_counts: dict[str, int] | None = None,
    include_clicks: bool | None = None,
    source_offset_s: float = 0.0,
) -> dict[str, Any]:
    counts = stored_counts or event_kind_counts(events)
    if not include_events:
        if stored_counts is not None:
            if event_kinds is not None:
                allowed = {str(k).strip() for k in event_kinds if str(k).strip()}
                total_filtered = sum(stored_counts.get(k, 0) for k in allowed)
            else:
                total_filtered = sum(stored_counts.values())
        else:
            _, _, total_filtered, _ = summarize_events(
                events,
                event_kinds=event_kinds,
                max_events=0,
                include_clicks=include_clicks,
            )
        return {
            "event_counts": counts,
            "events_total": total_filtered,
            "events_truncated": False,
            "events": [],
        }
    bounded, computed_counts, total_filtered, truncated = summarize_events(
        events,
        event_kinds=event_kinds,
        max_events=max_events,
        include_clicks=include_clicks,
    )
    ev_dicts: list[dict[str, Any]] = []
    off = float(source_offset_s or 0.0)
    for ev in bounded:
        d = ev.model_dump()
        if off > 0.0:
            d["parent_start_s"] = round(float(ev.start_s) + off, 3)
            d["parent_end_s"] = round(float(ev.end_s) + off, 3)
        ev_dicts.append(d)
    return {
        "event_counts": stored_counts or computed_counts,
        "events_total": total_filtered,
        "events_truncated": truncated,
        "events": ev_dicts,
    }


def _format_job_payload(rec: Any, include_result: bool) -> dict[str, Any]:
    data = rec.model_dump()
    res = data.get("result")
    data["has_result"] = res is not None
    data["manifest_path"] = res.get("manifest_path") if isinstance(res, dict) else None
    if not include_result:
        data["result"] = None
    return data


def _clean_sections_for_take(
    take: TakeRecord,
    source_label: str,
    full_events: list[DistractionEvent],
) -> dict[str, Any]:
    return compute_clean_sections(
        [
            {
                "label": source_label,
                "session_id": take.session_id,
                "take_id": take.take_id,
                "duration_s": take.metrics.duration_s,
                "edit_spans": extract_localized_edit_spans(take.recipe),
                "distraction_spans": extract_localized_distraction_spans(full_events),
            }
        ]
    )


def _format_metrics(
    metrics: AudioMetrics | dict[str, Any], compact_metrics: bool
) -> dict[str, Any]:
    data = metrics.model_dump() if isinstance(metrics, AudioMetrics) else dict(metrics)
    if compact_metrics:
        data["short_term_lufs_timeline"] = []
    return data


def _parse_sequence_pairs(sequence: list[dict[str, str]]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in sequence:
        sid = str(item.get("session_id", "")).strip()
        tid = str(item.get("take_id", "")).strip()
        if not sid or not tid:
            raise ValueError(
                "Each sequence entry must provide non-empty 'session_id' and 'take_id'."
            )
        pairs.append((sid, tid))
    return pairs


def _resolve_sequence_or_session(
    manager: SessionManager,
    *,
    session_id: str = "",
    take_id: str = "",
    sequence: list[dict[str, str]] | None = None,
) -> list[tuple[str, str]]:
    if sequence:
        return _parse_sequence_pairs(sequence)
    if session_id.strip():
        summary = manager.get_session(session_id.strip())
        resolved_tid = take_id.strip() or (
            summary.takes[-1].take_id if summary.takes else "take_000"
        )
        return [(session_id.strip(), resolved_tid)]
    return []


def _format_take_payload(
    manager: SessionManager,
    take: TakeRecord,
    event_kinds: list[str] | None = None,
    max_events: int = 50,
    include_events: bool = True,
    source_label: str = "",
    compact_metrics: bool = False,
    include_clicks: bool | None = None,
    compact_takes: bool = False,
) -> dict[str, Any]:
    if compact_takes:
        data: dict[str, Any] = {
            "take_id": take.take_id,
            "session_id": take.session_id,
            "label": take.label,
            "parent_take_id": take.parent_take_id,
            "created_at": take.created_at,
            "audio_path": take.audio_path,
            "spectrogram_path": take.spectrogram_path,
            "diff_spectrogram_path": take.diff_spectrogram_path,
            "source_offset_s": take.source_offset_s,
            "timeline_modified": take.timeline_modified,
            "metrics": _format_metrics(take.metrics, True),
            "event_counts": dict(take.event_counts),
        }
        return data

    data = take.model_dump()
    if compact_metrics and isinstance(data.get("metrics"), dict):
        data["metrics"]["short_term_lufs_timeline"] = []
    if take.events_path:
        full_events = manager.load_take_events(take.session_id, take.take_id)
    else:
        full_events = list(take.events)
    compact = _compact_events_payload(
        full_events,
        event_kinds=event_kinds,
        max_events=max_events,
        include_events=include_events,
        stored_counts=take.event_counts or None,
        include_clicks=include_clicks,
        source_offset_s=take.source_offset_s,
    )
    data.update(compact)
    data["clean_sections"] = _clean_sections_for_take(
        take, source_label or take.take_id, full_events
    )
    return data


def create_mcp_server(settings: Settings | None = None) -> MCPServer:
    """Create and configure the AmbientSound MCPServer instance."""
    cfg = settings or Settings.from_env()
    manager = SessionManager(cfg)
    job_manager = JobManager(cfg.workspace_dir)
    mcp = MCPServer("ambient-sound")

    @mcp.tool()
    def health_check() -> dict[str, Any]:
        """Verify MCP connectivity, sandboxed roots, ffmpeg availability, and workspace disk."""
        ffmpeg_path = shutil.which("ffmpeg")
        ffprobe_path = shutil.which("ffprobe")
        usage = shutil.disk_usage(cfg.workspace_dir)
        return {
            "status": "ok",
            "version": __version__,
            "python_version": platform.python_version(),
            "workspace_dir": str(cfg.workspace_dir),
            "allowed_roots": [str(r) for r in cfg.allowed_roots],
            "ffmpeg_available": ffmpeg_path is not None,
            "ffmpeg_path": ffmpeg_path,
            "ffprobe_available": ffprobe_path is not None,
            "workspace_free_gb": round(usage.free / (1024**3), 2),
            "streaming_threshold_s": STREAMING_THRESHOLD_S,
            "ui_base_url": cfg.ui_base_url,
            "filter_count": len(list_filter_specs()),
            "tool_count": 26,
        }

    @mcp.tool()
    def list_filters() -> dict[str, Any]:
        """List all supported ambient restoration DSP filters and parameter schemas."""
        specs = list_filter_specs()
        return {
            "filter_count": len(specs),
            "filters": specs,
            "step_schema": {
                "filter_name": "str (one of the filter names above)",
                "params": "dict[str, Any] (filter-specific parameters)",
            },
            "spans_precedence": (
                "When 'spans' is non-empty, each span overrides top-level start_s/end_s "
                "and inherits any omitted defaults from top-level params."
            ),
        }

    @mcp.tool()
    def create_session(
        audio_path: str,
        name: str = "",
        event_kinds: list[str] | None = None,
        max_events: int = 50,
        include_events: bool = True,
        include_clicks: bool | None = None,
        copy_source: bool = True,
        compact_metrics: bool = False,
    ) -> dict[str, Any]:
        """Ingest audio into a non-destructive session (take_000) in bounded RAM."""
        with _mcp_errors():
            summary = manager.create_session(audio_path, name=name, copy_source=copy_source)
            src_label = Path(summary.source_path).name if summary.source_path else summary.name
            payload = summary.model_dump()
            payload["takes"] = [
                _format_take_payload(
                    manager,
                    t,
                    event_kinds=event_kinds,
                    max_events=max_events,
                    include_events=include_events,
                    source_label=src_label,
                    compact_metrics=compact_metrics,
                    include_clicks=include_clicks,
                )
                for t in summary.takes
            ]
            return payload

    @mcp.tool()
    def list_sessions() -> dict[str, Any]:
        """List existing AmbientSound sessions in the workspace."""
        return {
            "sessions": [
                {
                    "session_id": s.session_id,
                    "name": s.name,
                    "created_at": s.created_at,
                    "source_path": s.source_path,
                    "num_takes": len(s.takes),
                    "num_ab_tests": len(s.ab_tests),
                }
                for s in manager.list_sessions()
            ]
        }

    @mcp.tool()
    def get_session_summary(
        session_id: str,
        event_kinds: list[str] | None = None,
        max_events: int = 50,
        include_events: bool = True,
        include_clicks: bool | None = None,
        compact_metrics: bool = False,
        take_ids: list[str] | None = None,
        compact_takes: bool = False,
    ) -> dict[str, Any]:
        """Return session metadata, takes, metrics, bounded events, and A/B tests."""
        with _mcp_errors():
            summary = manager.get_session(session_id)
            src_label = Path(summary.source_path).name if summary.source_path else summary.name
            payload = summary.model_dump()
            selected = (
                [t for t in summary.takes if t.take_id in set(take_ids)]
                if take_ids
                else summary.takes
            )
            payload["takes"] = [
                _format_take_payload(
                    manager,
                    t,
                    event_kinds=event_kinds,
                    max_events=max_events,
                    include_events=include_events,
                    source_label=src_label,
                    compact_metrics=compact_metrics,
                    include_clicks=include_clicks,
                    compact_takes=compact_takes,
                )
                for t in selected
            ]
            return payload

    @mcp.tool()
    def analyze_take(
        session_id: str,
        take_id: str,
        sensitivity: SensitivityLevel = "medium",
        min_event_s: float = 0.12,
        max_event_s: float = 45.0,
        freq_range_hz: list[float] | None = None,
        event_kinds: list[str] | None = None,
        max_events: int = 50,
        include_events: bool = True,
        include_clicks: bool | None = None,
        compact_metrics: bool = False,
        compare_to_take_id: str | None = None,
    ) -> dict[str, Any]:
        """Compute psychoacoustic metrics, cleanliness gate, and distraction events for a take."""
        with _mcp_errors():
            summary = manager.get_session(session_id)
            take = manager.get_take(session_id, take_id)
            src_label = Path(summary.source_path).name if summary.source_path else summary.name
            metrics, events = manager.analyze_take(
                session_id,
                take_id,
                sensitivity=sensitivity,
                min_event_s=min_event_s,
                max_event_s=max_event_s,
                freq_range_hz=_parse_freq_range(freq_range_hz),
                event_kinds=event_kinds,
            )
            out: dict[str, Any] = {
                "session_id": session_id,
                "take_id": take_id,
                "source_offset_s": take.source_offset_s,
                "metrics": _format_metrics(metrics, compact_metrics),
                **_compact_events_payload(
                    events,
                    event_kinds,
                    max_events,
                    include_events,
                    include_clicks=include_clicks,
                    source_offset_s=take.source_offset_s,
                ),
                "cleanliness_assessment": assess_do_no_harm(metrics, events),
                "clean_sections": _clean_sections_for_take(take, src_label, events),
            }
            if compare_to_take_id and compare_to_take_id.strip():
                out["comparison"] = manager.compare_takes(
                    session_id, take_id, compare_to_take_id.strip()
                )
            return out

    @mcp.tool()
    def analyze_region(
        session_id: str,
        take_id: str,
        start_s: float = 0.0,
        end_s: float | None = None,
        sensitivity: SensitivityLevel = "medium",
        min_event_s: float = 0.12,
        max_event_s: float = 45.0,
        freq_range_hz: list[float] | None = None,
        event_kinds: list[str] | None = None,
        max_events: int = 50,
        include_events: bool = True,
        include_clicks: bool | None = None,
        compact_metrics: bool = False,
        compare_to_take_id: str | None = None,
    ) -> dict[str, Any]:
        """Analyze a specific [start_s, end_s] time window with custom sensitivity & freq filter."""
        with _mcp_errors():
            take = manager.get_take(session_id, take_id)
            metrics, events = manager.analyze_region(
                session_id,
                take_id,
                start_s=start_s,
                end_s=end_s,
                sensitivity=sensitivity,
                min_event_s=min_event_s,
                max_event_s=max_event_s,
                freq_range_hz=_parse_freq_range(freq_range_hz),
                event_kinds=event_kinds,
            )
            out: dict[str, Any] = {
                "session_id": session_id,
                "take_id": take_id,
                "source_offset_s": take.source_offset_s,
                "window": {"start_s": start_s, "end_s": end_s},
                "metrics": _format_metrics(metrics, compact_metrics),
                **_compact_events_payload(
                    events,
                    event_kinds,
                    max_events,
                    include_events,
                    include_clicks=include_clicks,
                    source_offset_s=take.source_offset_s,
                ),
                "cleanliness_assessment": assess_do_no_harm(metrics, events),
            }
            if compare_to_take_id and compare_to_take_id.strip():
                out["comparison"] = manager.compare_takes(
                    session_id,
                    take_id,
                    compare_to_take_id.strip(),
                    start_s=start_s,
                    end_s=end_s,
                )
            return out

    @mcp.tool()
    def render_spectrogram(
        session_id: str,
        take_id: str,
        compare_to_take_id: str = "",
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> dict[str, str]:
        """Render overview or zoomed [start_s, end_s] spectrogram and optional 3-panel diff PNG."""
        with _mcp_errors():
            return manager.render_take_visuals(
                session_id,
                take_id,
                compare_to_take_id=compare_to_take_id,
                start_s=start_s,
                end_s=end_s,
            )

    @mcp.tool()
    def process_take(
        session_id: str,
        parent_take_id: str,
        steps: list[dict[str, Any]],
        label: str = "",
        notes: str = "",
        event_kinds: list[str] | None = None,
        max_events: int = 50,
        include_events: bool = True,
        include_clicks: bool | None = None,
        prev_audio_path: str | None = None,
        next_audio_path: str | None = None,
        seam_context_s: float = 8.0,
        compact_metrics: bool = False,
    ) -> dict[str, Any]:
        """Apply an ordered DSP filter recipe to parent_take_id as a new immutable take."""
        with _mcp_errors():
            summary = manager.get_session(session_id)
            src_label = Path(summary.source_path).name if summary.source_path else summary.name
            record = manager.process_take(
                session_id=session_id,
                parent_take_id=parent_take_id,
                steps=steps,
                label=label,
                notes=notes,
                prev_audio_path=prev_audio_path,
                next_audio_path=next_audio_path,
                seam_context_s=seam_context_s,
            )
            return _format_take_payload(
                manager,
                record,
                event_kinds=event_kinds,
                max_events=max_events,
                include_events=include_events,
                source_label=src_label,
                compact_metrics=compact_metrics,
                include_clicks=include_clicks,
            )

    @mcp.tool()
    def create_ab_test(
        session_id: str,
        take_a_id: str,
        take_b_id: str,
        focus_start_s: float | None = None,
        focus_end_s: float | None = None,
        blind: bool = False,
        notes: str = "",
        original_take_id: str | None = None,
        match_original: bool = False,
    ) -> dict[str, Any]:
        """Register a loudness-matched A/B listening test and return the local studio URL."""
        with _mcp_errors():
            record = manager.create_ab_test(
                session_id=session_id,
                take_a_id=take_a_id,
                take_b_id=take_b_id,
                focus_start_s=focus_start_s,
                focus_end_s=focus_end_s,
                blind=blind,
                notes=notes,
                original_take_id=original_take_id,
                match_original=match_original,
            )
            payload = record.model_dump()
            payload["ui_url"] = manager.ab_ui_url(session_id, record.ab_id)
            payload["workspace_dir"] = str(cfg.workspace_dir)
            return payload

    @mcp.tool()
    def delete_take(
        session_id: str,
        take_id: str,
        cascade: bool = False,
    ) -> dict[str, Any]:
        """Delete an abandoned non-root take (with optional cascade) to reclaim workspace disk."""
        with _mcp_errors():
            return manager.delete_take(session_id, take_id, cascade=cascade)

    @mcp.tool()
    def delete_session(session_id: str) -> dict[str, Any]:
        """Permanently delete a session directory and all its takes and A/B tests."""
        with _mcp_errors():
            return manager.delete_session(session_id)

    @mcp.tool()
    def list_ab_tests(session_id: str) -> dict[str, Any]:
        """List all A/B tests and recorded human verdicts for a session."""
        with _mcp_errors():
            return {
                "workspace_dir": str(cfg.workspace_dir),
                "ab_tests": [
                    {
                        **ab.model_dump(),
                        "ui_url": manager.ab_ui_url(session_id, ab.ab_id),
                    }
                    for ab in manager.list_ab_tests(session_id)
                ],
            }

    @mcp.tool()
    def get_ab_feedback(session_id: str, ab_id: str) -> dict[str, Any]:
        """Retrieve human listening winner, artifact tags, and comments for an A/B test."""
        with _mcp_errors():
            ab = manager.get_ab_test(session_id, ab_id)
            return {
                **ab.model_dump(),
                "ui_url": manager.ab_ui_url(session_id, ab.ab_id),
                "workspace_dir": str(cfg.workspace_dir),
            }

    @mcp.tool()
    def export_take(
        session_id: str,
        take_id: str,
        output_path: str,
        format: str = "wav",
        discard_take_audio: bool = False,
    ) -> dict[str, Any]:
        """Export a take with verified loudness/hash metadata and optional temp audio cleanup."""
        with _mcp_errors():
            return manager.export_take_detailed(
                session_id,
                take_id,
                output_path,
                audio_format=format,
                discard_take_audio=discard_take_audio,
            )

    @mcp.tool()
    def find_clean_sections(
        session_id: str = "",
        take_id: str = "",
        sequence: list[dict[str, str]] | None = None,
        max_edits_per_hour: float = 2.0,
        max_edited_pct: float = 1.0,
        include_low_severity: bool = False,
        round_to_file_local_minutes: bool = False,
    ) -> dict[str, Any]:
        """Rank clean ASMR stretches (15 min-10 hr) for a take or ordered sequence.

        When take_id is omitted with session_id, uses that session's latest take.
        sequence accepts [{"session_id": ..., "take_id": ...}, ...].
        """
        with _mcp_errors():
            pairs = _resolve_sequence_or_session(
                manager, session_id=session_id, take_id=take_id, sequence=sequence
            )
            return manager.find_clean_sections(
                pairs,
                max_edits_per_hour=max_edits_per_hour,
                max_edited_pct=max_edited_pct,
                include_low_severity=include_low_severity,
                round_to_file_local_minutes=round_to_file_local_minutes,
            )

    @mcp.tool()
    def verify_seams(
        sequence: list[dict[str, str]],
        window_s: float = 10.0,
        max_sample_step_excess_db: float = 18.0,
    ) -> dict[str, Any]:
        """Measure LUFS, 2 s RMS, slew-normalized sample step, and spectral seam continuity.

        sequence accepts [{"session_id": ..., "take_id": ...}, ...].
        """
        with _mcp_errors():
            pairs = _parse_sequence_pairs(sequence)
            return manager.verify_seams(
                pairs,
                window_s=window_s,
                max_sample_step_excess_db=max_sample_step_excess_db,
            )

    @mcp.tool()
    def calibrate_sequence(
        sequence: list[dict[str, str]],
        shared_steps_before_gain: list[dict[str, Any]],
        target_lufs: float = -14.5,
        ceiling_dbtp: float = -1.0,
        sample_windows_per_file: int = 5,
        window_s: float = 20.0,
        shared_steps_after_gain: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Solve one shared master gain across a continuous split sequence on clean slices.

        sequence accepts [{"session_id": ..., "take_id": ...}, ...].
        shared_steps_before_gain may be [] when no filters precede the shared gain stage.
        """
        with _mcp_errors():
            pairs = _parse_sequence_pairs(sequence)
            return manager.calibrate_sequence(
                pairs,
                shared_steps_before_gain=shared_steps_before_gain,
                target_lufs=target_lufs,
                ceiling_dbtp=ceiling_dbtp,
                sample_windows_per_file=sample_windows_per_file,
                window_s=window_s,
                shared_steps_after_gain=shared_steps_after_gain,
            )

    @mcp.tool()
    def process_sequence(
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
        output_dir: str | None = None,
        compile_clean_targets: list[str] | None = None,
        compile_clean_tiers: list[str] | None = None,
        include_combined_full: bool = False,
        clean_fade_in_s: float = 2.0,
        clean_fade_out_s: float = 3.0,
        filename_prefix: str = "",
    ) -> dict[str, Any]:
        """Batch-process an ordered split sequence with seam padding, export, and QC.

        Each entry in items accepts: {"session_id" (required), "parent_take_id" (default
        "take_000"), "local_steps" (list[FilterStep]), "label", "notes", "export_path"}.
        When output_dir is given inside AMBIENTSOUND_ALLOWED_ROOTS, auto-exports full
        masters to <output_dir>/<stem>_master.<ext> (unless export_path overrides it)
        and optionally compiles clean_sections into <output_dir>/clean_sections/.
        """
        with _mcp_errors():
            return manager.process_sequence(
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
            )

    @mcp.tool()
    def start_sequence_job(
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
        output_dir: str | None = None,
        compile_clean_targets: list[str] | None = None,
        compile_clean_tiers: list[str] | None = None,
        include_combined_full: bool = False,
        clean_fade_in_s: float = 2.0,
        clean_fade_out_s: float = 3.0,
        filename_prefix: str = "",
    ) -> dict[str, Any]:
        """Launch `process_sequence` asynchronously so long multi-hour runs never time out."""
        with _mcp_errors():
            rec = job_manager.start_sequence_job(
                manager,
                items=items,
                shared_prefix_steps=shared_prefix_steps,
                shared_suffix_steps=shared_suffix_steps,
                outer_fade_in_s=outer_fade_in_s,
                outer_fade_out_s=outer_fade_out_s,
                seam_context_s=seam_context_s,
                verify_seams=verify_seams,
                clean_sections=clean_sections,
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
            )
            return rec.model_dump()

    @mcp.tool()
    def compile_clean_sections(
        output_dir: str,
        session_id: str = "",
        take_id: str = "",
        sequence: list[dict[str, str]] | None = None,
        targets: list[str] | None = None,
        tiers: list[str] | None = None,
        include_full_masters: bool = False,
        include_combined_full: bool = False,
        filename_prefix: str = "",
        format: str = "wav",
        fade_in_s: float = 2.0,
        fade_out_s: float = 3.0,
        max_edits_per_hour: float = 2.0,
        max_edited_pct: float = 1.0,
        include_low_severity: bool = False,
        round_to_file_local_minutes: bool = False,
        distinct_regions: bool = False,
        max_files: int | None = None,
    ) -> dict[str, Any]:
        """Compile ranked YouTube ASMR clean sections (and optional masters) to output_dir.

        Splits within a single take or stitches seamlessly across multi-file sequences,
        writes clean audio files + clean_sections_manifest.json into output_dir inside
        AMBIENTSOUND_ALLOWED_ROOTS, and auto-uses RF64 if a WAV export exceeds ~3.8 GB.
        """
        with _mcp_errors():
            pairs = _resolve_sequence_or_session(
                manager, session_id=session_id, take_id=take_id, sequence=sequence
            )
            return manager.compile_clean_sections(
                sequence=pairs,
                output_dir=output_dir,
                targets=targets,
                tiers=tiers,
                include_full_masters=include_full_masters,
                include_combined_full=include_combined_full,
                filename_prefix=filename_prefix,
                audio_format=format,
                fade_in_s=fade_in_s,
                fade_out_s=fade_out_s,
                max_edits_per_hour=max_edits_per_hour,
                max_edited_pct=max_edited_pct,
                include_low_severity=include_low_severity,
                round_to_file_local_minutes=round_to_file_local_minutes,
                distinct_regions=distinct_regions,
                max_files=max_files,
            )

    @mcp.tool()
    def start_compile_clean_sections_job(
        output_dir: str,
        session_id: str = "",
        take_id: str = "",
        sequence: list[dict[str, str]] | None = None,
        targets: list[str] | None = None,
        tiers: list[str] | None = None,
        include_full_masters: bool = False,
        include_combined_full: bool = False,
        filename_prefix: str = "",
        format: str = "wav",
        fade_in_s: float = 2.0,
        fade_out_s: float = 3.0,
        max_edits_per_hour: float = 2.0,
        max_edited_pct: float = 1.0,
        include_low_severity: bool = False,
        round_to_file_local_minutes: bool = False,
        distinct_regions: bool = False,
        max_files: int | None = None,
    ) -> dict[str, Any]:
        """Launch `compile_clean_sections` asynchronously for long multi-hour compilations."""
        with _mcp_errors():
            pairs = _resolve_sequence_or_session(
                manager, session_id=session_id, take_id=take_id, sequence=sequence
            )
            rec = job_manager.start_compile_clean_sections_job(
                manager,
                sequence=pairs,
                output_dir=output_dir,
                targets=targets,
                tiers=tiers,
                include_full_masters=include_full_masters,
                include_combined_full=include_combined_full,
                filename_prefix=filename_prefix,
                audio_format=format,
                fade_in_s=fade_in_s,
                fade_out_s=fade_out_s,
                max_edits_per_hour=max_edits_per_hour,
                max_edited_pct=max_edited_pct,
                include_low_severity=include_low_severity,
                round_to_file_local_minutes=round_to_file_local_minutes,
                distinct_regions=distinct_regions,
                max_files=max_files,
            )
            return rec.model_dump()

    @mcp.tool()
    def get_job_status(job_id: str, include_result: bool = True) -> dict[str, Any]:
        """Poll progress, per-item status, or final result of a background job."""
        with _mcp_errors():
            return _format_job_payload(job_manager.get_job(job_id), include_result)

    @mcp.tool()
    def wait_for_job(
        job_id: str,
        timeout_s: float = 30.0,
        poll_interval_s: float = 1.0,
        include_result: bool = True,
    ) -> dict[str, Any]:
        """Wait up to timeout_s (<=55 s) for a background job to progress or complete."""
        with _mcp_errors():
            return _format_job_payload(
                job_manager.wait_for_job(
                    job_id, timeout_s=timeout_s, poll_interval_s=poll_interval_s
                ),
                include_result,
            )

    @mcp.tool()
    def cancel_job(job_id: str) -> dict[str, Any]:
        """Request cooperative cancellation of a queued or running background job."""
        with _mcp_errors():
            return job_manager.cancel_job(job_id).model_dump()

    @mcp.tool()
    def list_jobs(
        status: str | None = None, limit: int = 20, include_result: bool = False
    ) -> dict[str, Any]:
        """List recent background jobs persisted in workspace/jobs (compact by default)."""
        with _mcp_errors():
            return {
                "jobs": [
                    _format_job_payload(j, include_result)
                    for j in job_manager.list_jobs(status=status, limit=limit)
                ]
            }

    return mcp


def main() -> None:
    """Entry point for running the stdio MCP server."""
    logger.info("Starting AmbientSound MCP server v%s over stdio", __version__)
    server = create_mcp_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
