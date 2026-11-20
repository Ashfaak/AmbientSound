"""Model Context Protocol (MCP) stdio server for AmbientSound."""

from __future__ import annotations

import logging
import platform
import shutil
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer

from ambient_sound import __version__
from ambient_sound.analysis import SensitivityLevel
from ambient_sound.config import Settings
from ambient_sound.dsp import list_filter_specs
from ambient_sound.session import STREAMING_THRESHOLD_S, SessionManager

# Strictly keep stdout clean for MCP JSON-RPC frames.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ambient_sound.mcp")


def _parse_freq_range(freq_range_hz: list[float] | None) -> tuple[float, float] | None:
    if not freq_range_hz:
        return None
    if len(freq_range_hz) != 2:
        raise ValueError("freq_range_hz must contain exactly two floats: [low_hz, high_hz].")
    return float(freq_range_hz[0]), float(freq_range_hz[1])


def create_mcp_server(settings: Settings | None = None) -> MCPServer:
    """Create and configure the AmbientSound MCPServer instance."""
    cfg = settings or Settings.from_env()
    manager = SessionManager(cfg)
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
        }

    @mcp.tool()
    def list_filters() -> dict[str, Any]:
        """List all supported ambient restoration DSP filters and parameter schemas."""
        return {"filters": list_filter_specs()}

    @mcp.tool()
    def create_session(audio_path: str, name: str = "") -> dict[str, Any]:
        """Ingest audio into a non-destructive session (take_000) in bounded RAM."""
        summary = manager.create_session(audio_path, name=name)
        return summary.model_dump()

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
    def get_session_summary(session_id: str) -> dict[str, Any]:
        """Return complete metadata, takes, metrics, events, and A/B tests for a session."""
        return manager.get_session(session_id).model_dump()

    @mcp.tool()
    def analyze_take(
        session_id: str,
        take_id: str,
        sensitivity: SensitivityLevel = "medium",
        min_event_s: float = 0.12,
        max_event_s: float = 45.0,
        freq_range_hz: list[float] | None = None,
    ) -> dict[str, Any]:
        """Compute psychoacoustic metrics and configurable distraction events for a full take."""
        metrics, events = manager.analyze_take(
            session_id,
            take_id,
            sensitivity=sensitivity,
            min_event_s=min_event_s,
            max_event_s=max_event_s,
            freq_range_hz=_parse_freq_range(freq_range_hz),
        )
        return {
            "session_id": session_id,
            "take_id": take_id,
            "metrics": metrics.model_dump(),
            "events": [e.model_dump() for e in events],
        }

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
    ) -> dict[str, Any]:
        """Analyze a specific [start_s, end_s] time window with custom sensitivity & freq filter."""
        metrics, events = manager.analyze_region(
            session_id,
            take_id,
            start_s=start_s,
            end_s=end_s,
            sensitivity=sensitivity,
            min_event_s=min_event_s,
            max_event_s=max_event_s,
            freq_range_hz=_parse_freq_range(freq_range_hz),
        )
        return {
            "session_id": session_id,
            "take_id": take_id,
            "window": {"start_s": start_s, "end_s": end_s},
            "metrics": metrics.model_dump(),
            "events": [e.model_dump() for e in events],
        }

    @mcp.tool()
    def render_spectrogram(
        session_id: str,
        take_id: str,
        compare_to_take_id: str = "",
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> dict[str, str]:
        """Render overview or zoomed [start_s, end_s] spectrogram and optional 3-panel diff PNG."""
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
    ) -> dict[str, Any]:
        """Apply an ordered DSP filter recipe to parent_take_id as a new immutable take."""
        record = manager.process_take(
            session_id=session_id,
            parent_take_id=parent_take_id,
            steps=steps,
            label=label,
            notes=notes,
        )
        return record.model_dump()

    @mcp.tool()
    def create_ab_test(
        session_id: str,
        take_a_id: str,
        take_b_id: str,
        focus_start_s: float | None = None,
        focus_end_s: float | None = None,
        blind: bool = False,
        notes: str = "",
    ) -> dict[str, Any]:
        """Register a loudness-matched A/B listening test and return the local studio URL."""
        record = manager.create_ab_test(
            session_id=session_id,
            take_a_id=take_a_id,
            take_b_id=take_b_id,
            focus_start_s=focus_start_s,
            focus_end_s=focus_end_s,
            blind=blind,
            notes=notes,
        )
        payload = record.model_dump()
        payload["ui_url"] = manager.ab_ui_url(session_id, record.ab_id)
        payload["workspace_dir"] = str(cfg.workspace_dir)
        return payload

    @mcp.tool()
    def list_ab_tests(session_id: str) -> dict[str, Any]:
        """List all A/B tests and recorded human verdicts for a session."""
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
    ) -> dict[str, str]:
        """Export a take to wav, flac, mp3, or ogg inside AMBIENTSOUND_ALLOWED_ROOTS."""
        out = manager.export_take(session_id, take_id, output_path, audio_format=format)
        return {
            "session_id": session_id,
            "take_id": take_id,
            "format": format.lower(),
            "output_path": str(out),
        }

    return mcp


def main() -> None:
    """Entry point for running the stdio MCP server."""
    logger.info("Starting AmbientSound MCP server v%s over stdio", __version__)
    server = create_mcp_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
