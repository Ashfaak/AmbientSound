"""FastAPI local web server for the loudness-matched A/B Listening Studio."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field

from ambient_sound import __version__
from ambient_sound.config import Settings
from ambient_sound.io_audio import render_wav_bytes
from ambient_sound.models import ABTestRecord, TakeRecord
from ambient_sound.session import SessionManager

STATIC_DIR = Path(__file__).resolve().parent / "static"


class VerdictRequest(BaseModel):
    """Payload submitted by the browser UI when the user judges an A/B test."""

    winner: Literal["a", "b", "tie", "original"]
    artifact_tags: list[str] = Field(default_factory=list)
    comment: str = ""


def _serialize_take_for_ui(
    manager: SessionManager,
    session_id: str,
    take: TakeRecord,
    redact: bool,
    blind_label: str,
) -> dict[str, Any]:
    has_res = manager.can_stream_residual(session_id, take.take_id)
    if redact:
        return {
            "take_id": take.take_id,
            "label": blind_label,
            "metrics": {
                "duration_s": take.metrics.duration_s,
                "sample_rate": take.metrics.sample_rate,
                "channels": take.metrics.channels,
                "integrated_lufs": take.metrics.integrated_lufs,
            },
            "events": [],
            "recipe": None,
            "has_residual": has_res,
            "has_diff_spectrogram": take.diff_spectrogram_path is not None,
        }
    return {
        "take_id": take.take_id,
        "label": take.label,
        "metrics": take.metrics.model_dump(),
        "events": [e.model_dump() for e in take.events],
        "recipe": take.recipe.model_dump() if take.recipe else None,
        "has_residual": has_res,
        "has_diff_spectrogram": take.diff_spectrogram_path is not None,
    }


def _enrich_ab_payload(
    manager: SessionManager, session_id: str, ab: ABTestRecord
) -> dict[str, Any]:
    take_a = manager.get_take(session_id, ab.take_a_id)
    take_b = manager.get_take(session_id, ab.take_b_id)
    take_o = manager.get_take(session_id, ab.original_take_id)
    redact = ab.blind and ab.verdict is None
    return {
        "ab_test": ab.model_dump(),
        "redacted": redact,
        "takes": {
            "a": _serialize_take_for_ui(manager, session_id, take_a, redact, "Candidate A (Blind)"),
            "b": _serialize_take_for_ui(manager, session_id, take_b, redact, "Candidate B (Blind)"),
            "original": _serialize_take_for_ui(manager, session_id, take_o, False, "Original"),
        },
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the FastAPI application for the local A/B Listening Studio."""
    cfg = settings or Settings.from_env()
    manager = SessionManager(cfg)
    app = FastAPI(title="AmbientSound A/B Listening Studio", version=__version__)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html_path = STATIC_DIR / "index.html"
        if not html_path.exists():
            raise HTTPException(status_code=500, detail="UI static asset missing.")
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/api/sessions")
    def api_list_sessions() -> list[dict[str, Any]]:
        return [
            {
                "session_id": s.session_id,
                "name": s.name,
                "created_at": s.created_at,
                "num_takes": len(s.takes),
                "num_ab_tests": len(s.ab_tests),
            }
            for s in manager.list_sessions()
        ]

    @app.get("/api/sessions/{session_id}")
    def api_get_session(session_id: str) -> dict[str, Any]:
        try:
            return manager.get_session(session_id).model_dump()
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/ab-tests")
    def api_list_ab_tests(session_id: str) -> list[dict[str, Any]]:
        try:
            return [ab.model_dump() for ab in manager.list_ab_tests(session_id)]
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/ab-tests/{ab_id}")
    def api_get_ab_test(session_id: str, ab_id: str) -> dict[str, Any]:
        try:
            ab = manager.get_ab_test(session_id, ab_id)
            return _enrich_ab_payload(manager, session_id, ab)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/ab-tests/{ab_id}/verdict")
    def api_submit_verdict(session_id: str, ab_id: str, body: VerdictRequest) -> dict[str, Any]:
        try:
            updated = manager.submit_ab_verdict(
                session_id=session_id,
                ab_id=ab_id,
                winner=body.winner,
                artifact_tags=body.artifact_tags,
                comment=body.comment,
            )
            return _enrich_ab_payload(manager, session_id, updated)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def _take_file_response(
        session_id: str,
        take_id: str,
        attr: str,
        media_type: str,
    ) -> FileResponse:
        try:
            take = manager.get_take(session_id, take_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raw_path = getattr(take, attr, None)
        if not raw_path:
            raise HTTPException(status_code=404, detail=f"{attr} not available for {take_id}.")
        p = Path(raw_path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"File missing on disk: {p.name}")
        return FileResponse(str(p), media_type=media_type)

    @app.get("/api/sessions/{session_id}/takes/{take_id}/audio")
    def api_take_audio(
        session_id: str,
        take_id: str,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> Response:
        if start_s is not None or end_s is not None:
            try:
                arr, sr = manager.load_take_audio_slice(
                    session_id, take_id, start_s=start_s or 0.0, end_s=end_s
                )
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return Response(content=render_wav_bytes(arr, sr), media_type="audio/wav")
        return _take_file_response(session_id, take_id, "audio_path", "audio/wav")

    @app.get("/api/sessions/{session_id}/takes/{take_id}/residual")
    def api_take_residual(
        session_id: str,
        take_id: str,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> Response:
        if start_s is not None or end_s is not None:
            try:
                arr, sr = manager.load_take_residual_slice(
                    session_id, take_id, start_s=start_s or 0.0, end_s=end_s
                )
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return Response(content=render_wav_bytes(arr, sr), media_type="audio/wav")
        return _take_file_response(session_id, take_id, "residual_path", "audio/wav")

    @app.get("/api/sessions/{session_id}/takes/{take_id}/spectrogram")
    def api_take_spectrogram(
        session_id: str,
        take_id: str,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> FileResponse:
        if start_s is not None or end_s is not None:
            try:
                vis = manager.render_take_visuals(
                    session_id, take_id, start_s=start_s, end_s=end_s
                )
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return FileResponse(vis["spectrogram_path"], media_type="image/png")
        return _take_file_response(session_id, take_id, "spectrogram_path", "image/png")

    @app.get("/api/sessions/{session_id}/takes/{take_id}/diff-spectrogram")
    def api_take_diff_spectrogram(
        session_id: str,
        take_id: str,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> FileResponse:
        if start_s is not None or end_s is not None:
            try:
                take = manager.get_take(session_id, take_id)
                if not take.parent_take_id:
                    raise HTTPException(
                        status_code=404, detail=f"diff_spectrogram not available for {take_id}."
                    )
                vis = manager.render_take_visuals(
                    session_id,
                    take_id,
                    compare_to_take_id=take.parent_take_id,
                    start_s=start_s,
                    end_s=end_s,
                )
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return FileResponse(vis["diff_spectrogram_path"], media_type="image/png")
        return _take_file_response(session_id, take_id, "diff_spectrogram_path", "image/png")

    return app


def run_ui(
    host: str = "127.0.0.1",
    port: int = 8765,
    settings: Settings | None = None,
) -> None:
    """Start the local FastAPI A/B Listening Studio with uvicorn."""
    app = create_app(settings=settings)
    uvicorn.run(app, host=host, port=port, log_level="info")


def main(argv: list[str] | None = None) -> None:
    """Entry point for ambient-sound-ui."""
    parser = argparse.ArgumentParser(description="Run AmbientSound local A/B Listening Studio")
    parser.add_argument("--host", default=None, help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default: 8765)")
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    run_ui(
        host=args.host or settings.ui_host,
        port=args.port or settings.ui_port,
        settings=settings,
    )


if __name__ == "__main__":
    main()
