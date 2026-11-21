"""Tests for the FastAPI local A/B Listening Studio endpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from ambient_sound.config import Settings
from ambient_sound.io_audio import save_wav
from ambient_sound.session import SessionManager
from ambient_sound.web_ui import create_app


def test_web_ui_endpoints_and_blind_redaction(tmp_path: Path) -> None:
    settings = Settings(workspace_dir=tmp_path / ".ambientsound", allowed_roots=[tmp_path])
    manager = SessionManager(settings)

    sr = 44100
    rng = np.random.default_rng(11)
    src = save_wav(tmp_path / "sample.wav", rng.normal(0.0, 0.03, size=(sr, 2)), sr)
    session = manager.create_session(src, name="Park")
    t1 = manager.process_take(
        session.session_id,
        "take_000",
        [{"filter_name": "highpass_rumble", "params": {"cutoff_hz": 55.0}}],
        label="Gentle HPF",
    )
    t2 = manager.process_take(
        session.session_id,
        "take_000",
        [{"filter_name": "spectral_denoise", "params": {"strength": 0.3}}],
        label="Light Denoise",
    )
    ab = manager.create_ab_test(session.session_id, t1.take_id, t2.take_id, blind=True)

    client = TestClient(create_app(settings))

    r_home = client.get("/")
    assert r_home.status_code == 200
    assert "AmbientSound A/B Listening Studio" in r_home.text
    assert "missingBanner" in r_home.text
    assert "/api/info" in r_home.text
    assert "playheadLine" in r_home.text
    assert "specStage" in r_home.text
    assert "ab_diff" in r_home.text
    # Regression guard: windowed URLs must include '?' and slice Original at offset 0
    assert "?start_s=${state.winStart.toFixed(3)}" in r_home.text
    assert "origDur > winDur + 0.25" in r_home.text

    r_info = client.get("/api/info")
    assert r_info.status_code == 200
    info_data = r_info.json()
    assert info_data["workspace_dir"] == str(settings.workspace_dir)
    assert "version" in info_data

    r_sessions = client.get("/api/sessions")
    assert r_sessions.status_code == 200
    assert len(r_sessions.json()) == 1

    # Before verdict in blind mode, labels/recipes/step_metrics/event_counts are redacted
    r_ab = client.get(f"/api/sessions/{session.session_id}/ab-tests/{ab.ab_id}")
    assert r_ab.status_code == 200
    data = r_ab.json()
    assert data["redacted"] is True
    assert data["takes"]["a"]["recipe"] is None
    assert data["takes"]["a"]["step_metrics"] == []
    assert data["takes"]["a"]["event_counts"] == {}
    assert data["takes"]["a"]["events_total"] == 0
    assert data["takes"]["a"]["label"] == "Candidate A (Blind)"

    # Audio, residual, and spectrogram assets stream cleanly
    assert (
        client.get(f"/api/sessions/{session.session_id}/takes/{t1.take_id}/audio").status_code
        == 200
    )
    assert (
        client.get(f"/api/sessions/{session.session_id}/takes/{t1.take_id}/residual").status_code
        == 200
    )
    assert (
        client.get(f"/api/sessions/{session.session_id}/takes/{t1.take_id}/spectrogram").status_code
        == 200
    )
    assert (
        client.get(
            f"/api/sessions/{session.session_id}/takes/{t1.take_id}/diff-spectrogram"
        ).status_code
        == 200
    )

    # Sliced audio, residual, and spectrogram assets stream cleanly
    r_slice_a = client.get(
        f"/api/sessions/{session.session_id}/takes/{t1.take_id}/audio?start_s=0.1&end_s=0.6"
    )
    assert r_slice_a.status_code == 200
    assert len(r_slice_a.content) > 1000

    # Dynamic sliced residual even if residual.wav was not materialized on disk
    Path(t1.residual_path).unlink()
    r_slice_r = client.get(
        f"/api/sessions/{session.session_id}/takes/{t1.take_id}/residual?start_s=0.1&end_s=0.6"
    )
    assert r_slice_r.status_code == 200
    assert len(r_slice_r.content) > 1000

    assert (
        client.get(
            f"/api/sessions/{session.session_id}/takes/{t1.take_id}/spectrogram"
            "?start_s=0.1&end_s=0.6"
        ).status_code
        == 200
    )
    assert (
        client.get(
            f"/api/sessions/{session.session_id}/takes/{t1.take_id}/diff-spectrogram"
            "?start_s=0.1&end_s=0.6"
        ).status_code
        == 200
    )

    # Submit verdict -> reveals labels and recipes
    r_verdict = client.post(
        f"/api/sessions/{session.session_id}/ab-tests/{ab.ab_id}/verdict",
        json={"winner": "a", "artifact_tags": ["Natural ambience"], "comment": "A kept air intact"},
    )
    assert r_verdict.status_code == 200
    revealed = r_verdict.json()
    assert revealed["redacted"] is False
    assert revealed["takes"]["a"]["label"] == "Gentle HPF"
    assert revealed["takes"]["a"]["recipe"] is not None
    assert len(revealed["takes"]["a"]["step_metrics"]) == 1
    assert "event_counts" in revealed["takes"]["a"]
    assert "events_total" in revealed["takes"]["a"]
    assert revealed["ab_test"]["verdict"]["winner"] == "a"

    # Trimmed A/B diffs + A vs B Diff + X-Plot metadata headers
    t_trim_a = manager.process_take(
        session.session_id,
        t1.take_id,
        [{"filter_name": "trim_range", "params": {"start_s": 0.0, "end_s": 0.6}}],
        label="Trim A (0-0.6s)",
    )
    t_trim_b = manager.process_take(
        session.session_id,
        t2.take_id,
        [{"filter_name": "trim_range", "params": {"start_s": 0.0, "end_s": 0.6}}],
        label="Trim B (0-0.6s)",
    )
    ab_trim = manager.create_ab_test(session.session_id, t_trim_a.take_id, t_trim_b.take_id)
    for side in ("a", "b", "ab"):
        r_diff = client.get(
            f"/api/sessions/{session.session_id}/ab-tests/{ab_trim.ab_id}/diff-spectrogram"
            f"?side={side}"
        )
        assert r_diff.status_code == 200
        assert len(r_diff.content) > 1000
        assert "X-Plot-X0" in r_diff.headers
        assert float(r_diff.headers["X-Plot-X1"]) > float(r_diff.headers["X-Plot-X0"])
        assert float(r_diff.headers["X-Time-End"]) > float(r_diff.headers["X-Time-Start"])

    # Trimmed take /diff-spectrogram fallback (diff_spectrogram_path=None) must still return 200
    assert (
        client.get(
            f"/api/sessions/{session.session_id}/takes/{t_trim_a.take_id}/diff-spectrogram"
        ).status_code
        == 200
    )
