"""Tests for non-destructive session DAG, A/B loudness matching, and feedback logging."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ambient_sound.config import Settings
from ambient_sound.io_audio import save_wav
from ambient_sound.session import SessionManager


def test_session_take_dag_and_blind_ab_lifecycle(tmp_path: Path) -> None:
    settings = Settings(workspace_dir=tmp_path / ".ambientsound", allowed_roots=[tmp_path])
    manager = SessionManager(settings)

    sr = 44100
    n = sr * 2
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(5)
    audio = rng.normal(0.0, 0.03, size=(n, 2)) + (0.04 * np.sin(2.0 * np.pi * 60.0 * t))[:, None]
    src_wav = save_wav(tmp_path / "field.wav", audio, sr)

    summary = manager.create_session(src_wav, name="Forest Rain")
    assert len(summary.takes) == 1
    orig_take = summary.takes[0]
    assert orig_take.take_id == "take_000"
    assert Path(orig_take.audio_path).exists()
    assert Path(orig_take.spectrogram_path).exists()

    take_a = manager.process_take(
        summary.session_id,
        "take_000",
        [{"filter_name": "dehum_notch", "params": {"base_freq_hz": 60.0}}],
        label="De-hum 60Hz",
    )
    take_b = manager.process_take(
        summary.session_id,
        "take_000",
        [
            {"filter_name": "highpass_rumble", "params": {"cutoff_hz": 70.0}},
            {"filter_name": "dehum_notch", "params": {"base_freq_hz": 60.0}},
        ],
        label="HPF + De-hum",
    )
    assert take_a.take_id == "take_001"
    assert take_b.take_id == "take_002"
    assert Path(take_a.residual_path).exists()
    assert Path(take_a.diff_spectrogram_path).exists()

    ab = manager.create_ab_test(
        summary.session_id,
        take_a.take_id,
        take_b.take_id,
        focus_start_s=0.2,
        focus_end_s=1.5,
        blind=True,
        notes="Check low-end warmth vs hum cleanup.",
    )
    # Loudness matching must be attenuation-only (never positive boost)
    assert ab.gain_db_a <= 0.0
    assert ab.gain_db_b <= 0.0
    assert ab.gain_db_original <= 0.0

    judged = manager.submit_ab_verdict(
        summary.session_id,
        ab.ab_id,
        winner="b",
        artifact_tags=["Natural ambience"],
        comment="B sounds cleaner with no warmth loss.",
    )
    assert judged.verdict is not None
    assert judged.verdict.winner == "b"

    log_file = settings.workspace_dir / "sessions" / summary.session_id / "session_feedback_log.md"
    assert log_file.exists()
    assert "Natural ambience" in log_file.read_text(encoding="utf-8")

    exported = manager.export_take(
        summary.session_id, take_b.take_id, tmp_path / "out.flac", "flac"
    )
    assert exported.exists()

    with pytest.raises(PermissionError):
        manager.export_take(summary.session_id, take_b.take_id, Path("C:/Windows/Temp/outside.wav"))
