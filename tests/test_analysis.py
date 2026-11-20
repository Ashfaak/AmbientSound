"""Tests for psychoacoustic analysis and ambient distraction detectors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyloudnorm as pyln

from ambient_sound.analysis import StreamingLUFS, analyze_audio, analyze_file
from ambient_sound.io_audio import save_wav


def _make_pink_bed(n_samples: int, channels: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, 0.04, size=(n_samples, channels))
    bed = np.empty_like(white)
    for ch in range(channels):
        acc = 0.0
        for i in range(n_samples):
            acc = 0.92 * acc + 0.08 * white[i, ch]
            bed[i, ch] = acc * 2.0
    return bed


def test_analyze_detects_hum_click_and_transient() -> None:
    sr = 44100
    dur = 4.0
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float64) / sr

    bed = _make_pink_bed(n, 2)
    hum = (0.02 * np.sin(2.0 * np.pi * 60.0 * t) + 0.01 * np.sin(2.0 * np.pi * 120.0 * t))[
        :, np.newaxis
    ]
    audio = bed + hum

    # Inject sharp impulsive click at 0.9 s
    c0 = int(0.9 * sr)
    audio[c0 : c0 + 6, :] += 0.65

    # Inject 1 kHz localized distraction burst from 2.2 s to 2.7 s
    b0, b1 = int(2.2 * sr), int(2.7 * sr)
    audio[b0:b1, :] += (0.14 * np.hanning(b1 - b0) * np.sin(2.0 * np.pi * 1000.0 * t[b0:b1]))[
        :, np.newaxis
    ]

    metrics, events = analyze_audio(audio, sr)
    # Ensure JSON serialization succeeds with zero NaN/Inf issues
    dumped = json.dumps(metrics.model_dump(), allow_nan=False)
    assert "NaN" not in dumped

    assert metrics.channels == 2
    assert metrics.sample_rate == sr
    assert abs(metrics.duration_s - dur) < 0.01
    assert metrics.integrated_lufs is not None
    assert 60.0 in metrics.hum_peaks_hz

    kinds = {ev.kind for ev in events}
    assert "mains_hum" in kinds
    assert "impulsive_click" in kinds
    assert "transient_distraction" in kinds


def test_analyze_handles_short_and_silent_audio_without_nan() -> None:
    sr = 44100
    short = np.random.default_rng(1).normal(0.0, 0.01, size=(int(0.1 * sr), 1))
    m_short, _ = analyze_audio(short, sr)
    assert m_short.integrated_lufs is None
    json.dumps(m_short.model_dump(), allow_nan=False)

    silent = np.zeros((sr, 2), dtype=np.float64)
    m_silent, ev_silent = analyze_audio(silent, sr)
    assert m_silent.integrated_lufs is None
    assert m_silent.rms_dbfs is None
    assert ev_silent == []
    json.dumps(m_silent.model_dump(), allow_nan=False)


def test_streaming_lufs_parity_and_analyze_file_region(tmp_path: Path) -> None:
    sr = 48000
    n = sr * 6
    bed = _make_pink_bed(n, 2, seed=19)
    ref_lufs = float(pyln.Meter(sr).integrated_loudness(bed))

    acc = StreamingLUFS(sr, 2)
    acc.update(bed[: sr * 3])
    acc.update(bed[sr * 3 :])
    stream_lufs = acc.finalize()
    assert stream_lufs is not None
    assert abs(stream_lufs - ref_lufs) < 0.2

    wav_path = save_wav(tmp_path / "long_test.wav", bed, sr, subtype="PCM_24")
    m_reg, _ = analyze_file(wav_path, start_s=1.0, end_s=5.0)
    assert abs(m_reg.duration_s - 4.0) < 0.05
    assert m_reg.integrated_lufs is not None
    assert abs(m_reg.integrated_lufs - ref_lufs) < 0.5


def test_configurable_detection_slow_pass_and_machine_drone() -> None:
    sr = 44100
    dur = 8.0
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float64) / sr
    bed = _make_pink_bed(n, 2, seed=31) * 0.4

    # Continuous 140 Hz machine drone (non-50/60 Hz)
    drone = (0.025 * np.sin(2.0 * np.pi * 140.0 * t))[:, np.newaxis]

    # Slow 4.0 s passing vehicle rumble centered around 90 Hz from 2.5 s to 6.5 s
    p0, p1 = int(2.5 * sr), int(6.5 * sr)
    swell = np.hanning(p1 - p0)
    vehicle = np.zeros_like(bed)
    vehicle[p0:p1, :] = (0.14 * swell * np.sin(2.0 * np.pi * 90.0 * t[p0:p1]))[:, np.newaxis]

    mix = bed + drone + vehicle
    metrics, events = analyze_audio(
        mix,
        sr,
        sensitivity="high",
        min_event_s=0.2,
        max_event_s=20.0,
        freq_range_hz=(40.0, 130.0),
    )
    assert any(abs(f - 140.0) <= 5.0 for f in metrics.tonal_peaks_hz)
    kinds = {ev.kind for ev in events}
    assert "machine_drone" in kinds
    assert "slow_pass" in kinds
    slow_passes = [ev for ev in events if ev.kind == "slow_pass"]
    assert any(ev.start_s <= 3.5 and ev.end_s >= 5.5 for ev in slow_passes)
