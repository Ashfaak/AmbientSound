"""Tests for psychoacoustic analysis and ambient distraction detectors."""

from __future__ import annotations

import json

import numpy as np

from ambient_sound.analysis import analyze_audio


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
