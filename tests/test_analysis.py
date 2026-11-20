"""Tests for psychoacoustic analysis and ambient distraction detectors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyloudnorm as pyln

from ambient_sound.analysis import (
    StreamingLUFS,
    analyze_audio,
    analyze_file,
    summarize_events,
)
from ambient_sound.dsp import apply_recipe
from ambient_sound.io_audio import save_wav
from ambient_sound.models import DistractionEvent


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


def test_quiet_recording_narrow_band_min_event_and_lra() -> None:
    sr = 44100
    dur = 6.0
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float64) / sr
    # Very quiet ~ -52 LUFS background bed
    bed = _make_pink_bed(n, 2, seed=41) * 0.14

    # Sharp clicks that must be skipped when min_event_s > 0.15
    bed[int(0.7 * sr) : int(0.7 * sr) + 5, :] += 0.09
    bed[int(1.2 * sr) : int(1.2 * sr) + 5, :] += 0.09

    # Quiet 2.4 s narrowband 90 Hz swell (below old -60 dBFS band floor)
    s0, s1 = int(2.0 * sr), int(4.4 * sr)
    swell = np.hanning(s1 - s0)
    bed[s0:s1, :] += (0.008 * swell * np.sin(2.0 * np.pi * 90.0 * t[s0:s1]))[:, np.newaxis]

    metrics, filtered_events = analyze_audio(
        bed,
        sr,
        sensitivity="high",
        min_event_s=1.5,
        max_event_s=20.0,
        freq_range_hz=(40.0, 140.0),
    )
    assert "impulsive_click" not in {ev.kind for ev in filtered_events}
    assert any(
        ev.kind in {"slow_pass", "transient_distraction", "low_freq_rumble"}
        and ev.end_s > 2.2
        and ev.start_s < 4.2
        for ev in filtered_events
    )

    assert metrics.noise_floor_dbfs is not None
    assert metrics.noise_floor_dbfs < -40.0
    assert metrics.loudness_range_lu is not None
    assert metrics.loudness_range_lu >= 0.0
    assert metrics.short_term_lufs_stats is not None
    assert "median_lufs" in metrics.short_term_lufs_stats
    assert len(metrics.short_term_lufs_timeline) > 0

    synth_clicks = [
        DistractionEvent(
            id=f"c_{i}",
            start_s=0.1 * i,
            end_s=0.1 * i + 0.01,
            kind="impulsive_click",
            severity="medium",
        )
        for i in range(10)
    ]
    macro = DistractionEvent(
        id="m_1", start_s=2.0, end_s=6.0, kind="slow_pass", severity="high"
    )
    kept, counts, total, truncated = summarize_events(synth_clicks + [macro], max_events=3)
    assert total == 11
    assert truncated is True
    assert counts == {"impulsive_click": 10, "slow_pass": 1}
    assert any(ev.kind == "slow_pass" for ev in kept)


def test_bilateral_hum_and_persistent_tonal_detection() -> None:
    sr = 16000
    dur = 20.0
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float64) / sr
    bed = _make_pink_bed(n, 2, seed=55)

    # Steep 45 Hz highpass should NOT create a false 50 Hz mains_hum peak
    hpf_only = apply_recipe(
        bed[: sr * 4],
        sr,
        [{"filter_name": "highpass_rumble", "params": {"cutoff_hz": 45.0, "order": 4}}],
    )
    m_hpf, _ = analyze_audio(hpf_only, sr)
    assert 50.0 not in m_hpf.hum_peaks_hz

    # Continuous moderate steady tonal drone at 220 Hz + brief 1.5 s burst at 410 Hz in [6..7.5s]
    steady_220 = (0.014 * np.sin(2.0 * np.pi * 220.0 * t))[:, np.newaxis]
    brief_410 = np.zeros_like(bed)
    b0, b1 = int(6.0 * sr), int(7.5 * sr)
    brief_410[b0:b1, :] = (0.045 * np.sin(2.0 * np.pi * 410.0 * t[b0:b1]))[:, np.newaxis]

    m_drone, _ = analyze_audio(bed + steady_220 + brief_410, sr)
    assert any(abs(f - 220.0) <= 5.0 for f in m_drone.tonal_peaks_hz)
    assert not any(abs(f - 410.0) <= 8.0 for f in m_drone.tonal_peaks_hz)


def test_constant_q_tonal_median_and_slow_pass_hysteresis() -> None:
    sr = 16000
    dur = 20.0
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float64) / sr
    bed = _make_pink_bed(n, 2, seed=91)

    # Steady resonant 2680 Hz line (>1 kHz) + VERY loud brief 167 Hz tone only in [6.0..7.5 s]
    res_2680 = (
        0.008 * np.sin(2.0 * np.pi * 2678.0 * t) + 0.007 * np.sin(2.0 * np.pi * 2684.0 * t)
    )[:, np.newaxis]
    brief_167 = np.zeros_like(bed)
    b0, b1 = int(6.0 * sr), int(7.5 * sr)
    brief_167[b0:b1, :] = (0.14 * np.sin(2.0 * np.pi * 167.0 * t[b0:b1]))[:, np.newaxis]

    # Gentle +6 dB slow 6.0 s vehicle swell across 10.5..16.5 s with trailing tail
    p0, p1 = int(10.5 * sr), int(16.5 * sr)
    env = np.hanning(p1 - p0)
    slow_pass = np.zeros_like(bed)
    slow_pass[p0:p1, :] = (0.045 * env * np.sin(2.0 * np.pi * 110.0 * t[p0:p1]))[:, np.newaxis]

    m, events = analyze_audio(bed + res_2680 + brief_167 + slow_pass, sr, sensitivity="medium")
    assert any(abs(f - 2680.0) <= 12.0 for f in m.tonal_peaks_hz)
    assert not any(abs(f - 167.0) <= 6.0 for f in m.tonal_peaks_hz)

    assert m.ambient_bed_dbfs is not None
    assert m.high_freq_floor_dbfs is not None
    assert m.high_freq_floor_dbfs < m.ambient_bed_dbfs

    slow_ev = [ev for ev in events if ev.kind == "slow_pass"]
    assert any(ev.start_s <= 12.0 and ev.end_s >= 15.0 for ev in slow_ev)

    drone_ev = [ev for ev in events if ev.kind == "machine_drone"]
    assert drone_ev and all(
        ev.freq_band_hz is not None and (ev.freq_band_hz[1] - ev.freq_band_hz[0]) <= 11.0
        for ev in drone_ev
    )

    # Gain-invariant click count check (+0 dB vs +6 dB)
    quiet_clicks = _make_pink_bed(sr * 2, 2, seed=12) * 0.05
    quiet_clicks[int(0.5 * sr) : int(0.5 * sr) + 4, :] += 0.004
    quiet_clicks[int(1.2 * sr) : int(1.2 * sr) + 4, :] += 0.004
    _, ev_0 = analyze_audio(quiet_clicks, sr)
    _, ev_6 = analyze_audio(quiet_clicks * (10.0 ** (6.0 / 20.0)), sr)
    clicks_0 = [e for e in ev_0 if e.kind == "impulsive_click"]
    clicks_6 = [e for e in ev_6 if e.kind == "impulsive_click"]
    assert len(clicks_0) == 2
    assert len(clicks_0) == len(clicks_6)
