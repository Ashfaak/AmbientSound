"""Tests for linked-stereo CPU DSP filters and residual invariants."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyloudnorm as pyln

from ambient_sound.analysis import analyze_audio, chunked_true_peak_linear
from ambient_sound.dsp import (
    apply_recipe,
    apply_recipe_streaming,
    compute_residual,
    list_filter_specs,
)
from ambient_sound.io_audio import load_audio, save_wav


def _rms_db(x: np.ndarray) -> float:
    val = float(np.sqrt(np.mean(x * x)))
    return 20.0 * np.log10(max(val, 1e-12))


def test_filter_specs_registered() -> None:
    specs = list_filter_specs()
    names = {s["name"] for s in specs}
    assert {
        "highpass_rumble",
        "dehum_notch",
        "declick_impulse",
        "spectral_denoise",
        "band_attenuate",
        "ambience_patch",
        "parametric_eq",
        "loudness_normalize",
    }.issubset(names)


def test_highpass_and_dehum_attenuate_targets() -> None:
    sr = 44100
    n = sr * 2
    t = np.arange(n, dtype=np.float64) / sr
    sig = (
        0.25 * np.sin(2.0 * np.pi * 28.0 * t)
        + 0.15 * np.sin(2.0 * np.pi * 60.0 * t)
        + 0.15 * np.sin(2.0 * np.pi * 600.0 * t)
    )
    stereo = np.column_stack([sig, sig * 0.9])

    out = apply_recipe(
        stereo,
        sr,
        [
            {"filter_name": "highpass_rumble", "params": {"cutoff_hz": 50.0, "order": 4}},
            {
                "filter_name": "dehum_notch",
                "params": {"base_freq_hz": 60.0, "harmonics": 3, "q": 30.0},
            },
        ],
    )
    assert out.shape == stereo.shape

    # Measure leftover energy at 28 Hz and 60 Hz via correlation
    p28_in = abs(float(np.mean(stereo[:, 0] * np.sin(2.0 * np.pi * 28.0 * t))))
    p28_out = abs(float(np.mean(out[:, 0] * np.sin(2.0 * np.pi * 28.0 * t))))
    p60_in = abs(float(np.mean(stereo[:, 0] * np.sin(2.0 * np.pi * 60.0 * t))))
    p60_out = abs(float(np.mean(out[:, 0] * np.sin(2.0 * np.pi * 60.0 * t))))
    p600_in = abs(float(np.mean(stereo[:, 0] * np.sin(2.0 * np.pi * 600.0 * t))))
    p600_out = abs(float(np.mean(out[:, 0] * np.sin(2.0 * np.pi * 600.0 * t))))

    assert p28_out < p28_in * 0.05
    assert p60_out < p60_in * 0.1
    assert p600_out > p600_in * 0.9


def test_declick_repairs_impulse_spike() -> None:
    sr = 44100
    rng = np.random.default_rng(12)
    stereo = rng.normal(0.0, 0.01, size=(sr, 2))
    idx = 20000
    stereo[idx : idx + 4, :] += 0.75

    cleaned = apply_recipe(
        stereo,
        sr,
        [
            {
                "filter_name": "declick_impulse",
                "params": {"threshold_sigma": 6.0, "max_width_ms": 3.0},
            }
        ],
    )
    assert cleaned.shape == stereo.shape
    assert float(np.max(np.abs(cleaned[idx - 10 : idx + 15]))) < 0.1


def test_spectral_denoise_preserves_stereo_correlation() -> None:
    sr = 44100
    n = sr * 2
    rng = np.random.default_rng(9)
    shared = rng.normal(0.0, 0.03, size=(n, 1))
    stereo = np.hstack([shared, shared * 0.95 + rng.normal(0.0, 0.005, size=(n, 1))])

    out = apply_recipe(
        stereo,
        sr,
        [{"filter_name": "spectral_denoise", "params": {"strength": 0.4, "floor_db": -16.0}}],
    )
    metrics_in, _ = analyze_audio(stereo, sr)
    metrics_out, _ = analyze_audio(out, sr)
    assert metrics_in.stereo_correlation is not None
    assert metrics_out.stereo_correlation is not None
    assert metrics_out.stereo_correlation > 0.9


def test_band_attenuate_localized_and_ambience_patch_seamless() -> None:
    sr = 44100
    n = sr * 4
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(99)
    bed = rng.normal(0.0, 0.02, size=(n, 2))

    # Add 1200 Hz distraction burst from 1.5 s to 2.2 s
    b0, b1 = int(1.5 * sr), int(2.2 * sr)
    burst = np.zeros_like(bed)
    burst[b0:b1, :] = (0.18 * np.sin(2.0 * np.pi * 1200.0 * t[b0:b1]))[:, np.newaxis]
    noisy = bed + burst

    attenuated = apply_recipe(
        noisy,
        sr,
        [
            {
                "filter_name": "band_attenuate",
                "params": {
                    "start_s": 1.45,
                    "end_s": 2.25,
                    "low_hz": 1000.0,
                    "high_hz": 1400.0,
                    "attenuation_db": -24.0,
                    "fade_ms": 25.0,
                },
            }
        ],
    )
    assert attenuated.shape == noisy.shape
    # Before burst region remains untouched
    assert np.allclose(attenuated[: int(1.0 * sr)], noisy[: int(1.0 * sr)])
    # Burst energy drops substantially
    assert _rms_db(attenuated[b0:b1]) < _rms_db(noisy[b0:b1]) - 10.0

    patched = apply_recipe(
        noisy,
        sr,
        [
            {
                "filter_name": "ambience_patch",
                "params": {"start_s": 1.48, "end_s": 2.22, "crossfade_ms": 40.0},
            }
        ],
    )
    assert patched.shape == noisy.shape
    # Patched span matches clean ambient RMS rather than leaving a silence hole or loud burst
    clean_rms = _rms_db(bed)
    patched_rms = _rms_db(patched[b0:b1])
    assert abs(patched_rms - clean_rms) < 3.0

    residual = compute_residual(noisy, patched)
    assert np.allclose(noisy - patched, residual)


def test_eq_and_loudness_normalize_respects_peak_ceiling() -> None:
    sr = 44100
    rng = np.random.default_rng(3)
    bed = rng.normal(0.0, 0.03, size=(sr * 2, 2))
    eq = apply_recipe(
        bed,
        sr,
        [
            {
                "filter_name": "parametric_eq",
                "params": {"freq_hz": 1000.0, "gain_db": -3.0, "q": 1.4},
            }
        ],
    )
    assert eq.shape == bed.shape

    norm = apply_recipe(
        eq,
        sr,
        [
            {
                "filter_name": "loudness_normalize",
                "params": {"target_lufs": -18.0, "true_peak_ceiling_dbtp": -1.0},
            }
        ],
    )
    lufs = pyln.Meter(sr).integrated_loudness(norm)
    assert abs(lufs - (-18.0)) < 0.8
    assert float(np.max(np.abs(norm))) <= 10.0 ** (-1.0 / 20.0) + 1e-3


def test_drone_notch_trim_cut_fade_compress_and_limiter() -> None:
    sr = 44100
    dur = 4.0
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(77)
    bed = rng.normal(0.0, 0.02, size=(n, 2))
    drone = (0.12 * np.sin(2.0 * np.pi * 140.0 * t))[:, None]
    bee = (0.08 * np.sin(2.0 * np.pi * 260.0 * t))[:, None]
    sig = bed + drone + bee

    notched = apply_recipe(
        sig,
        sr,
        [
            {
                "filter_name": "drone_notch",
                "params": {"freqs_hz": [140.0], "q": 25.0, "attenuation_db": -24.0},
            }
        ],
    )
    p140_in = abs(float(np.mean(sig[:, 0] * np.sin(2.0 * np.pi * 140.0 * t))))
    p140_out = abs(float(np.mean(notched[:, 0] * np.sin(2.0 * np.pi * 140.0 * t))))
    p260_in = abs(float(np.mean(sig[:, 0] * np.sin(2.0 * np.pi * 260.0 * t))))
    p260_out = abs(float(np.mean(notched[:, 0] * np.sin(2.0 * np.pi * 260.0 * t))))
    assert p140_out < p140_in * 0.2
    assert p260_out > p260_in * 0.85

    trimmed = apply_recipe(
        notched, sr, [{"filter_name": "trim_range", "params": {"start_s": 0.5, "end_s": 2.5}}]
    )
    assert abs(trimmed.shape[0] - sr * 2) <= 2

    cut = apply_recipe(
        notched,
        sr,
        [
            {
                "filter_name": "cut_regions",
                "params": {"regions": [{"start_s": 1.0, "end_s": 1.8}], "crossfade_ms": 200.0},
            }
        ],
    )
    assert cut.shape[0] < notched.shape[0] - int(0.6 * sr)

    faded = apply_recipe(
        notched,
        sr,
        [{"filter_name": "fade_edges", "params": {"fade_in_s": 0.3, "fade_out_s": 0.4}}],
    )
    assert float(np.max(np.abs(faded[:10]))) < 1e-4
    assert float(np.max(np.abs(faded[-10:]))) < 1e-4

    # Compressor + true_peak_limiter
    spiky = bed.copy() * 4.0
    spiky[sr : sr + 50, :] = 1.8
    mastered = apply_recipe(
        spiky,
        sr,
        [
            {
                "filter_name": "soft_compress",
                "params": {"threshold_db": -22.0, "ratio": 2.2, "makeup_db": 3.0},
            },
            {
                "filter_name": "true_peak_limiter",
                "params": {"ceiling_dbtp": -1.0, "release_ms": 60.0},
            },
        ],
    )
    tp_db = 20.0 * np.log10(max(chunked_true_peak_linear(mastered), 1e-12))
    assert tp_db <= -0.95


def test_batched_spans_and_streaming_pipeline(tmp_path: Path) -> None:
    sr = 44100
    n = sr * 5
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(101)
    bed = rng.normal(0.0, 0.025, size=(n, 2))
    burst1 = np.zeros_like(bed)
    burst1[int(0.8 * sr) : int(1.6 * sr), :] = (
        0.14 * np.sin(2.0 * np.pi * 110.0 * t[int(0.8 * sr) : int(1.6 * sr)])
    )[:, None]
    burst2 = np.zeros_like(bed)
    burst2[int(3.0 * sr) : int(3.8 * sr), :] = (
        0.14 * np.sin(2.0 * np.pi * 110.0 * t[int(3.0 * sr) : int(3.8 * sr)])
    )[:, None]
    noisy = bed + burst1 + burst2

    multi_att = apply_recipe(
        noisy,
        sr,
        [
            {
                "filter_name": "band_attenuate",
                "params": {
                    "low_hz": 70.0,
                    "high_hz": 170.0,
                    "attenuation_db": -22.0,
                    "fade_ms": 150.0,
                    "envelope_mode": "swell",
                    "spans": [
                        {"start_s": 0.8, "end_s": 1.6},
                        {"start_s": 3.0, "end_s": 3.8},
                    ],
                },
            }
        ],
    )
    assert _rms_db(multi_att[int(0.95 * sr) : int(1.45 * sr)]) < _rms_db(
        noisy[int(0.95 * sr) : int(1.45 * sr)]
    ) - 8.0
    assert _rms_db(multi_att[int(3.15 * sr) : int(3.65 * sr)]) < _rms_db(
        noisy[int(3.15 * sr) : int(3.65 * sr)]
    ) - 8.0

    src_wav = save_wav(tmp_path / "stream_in.wav", noisy, sr, subtype="PCM_24")
    dst_wav = tmp_path / "stream_out.wav"
    steps = [
        {"filter_name": "highpass_rumble", "params": {"cutoff_hz": 55.0}},
        {
            "filter_name": "ambience_patch",
            "params": {
                "crossfade_ms": 50.0,
                "spans": [
                    {"start_s": 0.8, "end_s": 1.6},
                    {"start_s": 3.0, "end_s": 3.8},
                ],
            },
        },
        {
            "filter_name": "loudness_normalize",
            "params": {"target_lufs": -16.0, "true_peak_ceiling_dbtp": -1.0},
        },
        {"filter_name": "fade_edges", "params": {"fade_in_s": 0.2, "fade_out_s": 0.2}},
    ]
    out_info = apply_recipe_streaming(src_wav, dst_wav, steps, subtype="PCM_24", block_seconds=2.0)
    assert out_info.frames == n
    loaded, _ = load_audio(dst_wav)
    assert loaded.shape == noisy.shape
    lufs = pyln.Meter(sr).integrated_loudness(loaded)
    assert abs(lufs - (-16.0)) < 1.0
