"""Tests for linked-stereo CPU DSP filters and residual invariants."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyloudnorm as pyln

from ambient_sound.analysis import analyze_audio, chunked_true_peak_linear
from ambient_sound.dsp import (
    apply_recipe,
    apply_recipe_streaming,
    apply_recipe_with_seam_padding,
    apply_recipe_with_telemetry,
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
    assert len(specs) == 15
    assert {
        "highpass_rumble",
        "dehum_notch",
        "drone_notch",
        "declick_impulse",
        "spectral_denoise",
        "band_attenuate",
        "ambience_patch",
        "trim_range",
        "cut_regions",
        "fade_edges",
        "gain",
        "parametric_eq",
        "soft_compress",
        "true_peak_limiter",
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


def test_gain_and_limit_peaks_mastering_in_memory_and_streaming(tmp_path: Path) -> None:
    sr = 44100
    n = sr * 4
    rng = np.random.default_rng(88)
    # Quiet ~ -44 LUFS spiky ambient audio
    spiky = rng.normal(0.0, 0.004, size=(n, 2))
    for k in range(1, 9):
        idx = int(k * 0.4 * sr)
        spiky[idx : idx + 6, :] += 0.08

    gained = apply_recipe(spiky, sr, [{"filter_name": "gain", "params": {"gain_db": 12.0}}])
    assert abs((_rms_db(gained) - _rms_db(spiky)) - 12.0) < 0.1

    no_limit = apply_recipe(
        spiky,
        sr,
        [
            {
                "filter_name": "loudness_normalize",
                "params": {
                    "target_lufs": -14.0,
                    "true_peak_ceiling_dbtp": -1.0,
                    "limit_peaks": False,
                },
            }
        ],
    )
    assert pyln.Meter(sr).integrated_loudness(no_limit) < -18.0

    limited = apply_recipe(
        spiky,
        sr,
        [
            {
                "filter_name": "loudness_normalize",
                "params": {
                    "target_lufs": -14.0,
                    "true_peak_ceiling_dbtp": -1.0,
                    "limit_peaks": True,
                },
            }
        ],
    )
    lufs_mem = pyln.Meter(sr).integrated_loudness(limited)
    tp_mem = 20.0 * np.log10(max(chunked_true_peak_linear(limited), 1e-12))
    assert abs(lufs_mem - (-14.0)) <= 0.8
    assert tp_mem <= -0.95

    src_wav = save_wav(tmp_path / "quiet_spiky.wav", spiky, sr, subtype="PCM_24")
    dst_wav = tmp_path / "master_limited.wav"
    tele = []
    apply_recipe_streaming(
        src_wav,
        dst_wav,
        [
            {
                "filter_name": "loudness_normalize",
                "params": {
                    "target_lufs": -14.0,
                    "true_peak_ceiling_dbtp": -1.0,
                    "limit_peaks": True,
                },
            }
        ],
        subtype="PCM_24",
        block_seconds=2.0,
        telemetry_out=tele,
    )
    loaded, _ = load_audio(dst_wav)
    lufs_str = pyln.Meter(sr).integrated_loudness(loaded)
    tp_str = 20.0 * np.log10(max(chunked_true_peak_linear(loaded), 1e-12))
    assert abs(lufs_str - (-14.0)) <= 0.8
    assert tp_str <= -0.95
    assert len(tele) == 1
    assert tele[0].details.get("peak_limited") is True


def test_adaptive_band_attenuate_and_telemetry(tmp_path: Path) -> None:
    sr = 44100
    n = sr * 6
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(64)
    bed = rng.normal(0.0, 0.02, size=(n, 2))

    # Asymmetric 140 Hz distraction concentrated at 1.4..2.3 s inside a wide 1.0..5.0 s span
    noisy = bed.copy()
    p0, p1 = int(1.4 * sr), int(2.3 * sr)
    noisy[p0:p1, :] += (0.20 * np.sin(2.0 * np.pi * 140.0 * t[p0:p1]))[:, None]

    adaptive = apply_recipe(
        noisy,
        sr,
        [
            {
                "filter_name": "band_attenuate",
                "params": {
                    "start_s": 1.0,
                    "end_s": 5.0,
                    "low_hz": 80.0,
                    "high_hz": 220.0,
                    "attenuation_db": -20.0,
                    "fade_ms": 200.0,
                    "envelope_mode": "adaptive",
                },
            }
        ],
    )
    assert _rms_db(adaptive[int(1.6 * sr) : int(2.1 * sr)]) < (
        _rms_db(noisy[int(1.6 * sr) : int(2.1 * sr)]) - 10.0
    )
    # Clean region inside the same 1..5 s span must NOT be hollowed out
    assert (
        abs(
            _rms_db(adaptive[int(3.2 * sr) : int(4.4 * sr)])
            - _rms_db(noisy[int(3.2 * sr) : int(4.4 * sr)])
        )
        < 2.5
    )

    _, tele_mem = apply_recipe_with_telemetry(
        noisy,
        sr,
        [
            {"filter_name": "highpass_rumble", "params": {"cutoff_hz": 50.0}},
            {
                "filter_name": "ambience_patch",
                "params": {
                    "spans": [{"start_s": 1.4, "end_s": 2.3, "crossfade_ms": 60.0}]
                },
            },
        ],
    )
    assert len(tele_mem) == 2
    assert tele_mem[0].input_rms_dbfs is not None
    assert tele_mem[1].details["patched_spans"][0]["crossfade_ms"] == 60.0
    assert "donor_start_s" in tele_mem[1].details["patched_spans"][0]

    src_wav = save_wav(tmp_path / "adapt_in.wav", noisy, sr, subtype="PCM_24")
    tele_str = []
    apply_recipe_streaming(
        src_wav,
        tmp_path / "adapt_out.wav",
        [
            {"filter_name": "highpass_rumble", "params": {"cutoff_hz": 50.0}},
            {
                "filter_name": "ambience_patch",
                "params": {
                    "spans": [{"start_s": 1.4, "end_s": 2.3, "crossfade_ms": 55.0}]
                },
            },
        ],
        subtype="PCM_24",
        block_seconds=2.5,
        telemetry_out=tele_str,
    )
    assert len(tele_str) == 2
    assert tele_str[0].input_lufs is not None
    assert tele_str[0].output_lufs is not None
    assert tele_str[1].input_lufs is not None
    assert tele_str[1].output_lufs is not None
    assert tele_str[1].details["patched_spans"][0]["crossfade_ms"] == 55.0


def test_split_file_seam_continuity_and_validation(tmp_path: Path) -> None:
    import pytest

    sr = 16000
    n_half = sr * 6
    n_full = n_half * 2
    t = np.arange(n_full, dtype=np.float64) / sr
    rng = np.random.default_rng(202)
    bed = rng.normal(0.0, 0.015, size=(n_full, 2))
    # Continuous bee-like buzz + transient across the split point at t = 6.0 s
    buzz = (0.04 * np.sin(2.0 * np.pi * 240.0 * t) + 0.02 * np.sin(2.0 * np.pi * 35.0 * t))[:, None]
    full = bed + buzz
    full[int(5.9 * sr) : int(6.1 * sr), :] += 0.25

    part1_wav = save_wav(tmp_path / "part1.wav", full[:n_half], sr, subtype="PCM_24")
    part2_wav = save_wav(tmp_path / "part2.wav", full[n_half:], sr, subtype="PCM_24")

    chain = [
        {"filter_name": "highpass_rumble", "params": {"cutoff_hz": 55.0, "order": 4}},
        {
            "filter_name": "parametric_eq",
            "params": {"freq_hz": 160.0, "gain_db": 2.0, "filter_type": "lowshelf"},
        },
        {
            "filter_name": "soft_compress",
            "params": {"threshold_db": -28.0, "ratio": 2.0, "release_ms": 250.0},
        },
        {"filter_name": "gain", "params": {"gain_db": 10.0}},
        {"filter_name": "true_peak_limiter", "params": {"ceiling_dbtp": -1.5, "release_ms": 60.0}},
    ]
    ref_full = apply_recipe(full, sr, chain)

    # In-memory seam padding parity
    p1_mem, _ = apply_recipe_with_seam_padding(
        full[:n_half], sr, chain, next_audio_path=part2_wav, seam_context_s=3.0
    )
    p2_mem, _ = apply_recipe_with_seam_padding(
        full[n_half:], sr, chain, prev_audio_path=part1_wav, seam_context_s=3.0
    )
    seam_slice_ref = ref_full[n_half - 1000 : n_half + 1000]
    seam_slice_mem = np.vstack([p1_mem[-1000:], p2_mem[:1000]])
    assert float(np.max(np.abs(seam_slice_mem - seam_slice_ref))) < 1e-3

    # Streaming seam padding parity + PCM_24 WAV verification + telemetry check
    out1_wav = tmp_path / "out1.wav"
    out2_wav = tmp_path / "out2.wav"
    patch_chain = [
        *chain,
        {
            "filter_name": "ambience_patch",
            "params": {"start_s": 1.0, "end_s": 1.6, "source_start_s": 2.5, "crossfade_ms": 40.0},
        },
    ]
    tele2: list = []
    info1 = apply_recipe_streaming(
        part1_wav,
        out1_wav,
        chain,
        subtype="PCM_24",
        block_seconds=3.0,
        next_audio_path=part2_wav,
        seam_context_s=3.0,
    )
    info2 = apply_recipe_streaming(
        part2_wav,
        out2_wav,
        patch_chain,
        subtype="PCM_24",
        block_seconds=3.0,
        telemetry_out=tele2,
        prev_audio_path=part1_wav,
        seam_context_s=3.0,
    )
    assert info1.frames == n_half and info1.format_name.upper() == "WAV"
    assert info2.frames == n_half and info2.format_name.upper() == "WAV"
    out1_arr, _ = load_audio(out1_wav)
    out2_arr, _ = load_audio(out2_wav)
    seam_jump = float(np.max(np.abs(out2_arr[0] - out1_arr[-1])))
    ref_jump = float(np.max(np.abs(ref_full[n_half] - ref_full[n_half - 1])))
    assert abs(seam_jump - ref_jump) < 5e-4
    # Patch telemetry must remain in file-local [0, 6 s] coordinates
    span_info = tele2[-1].details["patched_spans"][0]
    assert abs(span_info["target_start_s"] - 0.98) < 0.05
    assert abs(span_info["donor_start_s"] - 2.5) < 0.05

    # Seam padding + trim_range must fail explicitly
    with pytest.raises(ValueError, match="trim_range"):
        apply_recipe_with_seam_padding(
            full[:n_half],
            sr,
            [{"filter_name": "trim_range", "params": {"start_s": 0.5, "end_s": 2.0}}],
            next_audio_path=part2_wav,
        )
    with pytest.raises(ValueError, match="trim_range"):
        apply_recipe_streaming(
            part1_wav,
            tmp_path / "invalid.wav",
            [{"filter_name": "trim_range", "params": {"start_s": 0.5, "end_s": 2.0}}],
            next_audio_path=part2_wav,
        )
