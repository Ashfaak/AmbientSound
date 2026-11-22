"""Linked-stereo CPU DSP engine supporting both in-memory and streaming long-file processing."""

from __future__ import annotations

import copy
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import interpolate, ndimage, signal

from ambient_sound.analysis import (
    StreamingLUFS,
    StreamingTruePeak,
    chunked_true_peak_linear,
)
from ambient_sound.engines import (
    _hf_patch_flatness,
    apply_masked_stft_transform,
    ar_interpolate_gap,
    compute_bird_protect_mask,
    cosine_edge_mask,
    detect_channel_layout,
    detect_clipping_mask,
    estimate_gcc_phat_lag,
    finite_or_none,
    fit_least_squares_fir,
    fractional_shift_channel,
)
from ambient_sound.io_audio import (
    ensure_2d_audio,
    iter_audio_blocks,
    load_audio_slice,
    probe_audio,
)
from ambient_sound.models import AudioInfo, FilterStep, StepTelemetry

_STREAM_TEMP_FORMAT = "RF64"

FILTER_SPECS: list[dict[str, Any]] = [
    {
        "name": "highpass_rumble",
        "summary": (
            "Zero-phase Butterworth high-pass filter to remove subsonic wind and mic rumble."
        ),
        "ambient_guidance": (
            "Use 45-85 Hz to clear wind/rumble while protecting 200-300 Hz bee buzz; "
            "raise to 80-120 Hz only on heavy urban traffic without low ambient fundamentals."
        ),
        "params": {
            "cutoff_hz": {"type": "float", "default": 60.0, "min": 15.0, "max": 300.0},
            "order": {"type": "int", "default": 4, "min": 1, "max": 8},
        },
    },
    {
        "name": "dehum_notch",
        "summary": (
            "Zero-phase cascaded IIR notch filter removing 50/60 Hz mains hum and harmonics."
        ),
        "ambient_guidance": (
            "Check hum_peaks_hz in analyze_take first to pick 50.0 or 60.0 Hz base_freq_hz."
        ),
        "params": {
            "base_freq_hz": {"type": "float", "default": 60.0, "min": 20.0, "max": 200.0},
            "harmonics": {"type": "int", "default": 4, "min": 1, "max": 8},
            "q": {"type": "float", "default": 30.0, "min": 5.0, "max": 80.0},
        },
    },
    {
        "name": "drone_notch",
        "summary": (
            "Zero-phase proportional peaking-cut notches at arbitrary machine-drone frequencies."
        ),
        "ambient_guidance": (
            "Use tonal_peaks_hz from analyze_take/analyze_region to attenuate steady machinery "
            "lines selectively without carving out surrounding bee or insect harmonics."
        ),
        "params": {
            "freqs_hz": {"type": "list[float]", "required": True},
            "q": {"type": "float", "default": 25.0, "min": 2.0, "max": 80.0},
            "attenuation_db": {"type": "float", "default": -24.0, "min": -48.0, "max": -1.0},
            "harmonics": {"type": "int", "default": 1, "min": 1, "max": 6},
        },
    },
    {
        "name": "declick_impulse",
        "summary": (
            "MAD spike detector + PCHIP or AR interpolation (global or localized spans)."
        ),
        "ambient_guidance": (
            "Pass spans=[{start_s, end_s}], method='ar', min_flatness=0.20 for cable-tie ratchet "
            "click trains while protecting birdsong and skipping global organic ticks."
        ),
        "params": {
            "threshold_sigma": {"type": "float", "default": 6.0, "min": 3.0, "max": 20.0},
            "max_width_ms": {"type": "float", "default": 3.0, "min": 0.5, "max": 12.0},
            "start_s": {"type": "float | null", "default": None},
            "end_s": {"type": "float | null", "default": None},
            "spans": {"type": "list[dict] | null", "default": None},
            "method": {"type": "str", "default": "pchip", "choices": ["pchip", "ar"]},
            "min_flatness": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0},
        },
    },
    {
        "name": "spectral_denoise",
        "summary": (
            "Linked-stereo soft STFT Wiener/spectral gate targeting steady noise/hiss."
        ),
        "ambient_guidance": (
            "Keep strength modest (0.15-0.35) on rain/bees/surf to avoid metallic shimmer."
        ),
        "params": {
            "strength": {"type": "float", "default": 0.45, "min": 0.0, "max": 1.0},
            "floor_db": {"type": "float", "default": -18.0, "min": -48.0, "max": -3.0},
            "n_fft": {"type": "int", "default": 2048, "min": 256, "max": 8192},
            "noise_start_s": {"type": "float | null", "default": None},
            "noise_end_s": {"type": "float | null", "default": None},
        },
    },
    {
        "name": "spectral_match_suppress",
        "summary": (
            "Excess-capped, bird-safe dynamic spectral suppressor with gap-buffered context "
            "and optional L/R channel-excess or narrowband tone surgery."
        ),
        "ambient_guidance": (
            "Use spatial_mode='channel_excess' for L/R-biased cars, narrowband_mode=True "
            "+ protect_birds=True for Garmin watch beeps / steady narrow lines, and "
            "context_gap_s=1.5-3.0 so context never samples a car's own tail."
        ),
        "params": {
            "start_s": {"type": "float | null", "default": None},
            "end_s": {"type": "float | null", "default": None},
            "low_hz": {"type": "float", "default": 25.0, "min": 15.0, "max": 12000.0},
            "high_hz": {"type": "float", "default": 4000.0, "min": 60.0, "max": 20000.0},
            "strength": {"type": "float", "default": 1.0, "min": 0.1, "max": 1.5},
            "max_attenuation_db": {"type": "float", "default": -32.0, "min": -60.0, "max": -3.0},
            "texture_fill": {"type": "float", "default": 0.65, "min": 0.0, "max": 1.0},
            "context_s": {"type": "float", "default": 6.0, "min": 1.0, "max": 20.0},
            "context_gap_s": {"type": "float", "default": 1.5, "min": 0.0, "max": 15.0},
            "fade_ms": {"type": "float", "default": 1500.0, "min": 20.0, "max": 5000.0},
            "n_fft": {"type": "int", "default": 4096, "min": 1024, "max": 8192},
            "ref_start_s": {"type": "float | null", "default": None},
            "ref_end_s": {"type": "float | null", "default": None},
            "protect_birds": {"type": "bool", "default": False},
            "spatial_mode": {
                "type": "str",
                "default": "linked",
                "choices": ["linked", "channel_excess"],
            },
            "narrowband_mode": {"type": "bool", "default": False},
            "spans": {"type": "list[dict] | null", "default": None},
        },
    },
    {
        "name": "band_attenuate",
        "summary": (
            "Feathered time- and frequency-localized linked-stereo STFT attenuation "
            "(single or batched spans)."
        ),
        "ambient_guidance": (
            "Use envelope_mode='adaptive' for asymmetric 5-20 s vehicle passes so peak cut tracks "
            "the loudest excess without hollowing quiet edges, or 'swell' for symmetric fades. "
            "Pass spans=[{start_s, end_s, low_hz?, high_hz?, attenuation_db?, fade_ms?, "
            "envelope_mode?}] to batch multiple passes in one step."
        ),
        "params": {
            "start_s": {"type": "float | null", "default": None},
            "end_s": {"type": "float | null", "default": None},
            "low_hz": {"type": "float", "default": 40.0},
            "high_hz": {"type": "float", "default": 250.0},
            "attenuation_db": {"type": "float", "default": -18.0, "min": -60.0, "max": 12.0},
            "fade_ms": {"type": "float", "default": 30.0, "min": 2.0, "max": 5000.0},
            "envelope_mode": {
                "type": "str",
                "default": "flat",
                "choices": ["flat", "swell", "adaptive"],
            },
            "n_fft": {"type": "int", "default": 2048, "min": 256, "max": 8192},
            "spans": {"type": "list[dict] | null", "default": None},
        },
    },
    {
        "name": "ambience_patch",
        "summary": (
            "Replaces distracted time span(s) with non-repeating, knock-screened clean texture."
        ),
        "ambient_guidance": (
            "Supports min_donor_distance_s (avoids event tails), max_donor_crest 50ms knock gate, "
            "crossfade_ms up to 5000 ms, and reports loop_count & donor_quality."
        ),
        "params": {
            "start_s": {"type": "float | null", "default": None},
            "end_s": {"type": "float | null", "default": None},
            "source_start_s": {"type": "float | null", "default": None},
            "crossfade_ms": {"type": "float", "default": 40.0, "min": 5.0, "max": 5000.0},
            "min_donor_distance_s": {"type": "float", "default": 5.0, "min": 0.0, "max": 120.0},
            "max_donor_crest": {"type": "float", "default": 6.0, "min": 2.5, "max": 20.0},
            "spans": {"type": "list[dict] | null", "default": None},
        },
    },
    {
        "name": "trim_range",
        "summary": "Extracts a contiguous [start_s, end_s] time segment from the recording.",
        "ambient_guidance": (
            "Use for rapid 30-second audition previews on long files or trimming lead/tail setup."
        ),
        "params": {
            "start_s": {"type": "float", "default": 0.0, "min": 0.0},
            "end_s": {"type": "float | null", "default": None},
        },
    },
    {
        "name": "cut_regions",
        "summary": (
            "Removes severe distracted stretches and stitches remaining segments with "
            "equal-power crossfades."
        ),
        "ambient_guidance": (
            "Best fix for long-form ASMR/nature recordings when a prolonged vehicle pass or "
            "conversation cannot be cleanly separated."
        ),
        "params": {
            "regions": {"type": "list[dict]", "required": True},
            "crossfade_ms": {"type": "float", "default": 1000.0, "min": 10.0, "max": 5000.0},
        },
    },
    {
        "name": "fade_edges",
        "summary": "Applies smooth cosine or linear fade-in at start and fade-out at end.",
        "ambient_guidance": "Recommended 2-5 s cosine fades on final ASMR / field exports.",
        "params": {
            "fade_in_s": {"type": "float", "default": 2.0, "min": 0.0, "max": 60.0},
            "fade_out_s": {"type": "float", "default": 3.0, "min": 0.0, "max": 60.0},
            "curve": {"type": "str", "default": "cosine", "choices": ["cosine", "linear"]},
        },
    },
    {
        "name": "gain",
        "summary": "Pure sample-accurate linear gain stage in decibels (dB).",
        "ambient_guidance": (
            "Use before soft_compress on quiet recordings (e.g. -50 LUFS) so dynamics "
            "processing reaches threshold cleanly."
        ),
        "params": {
            "gain_db": {"type": "float", "default": 0.0, "min": -48.0, "max": 48.0},
        },
    },
    {
        "name": "parametric_eq",
        "summary": "RBJ Audio EQ Cookbook biquad filter (peaking, lowshelf, or highshelf).",
        "ambient_guidance": (
            "Use subtle +/- 2 to 4 dB lowshelf (warmth) or highshelf (>10 kHz air) for ASMR."
        ),
        "params": {
            "freq_hz": {"type": "float", "required": True, "min": 20.0, "max": 20000.0},
            "gain_db": {"type": "float", "required": True, "min": -24.0, "max": 24.0},
            "q": {"type": "float", "default": 1.4, "min": 0.2, "max": 20.0},
            "filter_type": {
                "type": "str",
                "default": "peaking",
                "choices": ["peaking", "lowshelf", "highshelf"],
            },
        },
    },
    {
        "name": "soft_compress",
        "summary": "Linked-stereo soft-knee RMS compressor for gentle dynamic leveling.",
        "ambient_guidance": (
            "Keep ratio mild (1.5-2.5) and slow attack/release so ambient breathing stays natural."
        ),
        "params": {
            "threshold_db": {"type": "float", "default": -24.0, "min": -60.0, "max": -3.0},
            "ratio": {"type": "float", "default": 2.0, "min": 1.0, "max": 10.0},
            "attack_ms": {"type": "float", "default": 25.0, "min": 1.0, "max": 500.0},
            "release_ms": {"type": "float", "default": 250.0, "min": 20.0, "max": 2000.0},
            "makeup_db": {"type": "float", "default": 0.0, "min": -12.0, "max": 24.0},
        },
    },
    {
        "name": "true_peak_limiter",
        "summary": "Linked-stereo lookahead peak limiter enforcing a strict ceiling.",
        "ambient_guidance": (
            "Place after loudness_normalize or after calibrated gain + soft_compress on "
            "continuous split sequences (e.g. -1.0 dBTP) for YouTube readiness."
        ),
        "params": {
            "ceiling_dbtp": {"type": "float", "default": -1.0, "min": -6.0, "max": -0.1},
            "release_ms": {"type": "float", "default": 80.0, "min": 10.0, "max": 1000.0},
        },
    },
    {
        "name": "loudness_normalize",
        "summary": "Integrated ITU-R BS.1770 LUFS normalization with optional peak limiting.",
        "ambient_guidance": (
            "Target -14 LUFS with limit_peaks=True (or pre-stage gain + soft_compress) for "
            "YouTube/ASMR playback, or -18 to -24 LUFS with limit_peaks=False for quiet beds."
        ),
        "params": {
            "target_lufs": {"type": "float", "default": -18.0, "min": -40.0, "max": -6.0},
            "true_peak_ceiling_dbtp": {"type": "float", "default": -1.0, "min": -6.0, "max": -0.1},
            "limit_peaks": {"type": "bool", "default": False},
            "limiter_release_ms": {"type": "float", "default": 80.0, "min": 10.0, "max": 1000.0},
        },
    },
    {
        "name": "declip_from_backup",
        "summary": (
            "Sub-sample GCC-PHAT lag + LS-FIR repair of clipped regions using a -12 dB "
            "safety backup WAV (with AR interpolation fallback)."
        ),
        "ambient_guidance": (
            "Splices calibrated safety-track audio ONLY across digital/analog-clipped spans with "
            "short crossfades so backup preamp noise is never exposed in quiet passages."
        ),
        "params": {
            "backup_audio_path": {"type": "str", "required": True},
            "clip_threshold": {"type": "float", "default": 0.985, "min": 0.85, "max": 0.9999},
            "crossfade_ms": {"type": "float", "default": 2.0, "min": 0.2, "max": 20.0},
            "fir_taps": {"type": "int", "default": 31, "min": 5, "max": 127},
            "max_lag_ms": {"type": "float", "default": 250.0, "min": 5.0, "max": 2000.0},
            "ar_fallback": {"type": "bool", "default": True},
        },
    },
    {
        "name": "spatial_mix",
        "summary": (
            "Channel layout conversion & spatial cleanup: SSH-6 M/S decode, SGH-6 dual-mic "
            "crossover/aligned blend, bass mono-ing, wind side-HPF, and L/R balance."
        ),
        "ambient_guidance": (
            "Check metrics.stereo_layout_hint first. Use low_mono_hz=120-160 + side_hpf_hz=140 "
            "for uncorrelated outdoor wind, or crossover_hz=600 for unaligned shotgun+surround."
        ),
        "params": {
            "mode": {
                "type": "str",
                "default": "lr_adjust",
                "choices": [
                    "auto",
                    "ms_decode",
                    "dual_mic_blend",
                    "dual_mono_to_stereo",
                    "lr_adjust",
                ],
            },
            "side_gain_db": {"type": "float", "default": 0.0, "min": -24.0, "max": 12.0},
            "directional_weight": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "directional_channel": {"type": "int", "default": 0, "min": 0, "max": 1},
            "crossover_hz": {"type": "float", "default": 0.0, "min": 0.0, "max": 4000.0},
            "align_delay": {"type": "bool", "default": True},
            "low_mono_hz": {"type": "float", "default": 0.0, "min": 0.0, "max": 300.0},
            "side_hpf_hz": {"type": "float", "default": 0.0, "min": 0.0, "max": 500.0},
            "balance_db": {"type": "float", "default": 0.0, "min": -12.0, "max": 12.0},
            "auto_balance": {"type": "bool", "default": False},
            "decorrelate_ms": {"type": "float", "default": 0.0, "min": 0.0, "max": 25.0},
        },
    },
    {
        "name": "harmonic_track_suppress",
        "summary": (
            "Frame-wise sweeping fundamental f0(t) harmonic comb suppressor for "
            "variable-RPM leafblowers, trimmers, and shifting engines."
        ),
        "ambient_guidance": (
            "Tracks f0(t) dynamically across 55-300 Hz harmonics with content-defined bird "
            "chirp protection and optional channel-bias excess capping."
        ),
        "params": {
            "start_s": {"type": "float | null", "default": None},
            "end_s": {"type": "float | null", "default": None},
            "f0_min_hz": {"type": "float", "default": 55.0, "min": 30.0, "max": 200.0},
            "f0_max_hz": {"type": "float", "default": 300.0, "min": 80.0, "max": 600.0},
            "max_harmonics": {"type": "int", "default": 12, "min": 3, "max": 20},
            "bandwidth_hz": {"type": "float", "default": 18.0, "min": 6.0, "max": 60.0},
            "strength": {"type": "float", "default": 1.0, "min": 0.1, "max": 1.5},
            "max_attenuation_db": {"type": "float", "default": -28.0, "min": -60.0, "max": -3.0},
            "broadband_assist": {"type": "float", "default": 0.35, "min": 0.0, "max": 1.0},
            "spatial_bias": {"type": "bool", "default": True},
            "protect_birds": {"type": "bool", "default": True},
            "texture_fill": {"type": "float", "default": 0.55, "min": 0.0, "max": 1.0},
            "context_s": {"type": "float", "default": 6.0, "min": 1.0, "max": 20.0},
            "context_gap_s": {"type": "float", "default": 1.5, "min": 0.0, "max": 15.0},
            "fade_ms": {"type": "float", "default": 1000.0, "min": 20.0, "max": 5000.0},
            "n_fft": {"type": "int", "default": 4096, "min": 1024, "max": 8192},
            "spans": {"type": "list[dict] | null", "default": None},
        },
    },
    {
        "name": "distance_soften",
        "summary": (
            "Excess-capped psychoacoustic distance pushback & voice muffling with air-absorption "
            "HF rolloff, onset smearing, and bird chirp protection."
        ),
        "ambient_guidance": (
            "Use mode='distance' or mode='bark_distance' for dog barks and close spikes; "
            "mode='voice_muffle' with formant_blur for speech. Preserves FM bird syllables."
        ),
        "params": {
            "start_s": {"type": "float | null", "default": None},
            "end_s": {"type": "float | null", "default": None},
            "mode": {
                "type": "str",
                "default": "distance",
                "choices": ["distance", "voice_muffle", "bark_distance"],
            },
            "low_hz": {"type": "float", "default": 150.0, "min": 20.0, "max": 4000.0},
            "high_hz": {"type": "float", "default": 6000.0, "min": 200.0, "max": 20000.0},
            "attenuation_db": {"type": "float", "default": -14.0, "min": -36.0, "max": -2.0},
            "air_absorption": {"type": "float", "default": 0.65, "min": 0.0, "max": 1.0},
            "attack_smear_ms": {"type": "float", "default": 12.0, "min": 0.0, "max": 30.0},
            "formant_blur": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "narrow_stereo": {"type": "float", "default": 0.35, "min": 0.0, "max": 1.0},
            "protect_birds": {"type": "bool", "default": True},
            "context_s": {"type": "float", "default": 6.0, "min": 1.0, "max": 20.0},
            "context_gap_s": {"type": "float", "default": 1.5, "min": 0.0, "max": 15.0},
            "fade_ms": {"type": "float", "default": 500.0, "min": 20.0, "max": 5000.0},
            "n_fft": {"type": "int", "default": 2048, "min": 512, "max": 8192},
            "spans": {"type": "list[dict] | null", "default": None},
        },
    },
    {
        "name": "bed_shape",
        "summary": (
            "Slow stationary-bed psychoacoustic comfort shaper (warmth tilt, 2-5 kHz de-harsh, "
            "comfort floor preservation, and optional subtle bed width boost)."
        ),
        "ambient_guidance": (
            "Whole-clip polishing filter beyond normalization. Shapes only the slow stationary "
            "noise bed toward a warmer slope while leaving FM birdsong & transients untouched."
        ),
        "params": {
            "sub_rumble_cut_db": {"type": "float", "default": -6.0, "min": -24.0, "max": 0.0},
            "harshness_dip_db": {"type": "float", "default": -3.0, "min": -12.0, "max": 0.0},
            "warmth_tilt_db": {"type": "float", "default": -1.5, "min": -6.0, "max": 6.0},
            "floor_preserve": {"type": "float", "default": 0.85, "min": 0.3, "max": 1.0},
            "bed_width_boost": {"type": "float", "default": 0.0, "min": -0.5, "max": 0.5},
            "deharsh_dynamic": {"type": "bool", "default": True},
            "protect_birds": {"type": "bool", "default": True},
            "n_fft": {"type": "int", "default": 2048, "min": 512, "max": 8192},
        },
    },
]


def list_filter_specs() -> list[dict[str, Any]]:
    """Return machine-readable filter specifications for MCP clients."""
    return FILTER_SPECS


def _safe_sosfiltfilt(sos: np.ndarray, data: np.ndarray) -> np.ndarray:
    padlen = 3 * (2 * sos.shape[0] + 1)
    if data.shape[0] > padlen:
        return signal.sosfiltfilt(sos, data, axis=0)
    return signal.sosfilt(sos, data, axis=0)


def _rbj_biquad_sos(
    sample_rate: int,
    freq_hz: float,
    gain_db: float,
    q: float,
    filter_type: str,
) -> np.ndarray:
    f0 = max(10.0, min(float(freq_hz), sample_rate * 0.45))
    q_val = max(0.1, min(80.0, float(q)))
    a_lin = 10.0 ** (float(gain_db) / 40.0)
    w0 = 2.0 * math.pi * f0 / float(sample_rate)
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2.0 * q_val)

    ftype = filter_type.strip().lower()
    if ftype == "peaking":
        b0 = 1.0 + alpha * a_lin
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * a_lin
        a0 = 1.0 + alpha / a_lin
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / a_lin
    elif ftype in {"lowshelf", "highshelf"}:
        sqrt_a = math.sqrt(a_lin)
        slope_term = (a_lin + 1.0 / a_lin) * (1.0 / max(q_val, 0.2) - 1.0) + 2.0
        alpha_s = (sin_w0 / 2.0) * math.sqrt(max(slope_term, 1e-6))
        if ftype == "lowshelf":
            b0 = a_lin * ((a_lin + 1.0) - (a_lin - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha_s)
            b1 = 2.0 * a_lin * ((a_lin - 1.0) - (a_lin + 1.0) * cos_w0)
            b2 = a_lin * ((a_lin + 1.0) - (a_lin - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha_s)
            a0 = (a_lin + 1.0) + (a_lin - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha_s
            a1 = -2.0 * ((a_lin - 1.0) + (a_lin + 1.0) * cos_w0)
            a2 = (a_lin + 1.0) + (a_lin - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha_s
        else:
            b0 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha_s)
            b1 = -2.0 * a_lin * ((a_lin - 1.0) + (a_lin + 1.0) * cos_w0)
            b2 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha_s)
            a0 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha_s
            a1 = 2.0 * ((a_lin - 1.0) - (a_lin + 1.0) * cos_w0)
            a2 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha_s
    else:
        raise ValueError(
            f"Unsupported filter_type '{filter_type}'. Use peaking, lowshelf, or highshelf."
        )

    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return signal.tf2sos(b, a)


def _highpass_rumble(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = 60.0,
    order: int = 4,
) -> np.ndarray:
    nyq = sample_rate * 0.5
    fc = max(5.0, min(float(cutoff_hz), nyq * 0.85))
    ord_clamped = max(1, min(int(order), 8))
    sos = signal.butter(ord_clamped, fc, btype="highpass", fs=sample_rate, output="sos")
    return _safe_sosfiltfilt(sos, audio)


def _dehum_notch(
    audio: np.ndarray,
    sample_rate: int,
    base_freq_hz: float = 60.0,
    harmonics: int = 4,
    q: float = 30.0,
) -> np.ndarray:
    out = audio.copy()
    nyq = sample_rate * 0.5
    q_val = max(2.0, min(float(q), 120.0))
    for h in range(1, max(1, int(harmonics)) + 1):
        f0 = float(base_freq_hz) * h
        if f0 >= nyq * 0.9 or f0 <= 5.0:
            break
        b, a = signal.iirnotch(f0, q_val, fs=sample_rate)
        if out.shape[0] > 18:
            out = signal.filtfilt(b, a, out, axis=0)
        else:
            out = signal.lfilter(b, a, out, axis=0)
    return out


def _drone_notch(
    audio: np.ndarray,
    sample_rate: int,
    freqs_hz: list[float],
    q: float = 25.0,
    attenuation_db: float = -24.0,
    harmonics: int = 1,
) -> np.ndarray:
    if not freqs_hz:
        return audio.copy()
    out = audio.copy()
    nyq = sample_rate * 0.5
    q_val = max(2.0, min(float(q), 80.0))
    atten = max(-48.0, min(-0.5, float(attenuation_db)))
    # sosfiltfilt applies magnitude squared (doubles dB); use half-dB per pass when zero-phase
    use_filtfilt = out.shape[0] > 24
    per_pass_db = atten * 0.5 if use_filtfilt else atten
    n_harm = max(1, min(int(harmonics), 6))

    seen: set[float] = set()
    for base in freqs_hz:
        for h in range(1, n_harm + 1):
            f0 = round(float(base) * h, 2)
            if f0 <= 10.0 or f0 >= nyq * 0.9 or f0 in seen:
                continue
            seen.add(f0)
            sos = _rbj_biquad_sos(sample_rate, f0, per_pass_db, q_val, "peaking")
            out = _safe_sosfiltfilt(sos, out)
    return out


def _declick_impulse(
    audio: np.ndarray,
    sample_rate: int,
    threshold_sigma: float = 6.0,
    max_width_ms: float = 3.0,
    start_s: float | None = None,
    end_s: float | None = None,
    spans: list[dict[str, Any]] | None = None,
    method: str = "pchip",
    min_flatness: float = 0.0,
) -> np.ndarray:
    n_samples, n_channels = audio.shape
    if n_samples < 32:
        return audio.copy()

    active_intervals: list[tuple[int, int]] = []
    if spans:
        for sp in spans:
            i0 = max(0, min(n_samples, int(round(float(sp["start_s"]) * sample_rate))))
            i1 = max(0, min(n_samples, int(round(float(sp["end_s"]) * sample_rate))))
            if i1 > i0 + 8:
                active_intervals.append((i0, i1))
        if not active_intervals:
            return audio.copy()
    elif start_s is not None or end_s is not None:
        i0 = max(0, min(n_samples, int(round(float(start_s or 0.0) * sample_rate))))
        i1 = (
            n_samples
            if end_s is None
            else max(0, min(n_samples, int(round(float(end_s) * sample_rate))))
        )
        if i1 <= i0 + 8:
            return audio.copy()
        active_intervals.append((i0, i1))
    else:
        active_intervals.append((0, n_samples))

    out = audio.copy()
    cutoff = min(1500.0, sample_rate * 0.35)
    sos = signal.butter(4, cutoff, btype="highpass", fs=sample_rate, output="sos")
    hp = _safe_sosfiltfilt(sos, out)
    mono = np.mean(out, axis=1)

    max_width_samples = max(4, int(float(max_width_ms) * 0.001 * sample_rate))
    merge_gap = max(1, int(0.001 * sample_rate))
    dilate = max(1, int(0.0005 * sample_rate))
    use_ar = str(method).strip().lower() == "ar"
    min_flat = max(0.0, float(min_flatness))

    for ch in range(n_channels):
        env = np.abs(hp[:, ch])
        med = float(np.median(env))
        mad = float(np.median(np.abs(env - med))) * 1.4826
        sigma = max(mad, 1e-7)
        hit_mask = ((env - med) / sigma > float(threshold_sigma)) & (env > 2e-3)
        if len(active_intervals) > 1 or active_intervals[0] != (0, n_samples):
            roi = np.zeros(n_samples, dtype=bool)
            for a0, a1 in active_intervals:
                roi[a0:a1] = True
            hit_mask &= roi
        hits = np.flatnonzero(hit_mask)
        if hits.size == 0:
            continue

        spans_i: list[tuple[int, int]] = []
        s = e = int(hits[0])
        for idx in hits[1:]:
            i = int(idx)
            if i - e <= merge_gap:
                e = i
            else:
                spans_i.append((s, e))
                s = e = i
        spans_i.append((s, e))

        for s_raw, e_raw in spans_i:
            s_i = max(1, s_raw - dilate)
            e_i = min(n_samples - 2, e_raw + dilate)
            if (e_i - s_i + 1) > max_width_samples:
                continue
            if (
                min_flat > 0.0
                and _hf_patch_flatness(mono, (s_i + e_i) // 2, sample_rate) < min_flat
            ):
                continue
            if use_ar:
                out[s_i : e_i + 1, ch] = ar_interpolate_gap(out[:, ch], s_i, e_i, order=16)
                continue
            left_ctx = np.arange(max(0, s_i - 4), s_i)
            right_ctx = np.arange(e_i + 1, min(n_samples, e_i + 5))
            if left_ctx.size == 0 or right_ctx.size == 0:
                continue
            x_ctx = np.concatenate([left_ctx, right_ctx])
            y_ctx = out[x_ctx, ch]
            x_fill = np.arange(s_i, e_i + 1)
            if x_ctx.size >= 4:
                interp = interpolate.PchipInterpolator(x_ctx, y_ctx)
                out[s_i : e_i + 1, ch] = interp(x_fill)
            else:
                out[s_i : e_i + 1, ch] = np.interp(x_fill, x_ctx, y_ctx)
    return out


def _stft_channels(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], int, int]:
    n_samples, n_channels = audio.shape
    nperseg = max(64, min(int(n_fft), n_samples))
    noverlap = nperseg - max(1, nperseg // 4)
    z_list: list[np.ndarray] = []
    freqs = np.array([])
    times = np.array([])
    for ch in range(n_channels):
        freqs, times, zxx = signal.stft(
            audio[:, ch],
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            boundary="zeros",
            padded=True,
        )
        z_list.append(zxx)
    return freqs, times, z_list, nperseg, noverlap


def _istft_channels(
    z_list: list[np.ndarray],
    sample_rate: int,
    nperseg: int,
    noverlap: int,
    target_len: int,
) -> np.ndarray:
    channels: list[np.ndarray] = []
    for zxx in z_list:
        _, rec = signal.istft(
            zxx,
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=True,
        )
        if rec.size < target_len:
            rec = np.pad(rec, (0, target_len - rec.size))
        channels.append(rec[:target_len])
    return np.column_stack(channels)


def _spectral_denoise(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 0.45,
    floor_db: float = -18.0,
    n_fft: int = 2048,
    noise_start_s: float | None = None,
    noise_end_s: float | None = None,
) -> np.ndarray:
    n_samples, _ = audio.shape
    strength_val = max(0.0, min(1.0, float(strength)))
    if n_samples < 64 or strength_val <= 0.0:
        return audio.copy()

    _, times, z_list, nperseg, noverlap = _stft_channels(audio, sample_rate, n_fft)
    power_mean = np.mean([np.abs(z) ** 2 for z in z_list], axis=0)
    mag_mean = np.sqrt(np.maximum(power_mean, 1e-24))

    if (
        noise_start_s is not None
        and noise_end_s is not None
        and float(noise_end_s) > float(noise_start_s)
    ):
        t_mask = (times >= float(noise_start_s)) & (times <= float(noise_end_s))
        if np.any(t_mask):
            noise_mag = np.median(mag_mean[:, t_mask], axis=1, keepdims=True)
        else:
            noise_mag = np.percentile(mag_mean, 20.0, axis=1, keepdims=True)
    else:
        frame_energy = np.sum(power_mean, axis=0)
        cutoff_e = np.percentile(frame_energy, 25.0)
        quiet_frames = frame_energy <= cutoff_e
        if np.any(quiet_frames):
            noise_mag = np.median(mag_mean[:, quiet_frames], axis=1, keepdims=True)
        else:
            noise_mag = np.percentile(mag_mean, 20.0, axis=1, keepdims=True)

    snr_ratio = (mag_mean / (noise_mag + 1e-12)) ** 2
    oversub = 1.0 + 1.5 * strength_val
    raw_gain = np.maximum(0.0, 1.0 - oversub / np.maximum(snr_ratio, 1e-6))

    floor_lin = 10.0 ** (max(-60.0, min(-1.0, float(floor_db))) / 20.0)
    shaped_gain = floor_lin + (1.0 - floor_lin) * (raw_gain ** 0.75)
    smoothed_gain = ndimage.uniform_filter(shaped_gain, size=(3, 3), mode="nearest")
    final_gain = (1.0 - strength_val) + strength_val * smoothed_gain

    masked = [z * final_gain for z in z_list]
    return _istft_channels(masked, sample_rate, nperseg, noverlap, n_samples)


def _cosine_edge_mask(coords: np.ndarray, start: float, end: float, fade: float) -> np.ndarray:
    """Smooth 1D mask equal to 1 inside [start, end] with raised-cosine transitions."""
    fade_w = max(1e-6, float(fade))
    mask = np.zeros_like(coords, dtype=np.float64)
    inside = (coords >= start) & (coords <= end)
    mask[inside] = 1.0

    left = (coords >= start - fade_w) & (coords < start)
    mask[left] = 0.5 * (1.0 - np.cos(np.pi * (coords[left] - (start - fade_w)) / fade_w))

    right = (coords > end) & (coords <= end + fade_w)
    mask[right] = 0.5 * (1.0 + np.cos(np.pi * (coords[right] - end) / fade_w))
    return mask


def _swell_time_mask(coords: np.ndarray, start: float, end: float, fade: float) -> np.ndarray:
    """Raised-cosine bell mask peaking at span center for slow vehicle passes."""
    s_ext = start - max(0.0, fade) * 0.5
    e_ext = end + max(0.0, fade) * 0.5
    span = max(1e-6, e_ext - s_ext)
    mask = np.zeros_like(coords, dtype=np.float64)
    active = (coords >= s_ext) & (coords <= e_ext)
    phase = (coords[active] - s_ext) / span
    mask[active] = 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))
    return mask


def _resolve_band_spans(
    duration_s: float,
    start_s: float | None,
    end_s: float | None,
    low_hz: float,
    high_hz: float,
    attenuation_db: float,
    fade_ms: float,
    envelope_mode: str,
    spans: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    raw_list = (
        spans
        if spans
        else (
            [{"start_s": start_s, "end_s": end_s}]
            if start_s is not None and end_s is not None
            else []
        )
    )
    resolved: list[dict[str, Any]] = []
    for item in raw_list:
        s_t = max(0.0, float(item["start_s"]))
        e_t = min(duration_s, float(item["end_s"]))
        if e_t <= s_t:
            continue
        resolved.append(
            {
                "start_s": s_t,
                "end_s": e_t,
                "low_hz": max(0.0, float(item.get("low_hz", low_hz))),
                "high_hz": float(item.get("high_hz", high_hz)),
                "attenuation_db": max(
                    -60.0, min(12.0, float(item.get("attenuation_db", attenuation_db)))
                ),
                "fade_ms": max(2.0, min(5000.0, float(item.get("fade_ms", fade_ms)))),
                "envelope_mode": str(item.get("envelope_mode", envelope_mode)).strip().lower(),
            }
        )
    return resolved


def _band_attenuate_with_info(
    audio: np.ndarray,
    sample_rate: int,
    start_s: float | None = None,
    end_s: float | None = None,
    low_hz: float = 40.0,
    high_hz: float = 250.0,
    attenuation_db: float = -18.0,
    fade_ms: float = 30.0,
    envelope_mode: str = "flat",
    n_fft: int = 2048,
    spans: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    n_samples, _ = audio.shape
    dur_s = n_samples / float(max(sample_rate, 1))
    resolved = _resolve_band_spans(
        dur_s, start_s, end_s, low_hz, high_hz, attenuation_db, fade_ms, envelope_mode, spans
    )
    if n_samples < 64 or not resolved:
        return audio.copy(), []

    freqs, times, z_list, nperseg, noverlap = _stft_channels(audio, sample_rate, n_fft)
    gain_total = np.ones((freqs.size, times.size), dtype=np.float64)

    active_ranges: list[tuple[int, int]] = []
    span_logs: list[dict[str, Any]] = []
    for sp in resolved:
        f_lo = sp["low_hz"]
        f_hi = min(sample_rate * 0.5, sp["high_hz"])
        if f_hi <= f_lo:
            continue
        fade_s = sp["fade_ms"] * 0.001
        freq_fade = max(25.0, 0.1 * (f_hi - f_lo))
        w_f = _cosine_edge_mask(freqs, f_lo, f_hi, freq_fade)
        if sp["envelope_mode"] == "swell":
            w_t = _swell_time_mask(times, sp["start_s"], sp["end_s"], fade_s)
            mask_2d = np.outer(w_f, w_t)
            target_lin = 10.0 ** (sp["attenuation_db"] / 20.0)
            gain_sp = 1.0 - mask_2d * (1.0 - target_lin)
        elif sp["envelope_mode"] == "adaptive" and sp["attenuation_db"] < 0.0:
            w_t = _cosine_edge_mask(times, sp["start_s"], sp["end_s"], fade_s)
            band_power_t = np.zeros(times.size, dtype=np.float64)
            wf_sq = (w_f * w_f)[:, np.newaxis]
            for z in z_list:
                band_power_t += np.sum((np.abs(z) ** 2) * wf_sq, axis=0)
            band_db_t = 10.0 * np.log10(np.maximum(band_power_t, 1e-24))
            dt_frame = float(times[1] - times[0]) if times.size > 1 else 0.02
            k_smooth = max(1, int(round(0.25 / max(dt_frame, 1e-4))))
            band_db_smooth = (
                ndimage.uniform_filter1d(band_db_t, size=k_smooth, mode="nearest")
                if k_smooth > 1 and band_db_t.size >= k_smooth
                else band_db_t
            )
            ref_mask = (
                (times >= sp["start_s"] - fade_s - 2.0) & (times < sp["start_s"] - fade_s * 0.25)
            ) | ((times > sp["end_s"] + fade_s * 0.25) & (times <= sp["end_s"] + fade_s + 2.0))
            in_mask = w_t > 0.05
            if np.sum(ref_mask) >= 3:
                ref_db = float(np.median(band_db_smooth[ref_mask]))
            elif np.any(in_mask):
                ref_db = float(np.percentile(band_db_smooth[in_mask], 15))
            else:
                ref_db = float(np.percentile(band_db_smooth, 15))
            excess_db = np.maximum(0.0, band_db_smooth - ref_db)
            cut_db_t = np.minimum(excess_db, -sp["attenuation_db"]) * w_t
            gain_t = 10.0 ** (-cut_db_t / 20.0)
            gain_sp = 1.0 - np.outer(w_f, 1.0 - gain_t)
        else:
            w_t = _cosine_edge_mask(times, sp["start_s"], sp["end_s"], fade_s)
            mask_2d = np.outer(w_f, w_t)
            target_lin = 10.0 ** (sp["attenuation_db"] / 20.0)
            gain_sp = 1.0 - mask_2d * (1.0 - target_lin)
        gain_total *= gain_sp

        active_tf = np.outer(w_f > 0.2, w_t > 0.1)
        if np.any(active_tf):
            cuts = -20.0 * np.log10(np.maximum(gain_sp[active_tf], 1e-6))
            max_cut = round(float(np.max(cuts)), 2)
            mean_cut = round(float(np.mean(cuts)), 2)
        else:
            max_cut = mean_cut = 0.0
        span_logs.append(
            {
                "start_s": round(sp["start_s"], 3),
                "end_s": round(sp["end_s"], 3),
                "low_hz": round(f_lo, 1),
                "high_hz": round(f_hi, 1),
                "envelope_mode": sp["envelope_mode"],
                "max_cut_db": max_cut,
                "mean_cut_db": mean_cut,
            }
        )

        pad_s = fade_s + (nperseg / float(sample_rate))
        i0 = max(0, int((sp["start_s"] - pad_s) * sample_rate))
        i1 = min(n_samples, int((sp["end_s"] + pad_s) * sample_rate))
        active_ranges.append((i0, i1))

    if not active_ranges:
        return audio.copy(), []

    masked = [z * gain_total for z in z_list]
    processed = _istft_channels(masked, sample_rate, nperseg, noverlap, n_samples)

    out = audio.copy()
    for i0, i1 in active_ranges:
        out[i0:i1] = processed[i0:i1]
    return out, span_logs


def _band_attenuate(
    audio: np.ndarray,
    sample_rate: int,
    start_s: float | None = None,
    end_s: float | None = None,
    low_hz: float = 40.0,
    high_hz: float = 250.0,
    attenuation_db: float = -18.0,
    fade_ms: float = 30.0,
    envelope_mode: str = "flat",
    n_fft: int = 2048,
    spans: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    out, _ = _band_attenuate_with_info(
        audio,
        sample_rate,
        start_s=start_s,
        end_s=end_s,
        low_hz=low_hz,
        high_hz=high_hz,
        attenuation_db=attenuation_db,
        fade_ms=fade_ms,
        envelope_mode=envelope_mode,
        n_fft=n_fft,
        spans=spans,
    )
    return out


def _resolve_match_suppress_spans(
    duration_s: float,
    sample_rate: int,
    start_s: float | None,
    end_s: float | None,
    low_hz: float,
    high_hz: float,
    strength: float,
    max_attenuation_db: float,
    texture_fill: float,
    context_s: float,
    context_gap_s: float,
    fade_ms: float,
    n_fft: int,
    ref_start_s: float | None,
    ref_end_s: float | None,
    protect_birds: bool,
    spatial_mode: str,
    narrowband_mode: bool,
    spans: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    raw_list = (
        spans
        if spans
        else (
            [{"start_s": start_s, "end_s": end_s}]
            if start_s is not None and end_s is not None
            else []
        )
    )
    nyq = sample_rate * 0.5
    resolved: list[dict[str, Any]] = []
    for item in raw_list:
        s_t = max(0.0, float(item["start_s"]))
        e_t = min(duration_s, float(item["end_s"]))
        if e_t <= s_t:
            continue
        f_lo = max(15.0, min(12000.0, float(item.get("low_hz", low_hz))))
        f_hi = max(f_lo + 20.0, min(nyq, float(item.get("high_hz", high_hz))))
        resolved.append(
            {
                "start_s": s_t,
                "end_s": e_t,
                "low_hz": f_lo,
                "high_hz": f_hi,
                "strength": max(0.1, min(1.5, float(item.get("strength", strength)))),
                "max_attenuation_db": max(
                    -60.0, min(-3.0, float(item.get("max_attenuation_db", max_attenuation_db)))
                ),
                "texture_fill": max(0.0, min(1.0, float(item.get("texture_fill", texture_fill)))),
                "context_s": max(1.0, min(20.0, float(item.get("context_s", context_s)))),
                "context_gap_s": max(
                    0.0, min(15.0, float(item.get("context_gap_s", context_gap_s)))
                ),
                "fade_ms": max(20.0, min(5000.0, float(item.get("fade_ms", fade_ms)))),
                "n_fft": max(1024, min(8192, int(item.get("n_fft", n_fft)))),
                "ref_start_s": item.get("ref_start_s", ref_start_s),
                "ref_end_s": item.get("ref_end_s", ref_end_s),
                "protect_birds": bool(item.get("protect_birds", protect_birds)),
                "spatial_mode": str(item.get("spatial_mode", spatial_mode)),
                "narrowband_mode": bool(item.get("narrowband_mode", narrowband_mode)),
            }
        )
    return resolved


def _suppress_window_stft(
    audio: np.ndarray,
    sample_rate: int,
    spans: list[dict[str, Any]],
    ref_overrides: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Delegate spectral_match_suppress to the unified excess-capped masked STFT engine."""
    return apply_masked_stft_transform(
        audio, sample_rate, spans, ref_overrides=ref_overrides, mode="spectral_match"
    )


def _spectral_match_suppress_with_info(
    audio: np.ndarray,
    sample_rate: int,
    start_s: float | None = None,
    end_s: float | None = None,
    low_hz: float = 25.0,
    high_hz: float = 4000.0,
    strength: float = 1.0,
    max_attenuation_db: float = -32.0,
    texture_fill: float = 0.65,
    context_s: float = 6.0,
    context_gap_s: float = 1.5,
    fade_ms: float = 1500.0,
    n_fft: int = 4096,
    ref_start_s: float | None = None,
    ref_end_s: float | None = None,
    protect_birds: bool = False,
    spatial_mode: str = "linked",
    narrowband_mode: bool = False,
    spans: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    dur_s = audio.shape[0] / float(max(sample_rate, 1))
    resolved = _resolve_match_suppress_spans(
        dur_s,
        sample_rate,
        start_s,
        end_s,
        low_hz,
        high_hz,
        strength,
        max_attenuation_db,
        texture_fill,
        context_s,
        context_gap_s,
        fade_ms,
        n_fft,
        ref_start_s,
        ref_end_s,
        protect_birds,
        spatial_mode,
        narrowband_mode,
        spans,
    )
    if not resolved:
        return audio.copy(), []
    return _suppress_window_stft(audio, sample_rate, resolved)


def _spectral_match_suppress(
    audio: np.ndarray,
    sample_rate: int,
    start_s: float | None = None,
    end_s: float | None = None,
    low_hz: float = 25.0,
    high_hz: float = 4000.0,
    strength: float = 1.0,
    max_attenuation_db: float = -32.0,
    texture_fill: float = 0.65,
    context_s: float = 6.0,
    context_gap_s: float = 1.5,
    fade_ms: float = 1500.0,
    n_fft: int = 4096,
    ref_start_s: float | None = None,
    ref_end_s: float | None = None,
    protect_birds: bool = False,
    spatial_mode: str = "linked",
    narrowband_mode: bool = False,
    spans: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    out, _ = _spectral_match_suppress_with_info(
        audio,
        sample_rate,
        start_s=start_s,
        end_s=end_s,
        low_hz=low_hz,
        high_hz=high_hz,
        strength=strength,
        max_attenuation_db=max_attenuation_db,
        texture_fill=texture_fill,
        context_s=context_s,
        context_gap_s=context_gap_s,
        fade_ms=fade_ms,
        n_fft=n_fft,
        ref_start_s=ref_start_s,
        ref_end_s=ref_end_s,
        protect_birds=protect_birds,
        spatial_mode=spatial_mode,
        narrowband_mode=narrowband_mode,
        spans=spans,
    )
    return out


def _build_tiled_donor_with_info(
    donor: np.ndarray, target_len: int, xfade: int
) -> tuple[np.ndarray, int]:
    """Create chunk of length target_len by crossfade-tiling donor; return (arr, loop_count)."""
    if donor.shape[0] >= target_len:
        return donor[:target_len].copy(), 0

    n_channels = donor.shape[1]
    if donor.shape[0] <= 2 * max(xfade, 1):
        reps = math.ceil(target_len / max(donor.shape[0], 1))
        return np.tile(donor, (reps, 1))[:target_len], max(1, reps - 1)

    xf = min(xfade, donor.shape[0] // 4)
    step = donor.shape[0] - xf
    out = np.zeros((target_len + donor.shape[0], n_channels), dtype=np.float64)

    t_xf = np.linspace(0.0, math.pi * 0.5, xf, endpoint=True)[:, np.newaxis]
    fade_in = np.sin(t_xf)
    fade_out = np.cos(t_xf)

    pos = 0
    tiles = 0
    first = True
    while pos < target_len:
        chunk = donor.copy()
        if not first and xf > 0:
            out[pos : pos + xf] = out[pos : pos + xf] * fade_out + chunk[:xf] * fade_in
            out[pos + xf : pos + donor.shape[0]] = chunk[xf:]
        else:
            out[pos : pos + donor.shape[0]] = chunk
            first = False
        pos += step
        tiles += 1
    return out[:target_len], max(0, tiles - 1)


def _merge_sample_intervals(
    spans: list[tuple[int, int]], total_samples: int
) -> list[tuple[int, int]]:
    """Merge sorted/unsorted sample intervals clamped to [0, total_samples]."""
    cleaned = sorted(
        (max(0, int(s)), min(total_samples, int(e))) for s, e in spans if int(e) > int(s)
    )
    if not cleaned:
        return []
    merged = [cleaned[0]]
    for s, e in cleaned[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _complement_sample_intervals(
    exclude_spans: list[tuple[int, int]], total_samples: int, min_len: int
) -> list[tuple[int, int]]:
    """Return complementary free intervals >= min_len outside exclude_spans."""
    merged = _merge_sample_intervals(exclude_spans, total_samples)
    free: list[tuple[int, int]] = []
    cur = 0
    for s, e in merged:
        if s - cur >= min_len:
            free.append((cur, s))
        cur = max(cur, e)
    if total_samples - cur >= min_len:
        free.append((cur, total_samples))
    return free


def _intervals_overlap_seconds(
    s0_s: float, e0_s: float, events_s: list[tuple[float, float]] | None
) -> bool:
    if not events_s:
        return False
    return any(ee > s0_s and es < e0_s for es, ee in events_s)


def _donor_quality_report(
    donor: np.ndarray,
    ctx_rms: float | None,
    d_s0: float,
    d_s1: float,
    raw_events_s: list[tuple[float, float]] | None,
) -> dict[str, Any]:
    mono = np.mean(ensure_2d_audio(donor), axis=1) if donor.size else np.zeros(0)
    rms = float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0
    pk = float(np.max(np.abs(mono))) if mono.size else 0.0
    crest = (
        round(pk / max(rms, 1e-12), 2)
        if rms > 1e-12 and math.isfinite(pk) and math.isfinite(rms)
        else None
    )
    rms_diff = (
        round(20.0 * math.log10(max(rms, 1e-12) / max(ctx_rms, 1e-12)), 2)
        if (
            ctx_rms
            and ctx_rms > 1e-12
            and rms > 1e-12
            and math.isfinite(ctx_rms)
            and math.isfinite(rms)
        )
        else 0.0
    )
    return {
        "crest_factor": crest if crest is not None and math.isfinite(crest) else None,
        "rms_diff_db": rms_diff if math.isfinite(rms_diff) else 0.0,
        "overlaps_event": _intervals_overlap_seconds(d_s0, d_s1, raw_events_s),
    }


def _match_donor_rms_and_tilt(
    donor: np.ndarray, ctx: np.ndarray | None, sample_rate: int
) -> np.ndarray:
    """Match overall RMS (within [0.5, 2.0]) and subtle 2-band low/high tilt against context."""
    out = donor.copy()
    if ctx is None or ctx.shape[0] < 32 or out.shape[0] < 32:
        return out
    ctx_rms = float(np.sqrt(np.mean(ctx * ctx)))
    d_rms = float(np.sqrt(np.mean(out * out)))
    if d_rms > 1e-9 and ctx_rms > 1e-9:
        out = out * max(0.5, min(2.0, ctx_rms / d_rms))
    if sample_rate >= 1000 and min(ctx.shape[0], out.shape[0]) >= 64:
        sos_lp = signal.butter(2, 250.0, btype="lowpass", fs=sample_rate, output="sos")
        c_low = _safe_sosfiltfilt(sos_lp, ctx)
        d_low = _safe_sosfiltfilt(sos_lp, out)
        c_hi = ctx - c_low
        d_hi = out - d_low
        cl_r = float(np.sqrt(np.mean(c_low * c_low))) + 1e-12
        ch_r = float(np.sqrt(np.mean(c_hi * c_hi))) + 1e-12
        dl_r = float(np.sqrt(np.mean(d_low * d_low))) + 1e-12
        dh_r = float(np.sqrt(np.mean(d_hi * d_hi))) + 1e-12
        s_lo = max(0.707, min(1.414, cl_r / dl_r))
        s_hi = max(0.707, min(1.414, ch_r / dh_r))
        blended = d_low * s_lo + d_hi * s_hi
        b_rms = float(np.sqrt(np.mean(blended * blended))) + 1e-12
        if ctx_rms > 1e-9:
            blended *= max(0.5, min(2.0, ctx_rms / b_rms))
        return blended
    return out


def _score_donor_windows_from_stats(
    free_spans: list[tuple[int, int]],
    win_frames: int,
    win_rms: np.ndarray,
    win_peak: np.ndarray,
    win_low_ratio: np.ndarray,
    target_len: int,
    target_mid_frame: float,
    target_rms: float,
    sample_rate: int,
    max_donor_crest: float = 6.0,
) -> tuple[int, int, int] | None:
    """Select optimal contiguous window [b0, b1] avoiding high-crest knocks."""
    total_w = win_rms.size
    if total_w == 0 or not free_spans:
        return None
    best_score = float("inf")
    best_span: tuple[int, int, int] | None = None
    k_target = max(1, int(math.ceil(target_len / float(win_frames))))
    ref_rms = max(float(target_rms), 1e-6)
    norm_600s = max(float(sample_rate) * 600.0, 1.0)
    crest_cap = max(2.0, float(max_donor_crest))

    for fs0, fs1 in free_spans:
        w0 = (fs0 + win_frames - 1) // win_frames
        w1 = min(total_w, fs1 // win_frames)
        if w1 <= w0:
            continue
        avail_k = w1 - w0
        k_use = min(avail_k, k_target)
        seg_rms = win_rms[w0:w1]
        seg_pk = win_peak[w0:w1]
        seg_low = win_low_ratio[w0:w1]
        if k_use == 1:
            mean_rms = seg_rms
            std_rms = np.zeros_like(seg_rms)
            max_pk = seg_pk
            max_block_crest = seg_pk / np.maximum(seg_rms, 1e-9)
            mean_low = seg_low
        else:
            cs1 = np.concatenate([[0.0], np.cumsum(seg_rms)])
            cs2 = np.concatenate([[0.0], np.cumsum(seg_rms * seg_rms)])
            csl = np.concatenate([[0.0], np.cumsum(seg_low)])
            mean_rms = (cs1[k_use:] - cs1[:-k_use]) / k_use
            var_rms = np.maximum(0.0, (cs2[k_use:] - cs2[:-k_use]) / k_use - mean_rms * mean_rms)
            std_rms = np.sqrt(var_rms)
            mean_low = (csl[k_use:] - csl[:-k_use]) / k_use
            max_pk = np.lib.stride_tricks.sliding_window_view(seg_pk, k_use).max(axis=1)
            win_c = seg_pk / np.maximum(seg_rms, 1e-9)
            max_block_crest = np.lib.stride_tricks.sliding_window_view(win_c, k_use).max(axis=1)

        rms_err = np.abs(mean_rms - ref_rms) / ref_rms
        cv = std_rms / np.maximum(mean_rms, 1e-9)
        crest = max_pk / np.maximum(mean_rms, 1e-9)
        knock_pen = np.where((crest > crest_cap) | (max_block_crest > crest_cap), 120.0, 0.0)
        starts_f = (w0 + np.arange(mean_rms.size)) * win_frames
        mids_f = starts_f + 0.5 * k_use * win_frames
        prox = np.minimum(1.0, np.abs(mids_f - target_mid_frame) / norm_600s)
        short_pen = 0.4 * max(0.0, 1.0 - (k_use / float(max(1, k_target))))
        scores = (
            2.5 * rms_err
            + 0.6 * cv
            + 0.35 * crest
            + 0.25 * mean_low
            + 0.08 * prox
            + short_pen
            + knock_pen
        )
        loc_idx = int(np.argmin(scores))
        sc = float(scores[loc_idx])
        if sc < best_score:
            best_score = sc
            b0 = int(starts_f[loc_idx])
            b1 = min(fs1, b0 + min(target_len, k_use * win_frames))
            best_span = (b0, b1, k_use)
    return best_span


def _build_window_stats_array(
    audio: np.ndarray, sample_rate: int, win_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-block RMS, peak, and sub-120 Hz energy ratio in O(N)."""
    mono = np.mean(ensure_2d_audio(audio), axis=1)
    n_w = mono.size // win_frames
    if n_w <= 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    reshaped = mono[: n_w * win_frames].reshape(n_w, win_frames)
    w_rms = np.sqrt(np.mean(reshaped * reshaped, axis=1))
    w_pk = np.max(np.abs(reshaped), axis=1)
    if sample_rate >= 400 and mono.size >= 64:
        sos_low = signal.butter(
            2, min(120.0, sample_rate * 0.35), btype="lowpass", fs=sample_rate, output="sos"
        )
        low_m = signal.sosfilt(sos_low, mono)[: n_w * win_frames].reshape(n_w, win_frames)
        w_low_rms = np.sqrt(np.mean(low_m * low_m, axis=1))
        w_low_ratio = w_low_rms / np.maximum(w_rms, 1e-9)
    else:
        w_low_ratio = np.zeros(n_w, dtype=np.float64)
    return w_rms, w_pk, w_low_ratio


def _stitch_non_repeating_chunks(
    chunks: list[np.ndarray], target_len: int, xfade: int
) -> tuple[np.ndarray, int]:
    """Combine distinct donor chunks up to target_len; return (array, loop_count)."""
    if not chunks:
        return np.zeros((target_len, 2), dtype=np.float64), 0
    if len(chunks) == 1:
        return _build_tiled_donor_with_info(chunks[0], target_len, xfade)
    out = chunks[0].copy()
    for seg in chunks[1:]:
        xf = min(xfade, out.shape[0] // 4, seg.shape[0] // 4)
        if xf > 0:
            t = np.linspace(0.0, math.pi * 0.5, xf, endpoint=True)[:, np.newaxis]
            joined = out[-xf:] * np.cos(t) + seg[:xf] * np.sin(t)
            out = np.vstack([out[:-xf], joined, seg[xf:]])
        else:
            out = np.vstack([out, seg])
        if out.shape[0] >= target_len:
            return out[:target_len].copy(), 0
    return _build_tiled_donor_with_info(out, target_len, xfade)


def _pick_clean_donor_with_span(
    audio: np.ndarray,
    sample_rate: int,
    exclude_spans: list[tuple[int, int]],
    target_len: int,
    xfade: int,
    exclude_events_s: list[tuple[float, float]] | None = None,
    used_spans: list[tuple[int, int]] | None = None,
    target_mid_frame: float | None = None,
    target_rms: float | None = None,
    stats_cache: tuple[int, np.ndarray, np.ndarray, np.ndarray] | None = None,
    min_donor_distance_s: float = 5.0,
    max_donor_crest: float = 6.0,
) -> tuple[np.ndarray, float, float, int, int]:
    """Select an unrepeated, event- and knock-screened clean donor segment."""
    n_samples, n_ch = audio.shape
    patch_margin = int(0.15 * sample_rate)
    dist_margin = max(patch_margin, int(max(0.0, float(min_donor_distance_s)) * sample_rate))
    event_margin = int(5.0 * sample_rate)
    win_frames = stats_cache[0] if stats_cache else max(64, int(round(0.25 * sample_rate)))
    min_region = max(win_frames * 2, int(0.25 * sample_rate))

    ex_patch = [(s - patch_margin, e + patch_margin) for s, e in exclude_spans]
    ex_dist = [(s - dist_margin, e + dist_margin) for s, e in exclude_spans]
    ex_used = [(s - patch_margin, e + patch_margin) for s, e in (used_spans or [])]
    ex_events = [
        (int(round(es * sample_rate)) - event_margin, int(round(ee * sample_rate)) + event_margin)
        for es, ee in (exclude_events_s or [])
    ]

    free_spans = _complement_sample_intervals(ex_dist + ex_used + ex_events, n_samples, min_region)
    if not free_spans:
        free_spans = _complement_sample_intervals(
            ex_patch + ex_used + ex_events, n_samples, min_region
        )
    if not free_spans:
        free_spans = _complement_sample_intervals(ex_patch + ex_used, n_samples, min_region)
    if not free_spans:
        free_spans = _complement_sample_intervals(ex_patch, n_samples, min_region)
    if not free_spans:
        ex0, ex1 = exclude_spans[0] if exclude_spans else (0, n_samples)
        fb0, fb1 = (0, ex0) if ex0 >= (n_samples - ex1) else (ex1, n_samples)
        fallback = audio[fb0:fb1] if fb1 > fb0 else audio
        if fallback.shape[0] == 0:
            return np.zeros((target_len, n_ch), dtype=np.float64), 0.0, 0.0, 0, 1
        tiled, loops = _build_tiled_donor_with_info(fallback, target_len, xfade)
        return (
            tiled,
            round(fb0 / float(sample_rate), 3),
            round(fb1 / float(sample_rate), 3),
            loops,
            1,
        )

    if stats_cache is None:
        w_rms, w_pk, w_low = _build_window_stats_array(audio, sample_rate, win_frames)
    else:
        _, w_rms, w_pk, w_low = stats_cache

    ref_rms = (
        float(target_rms)
        if target_rms and target_rms > 1e-9
        else (float(np.median(w_rms)) if w_rms.size else 0.01)
    )
    mid_f = float(target_mid_frame) if target_mid_frame is not None else 0.5 * n_samples

    picked = _score_donor_windows_from_stats(
        free_spans,
        win_frames,
        w_rms,
        w_pk,
        w_low,
        target_len,
        mid_f,
        ref_rms,
        sample_rate,
        max_donor_crest=max_donor_crest,
    )
    if picked is None:
        fs0, fs1 = free_spans[0]
        b0, b1 = fs0, min(fs1, fs0 + target_len)
    else:
        b0, b1, _ = picked

    if used_spans is not None:
        used_spans.append((b0, b1))

    if b1 - b0 >= target_len:
        return (
            audio[b0 : b0 + target_len].copy(),
            round(b0 / float(sample_rate), 3),
            round((b0 + target_len) / float(sample_rate), 3),
            0,
            1,
        )

    chunks = [audio[b0:b1]]
    got = b1 - b0
    local_used = list(used_spans or [])
    if (b0, b1) not in local_used:
        local_used.append((b0, b1))
    while got < target_len:
        cur_free = _complement_sample_intervals(
            ex_dist + [(s - patch_margin, e + patch_margin) for s, e in local_used] + ex_events,
            n_samples,
            min_region,
        )
        if not cur_free:
            cur_free = _complement_sample_intervals(
                ex_patch
                + [(s - patch_margin, e + patch_margin) for s, e in local_used]
                + ex_events,
                n_samples,
                min_region,
            )
        if not cur_free:
            cur_free = _complement_sample_intervals(
                ex_patch + [(s - patch_margin, e + patch_margin) for s, e in local_used],
                n_samples,
                min_region,
            )
        if not cur_free:
            break
        nxt = _score_donor_windows_from_stats(
            cur_free,
            win_frames,
            w_rms,
            w_pk,
            w_low,
            target_len - got,
            mid_f,
            ref_rms,
            sample_rate,
            max_donor_crest=max_donor_crest,
        )
        if nxt is None:
            break
        nb0, nb1, _ = nxt
        chunks.append(audio[nb0:nb1])
        local_used.append((nb0, nb1))
        if used_spans is not None:
            used_spans.append((nb0, nb1))
        got += nb1 - nb0

    assembled, loops = _stitch_non_repeating_chunks(chunks, target_len, xfade)
    return (
        assembled,
        round(b0 / float(sample_rate), 3),
        round(b1 / float(sample_rate), 3),
        loops,
        len(chunks),
    )


def _apply_patch_slice(
    out: np.ndarray,
    ref_audio: np.ndarray,
    sample_rate: int,
    p0: int,
    p1: int,
    xf_l: int,
    xf_r: int,
    donor: np.ndarray,
) -> tuple[np.ndarray, float | None]:
    """Splice donor into out[p0:p1] with RMS + 2-band tilt match and equal-power crossfades."""
    n_samples = ref_audio.shape[0]
    ctx_len = int(0.5 * sample_rate)
    ctx_parts = []
    if p0 > 16:
        ctx_parts.append(ref_audio[max(0, p0 - ctx_len) : p0])
    if n_samples - p1 > 16:
        ctx_parts.append(ref_audio[p1 : min(n_samples, p1 + ctx_len)])
    ctx = np.vstack(ctx_parts) if ctx_parts else None
    ctx_rms = float(np.sqrt(np.mean(ctx * ctx))) if ctx is not None and ctx.size else None
    matched = _match_donor_rms_and_tilt(donor, ctx, sample_rate)

    blended = matched.copy()
    if xf_l > 0:
        t_l = np.linspace(0.0, math.pi * 0.5, xf_l, endpoint=True)[:, np.newaxis]
        blended[:xf_l] = out[p0 : p0 + xf_l] * np.cos(t_l) + matched[:xf_l] * np.sin(t_l)
    if xf_r > 0:
        t_r = np.linspace(0.0, math.pi * 0.5, xf_r, endpoint=True)[:, np.newaxis]
        blended[-xf_r:] = matched[-xf_r:] * np.cos(t_r) + out[p1 - xf_r : p1] * np.sin(t_r)

    out[p0:p1] = blended
    return matched, ctx_rms


def _resolve_patch_spans(
    n_samples: int,
    sample_rate: int,
    start_s: float | None,
    end_s: float | None,
    source_start_s: float | None,
    crossfade_ms: float,
    spans: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    raw_list = (
        spans
        if spans
        else (
            [{"start_s": start_s, "end_s": end_s}]
            if start_s is not None and end_s is not None
            else []
        )
    )
    resolved: list[dict[str, Any]] = []
    for item in raw_list:
        i0 = max(0, min(n_samples, int(round(float(item["start_s"]) * sample_rate))))
        i1 = max(0, min(n_samples, int(round(float(item["end_s"]) * sample_rate))))
        if i1 <= i0:
            continue
        xf_ms = max(5.0, min(5000.0, float(item.get("crossfade_ms", crossfade_ms))))
        desired_xf = max(1, int(xf_ms * 0.001 * sample_rate))
        xf_l = min(desired_xf, i0, max(1, (i1 - i0) // 2))
        xf_r = min(desired_xf, n_samples - i1, max(1, (i1 - i0) // 2))
        p0 = max(0, i0 - xf_l // 2)
        p1 = min(n_samples, i1 + xf_r // 2)
        resolved.append(
            {
                "p0": p0,
                "p1": p1,
                "xf_l": xf_l,
                "xf_r": xf_r,
                "desired_xf": desired_xf,
                "crossfade_ms": xf_ms,
                "source_start_s": item.get("source_start_s", source_start_s),
            }
        )
    resolved.sort(key=lambda r: r["p0"])
    return resolved


def _ambience_patch_with_info(
    audio: np.ndarray,
    sample_rate: int,
    start_s: float | None = None,
    end_s: float | None = None,
    source_start_s: float | None = None,
    crossfade_ms: float = 40.0,
    min_donor_distance_s: float = 5.0,
    max_donor_crest: float = 6.0,
    spans: list[dict[str, Any]] | None = None,
    exclude_events: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    n_samples, _ = audio.shape
    if n_samples < 32:
        return audio.copy(), []
    resolved = _resolve_patch_spans(
        n_samples, sample_rate, start_s, end_s, source_start_s, crossfade_ms, spans
    )
    if not resolved:
        return audio.copy(), []

    exclude_spans = [(r["p0"], r["p1"]) for r in resolved]
    used_spans: list[tuple[int, int]] = []
    win_frames = max(64, int(round(0.25 * sample_rate)))
    stats_cache = (win_frames, *_build_window_stats_array(audio, sample_rate, win_frames))
    out = audio.copy()
    patch_log: list[dict[str, Any]] = []

    for r in resolved:
        p0, p1 = r["p0"], r["p1"]
        patch_len = p1 - p0
        ctx_n = int(0.5 * sample_rate)
        n_ch = audio.shape[1]
        ctx_l = audio[max(0, p0 - ctx_n) : p0] if p0 > 16 else np.zeros((0, n_ch))
        ctx_r = (
            audio[p1 : min(n_samples, p1 + ctx_n)]
            if n_samples - p1 > 16
            else np.zeros((0, n_ch))
        )
        ctx_parts = [c for c in (ctx_l, ctx_r) if c.shape[0] > 16]
        t_rms = float(np.sqrt(np.mean(np.vstack(ctx_parts) ** 2))) if ctx_parts else None

        if r["source_start_s"] is not None:
            s_val = float(r["source_start_s"])
            src_i0 = max(0, min(n_samples - 1, int(round(s_val * sample_rate))))
            src_i1 = min(n_samples, src_i0 + patch_len)
            raw_donor = audio[src_i0:src_i1]
            if raw_donor.shape[0] == 0:
                raw_donor = audio
            donor, loops = _build_tiled_donor_with_info(raw_donor, patch_len, r["desired_xf"])
            donor_s0 = round(src_i0 / float(sample_rate), 3)
            donor_s1 = round(src_i1 / float(sample_rate), 3)
            donor_chunks = 1
            src_mode = "manual"
        else:
            donor, donor_s0, donor_s1, loops, donor_chunks = _pick_clean_donor_with_span(
                audio,
                sample_rate,
                exclude_spans,
                patch_len,
                r["desired_xf"],
                exclude_events_s=exclude_events,
                used_spans=used_spans,
                target_mid_frame=0.5 * (p0 + p1),
                target_rms=t_rms,
                stats_cache=stats_cache,
                min_donor_distance_s=min_donor_distance_s,
                max_donor_crest=max_donor_crest,
            )
            src_mode = "auto"

        matched_donor, ctx_rms = _apply_patch_slice(
            out, audio, sample_rate, p0, p1, r["xf_l"], r["xf_r"], donor
        )
        quality = _donor_quality_report(
            matched_donor, ctx_rms, donor_s0, donor_s1, exclude_events
        )
        patch_log.append(
            {
                "target_start_s": round(p0 / float(sample_rate), 3),
                "target_end_s": round(p1 / float(sample_rate), 3),
                "donor_start_s": donor_s0,
                "donor_end_s": donor_s1,
                "source_mode": src_mode,
                "crossfade_ms": round(float(r.get("crossfade_ms", crossfade_ms)), 1),
                "loop_count": int(loops),
                "donor_chunks": int(donor_chunks),
                "donor_quality": quality,
            }
        )
    return out, patch_log


def _ambience_patch(
    audio: np.ndarray,
    sample_rate: int,
    start_s: float | None = None,
    end_s: float | None = None,
    source_start_s: float | None = None,
    crossfade_ms: float = 40.0,
    min_donor_distance_s: float = 5.0,
    max_donor_crest: float = 6.0,
    spans: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    out, _ = _ambience_patch_with_info(
        audio,
        sample_rate,
        start_s=start_s,
        end_s=end_s,
        source_start_s=source_start_s,
        crossfade_ms=crossfade_ms,
        min_donor_distance_s=min_donor_distance_s,
        max_donor_crest=max_donor_crest,
        spans=spans,
    )
    return out


def _trim_range(
    audio: np.ndarray,
    sample_rate: int,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> np.ndarray:
    n_samples, _ = audio.shape
    i0 = max(0, min(n_samples, int(round(max(0.0, float(start_s)) * sample_rate))))
    i1 = (
        n_samples
        if end_s is None
        else max(i0, min(n_samples, int(round(float(end_s) * sample_rate))))
    )
    return audio[i0:i1].copy()


def _compute_kept_intervals(
    n_samples: int,
    sample_rate: int,
    regions: list[dict[str, float]],
) -> list[tuple[int, int]]:
    cuts: list[tuple[int, int]] = []
    for reg in regions:
        s = max(0, min(n_samples, int(round(float(reg["start_s"]) * sample_rate))))
        e = max(0, min(n_samples, int(round(float(reg["end_s"]) * sample_rate))))
        if e > s:
            cuts.append((s, e))
    cuts.sort()

    merged: list[tuple[int, int]] = []
    for s, e in cuts:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    kept: list[tuple[int, int]] = []
    cur = 0
    for s, e in merged:
        if s > cur:
            kept.append((cur, s))
        cur = max(cur, e)
    if cur < n_samples:
        kept.append((cur, n_samples))
    return kept


def _cut_regions(
    audio: np.ndarray,
    sample_rate: int,
    regions: list[dict[str, float]],
    crossfade_ms: float = 1000.0,
) -> np.ndarray:
    n_samples, n_channels = audio.shape
    if not regions or n_samples == 0:
        return audio.copy()
    kept = _compute_kept_intervals(n_samples, sample_rate, regions)
    if not kept:
        return np.zeros((0, n_channels), dtype=np.float64)

    xf_desired = max(1, int(max(10.0, min(5000.0, float(crossfade_ms))) * 0.001 * sample_rate))
    out = audio[kept[0][0] : kept[0][1]].copy()
    for ks, ke in kept[1:]:
        seg = audio[ks:ke].copy()
        xf = min(xf_desired, out.shape[0] // 2, seg.shape[0] // 2)
        if xf > 0:
            t = np.linspace(0.0, math.pi * 0.5, xf, endpoint=True)[:, np.newaxis]
            joined = out[-xf:] * np.cos(t) + seg[:xf] * np.sin(t)
            out = np.vstack([out[:-xf], joined, seg[xf:]])
        else:
            out = np.vstack([out, seg])
    return out


def _fade_curve_vec(length: int, curve: str, fade_in: bool) -> np.ndarray:
    t = np.linspace(0.0, 1.0, length, endpoint=True)
    if not fade_in:
        t = 1.0 - t
    if curve.strip().lower() == "linear":
        return t[:, np.newaxis]
    return (0.5 * (1.0 - np.cos(np.pi * t)))[:, np.newaxis]


def _fade_edges(
    audio: np.ndarray,
    sample_rate: int,
    fade_in_s: float = 2.0,
    fade_out_s: float = 3.0,
    curve: str = "cosine",
) -> np.ndarray:
    n_samples, _ = audio.shape
    if n_samples == 0:
        return audio.copy()
    out = audio.copy()
    n_in = min(n_samples, max(0, int(round(float(fade_in_s) * sample_rate))))
    n_out = min(n_samples, max(0, int(round(float(fade_out_s) * sample_rate))))
    if n_in > 0:
        out[:n_in] *= _fade_curve_vec(n_in, curve, fade_in=True)
    if n_out > 0:
        out[n_samples - n_out :] *= _fade_curve_vec(n_out, curve, fade_in=False)
    return out


def _parametric_eq(
    audio: np.ndarray,
    sample_rate: int,
    freq_hz: float,
    gain_db: float,
    q: float = 1.4,
    filter_type: str = "peaking",
) -> np.ndarray:
    """RBJ Audio EQ Cookbook biquad EQ applied as second-order sections."""
    gain = max(-24.0, min(24.0, float(gain_db)))
    if abs(gain) < 1e-4:
        return audio.copy()
    sos = _rbj_biquad_sos(sample_rate, freq_hz, gain, q, filter_type)
    return signal.sosfilt(sos, audio, axis=0)


def _soft_compress(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -24.0,
    ratio: float = 2.0,
    attack_ms: float = 25.0,
    release_ms: float = 250.0,
    makeup_db: float = 0.0,
) -> np.ndarray:
    n_samples, _ = audio.shape
    if n_samples < 16:
        return audio.copy()

    ratio_val = max(1.0, min(10.0, float(ratio)))
    thresh = float(threshold_db)
    knee_db = 6.0
    half_knee = knee_db * 0.5
    makeup_lin = 10.0 ** (max(-12.0, min(24.0, float(makeup_db))) / 20.0)

    step = max(1, int(round(0.01 * sample_rate)))  # 10 ms control frames
    n_frames = max(1, math.ceil(n_samples / step))
    pad_len = n_frames * step - n_samples
    padded = np.pad(audio, ((0, pad_len), (0, 0))) if pad_len > 0 else audio
    frames = padded.reshape(n_frames, step, audio.shape[1])
    frame_rms = np.sqrt(np.mean(frames * frames, axis=(1, 2)))

    dt = step / float(sample_rate)
    alpha_att = 1.0 - math.exp(-dt / max(0.001, float(attack_ms) * 0.001))
    alpha_rel = 1.0 - math.exp(-dt / max(0.005, float(release_ms) * 0.001))

    smooth_env = np.empty(n_frames, dtype=np.float64)
    cur = float(frame_rms[0])
    for i in range(n_frames):
        x = float(frame_rms[i])
        alpha = alpha_att if x > cur else alpha_rel
        cur += alpha * (x - cur)
        smooth_env[i] = cur

    env_db = 20.0 * np.log10(np.maximum(smooth_env, 1e-9))
    gain_db = np.zeros_like(env_db)
    over = env_db - thresh
    slope = 1.0 - 1.0 / ratio_val

    hard = over >= half_knee
    soft = (over > -half_knee) & (over < half_knee)
    gain_db[hard] = -slope * over[hard]
    gain_db[soft] = -slope * ((over[soft] + half_knee) ** 2) / (2.0 * knee_db)

    frame_gains = (10.0 ** (gain_db / 20.0)) * makeup_lin
    x_frames = (np.arange(n_frames, dtype=np.float64) + 0.5) * step
    x_samples = np.arange(n_samples, dtype=np.float64)
    sample_gains = np.interp(x_samples, x_frames, frame_gains)
    return audio * sample_gains[:, np.newaxis]


def _true_peak_limiter(
    audio: np.ndarray,
    sample_rate: int,
    ceiling_dbtp: float = -1.0,
    release_ms: float = 80.0,
) -> np.ndarray:
    n_samples, _ = audio.shape
    if n_samples == 0:
        return audio.copy()

    ceiling_lin = 10.0 ** (max(-6.0, min(-0.05, float(ceiling_dbtp))) / 20.0)
    # Slight safety margin handles inter-sample overshoots cleanly
    target_lin = ceiling_lin * 0.985

    step = max(1, int(round(0.002 * sample_rate)))  # 2 ms control frames
    n_frames = max(1, math.ceil(n_samples / step))
    pad_len = n_frames * step - n_samples
    padded = np.pad(audio, ((0, pad_len), (0, 0))) if pad_len > 0 else audio
    frames = padded.reshape(n_frames, step, audio.shape[1])
    frame_peaks = np.max(np.abs(frames), axis=(1, 2))

    # Include 4x oversampled per-frame peaks so transient inter-sample overshoots
    # are absorbed locally rather than triggering a global gain reduction later.
    if padded.shape[0] >= 4:
        chunk_frames = 512
        pad_f = 8
        up_peaks = np.empty(n_frames, dtype=np.float64)
        for f0 in range(0, n_frames, chunk_frames):
            f1 = min(n_frames, f0 + chunk_frames)
            s_a = max(0, (f0 - pad_f) * step)
            s_b = min(padded.shape[0], (f1 + pad_f) * step)
            up = signal.resample_poly(padded[s_a:s_b], 4, 1, axis=0)
            off = (f0 * step - s_a) * 4
            cnt = (f1 - f0) * step * 4
            seg_up = up[off : off + cnt].reshape(f1 - f0, step * 4, audio.shape[1])
            up_peaks[f0:f1] = np.max(np.abs(seg_up), axis=(1, 2))
        frame_peaks = np.maximum(frame_peaks, up_peaks)

    lookahead_frames = max(2, int(round(0.006 / 0.002)))
    peak_win = ndimage.maximum_filter1d(frame_peaks, size=lookahead_frames, mode="nearest")

    dt = step / float(sample_rate)
    decay = math.exp(-dt / max(0.005, float(release_ms) * 0.001))
    held = np.empty(n_frames, dtype=np.float64)
    cur = float(peak_win[0])
    for i in range(n_frames):
        cur = max(float(peak_win[i]), cur * decay)
        held[i] = cur

    frame_gains = np.minimum(1.0, target_lin / np.maximum(held, 1e-12))
    x_frames = (np.arange(n_frames, dtype=np.float64) + 0.5) * step
    x_samples = np.arange(n_samples, dtype=np.float64)
    sample_gains = np.interp(x_samples, x_frames, frame_gains)
    limited = audio * sample_gains[:, np.newaxis]

    # Final true-peak safety trim if a fast transient exceeds ceiling_dbtp
    tp_lin = chunked_true_peak_linear(limited)
    if tp_lin > ceiling_lin:
        limited = limited * (ceiling_lin * 0.998 / tp_lin)
    return np.clip(limited, -ceiling_lin, ceiling_lin)


def _gain(audio: np.ndarray, sample_rate: int, gain_db: float = 0.0) -> np.ndarray:
    del sample_rate
    g_db = max(-48.0, min(48.0, float(gain_db)))
    if abs(g_db) < 1e-6:
        return audio.copy()
    return audio * (10.0 ** (g_db / 20.0))


def _measure_lufs_and_tp(audio: np.ndarray, sample_rate: int) -> tuple[float | None, float]:
    acc = StreamingLUFS(sample_rate, audio.shape[1])
    acc.update(audio)
    return acc.finalize(), max(chunked_true_peak_linear(audio), 1e-12)


def _loudness_normalize_with_info(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = -18.0,
    true_peak_ceiling_dbtp: float = -1.0,
    limit_peaks: bool = False,
    limiter_release_ms: float = 80.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    duration_s = audio.shape[0] / float(max(sample_rate, 1))
    rms = float(np.sqrt(np.mean(audio * audio)))
    if duration_s < 0.45 or rms <= 1e-10:
        return audio.copy(), {"applied_gain_db": 0.0, "peak_limited": bool(limit_peaks)}

    cur_lufs, tp_lin = _measure_lufs_and_tp(audio, sample_rate)
    if cur_lufs is None or not math.isfinite(cur_lufs):
        return audio.copy(), {"applied_gain_db": 0.0, "peak_limited": bool(limit_peaks)}

    ceiling = float(true_peak_ceiling_dbtp)
    desired_gain_db = max(-48.0, min(48.0, float(target_lufs) - cur_lufs))

    if not limit_peaks:
        tp_db = 20.0 * math.log10(tp_lin)
        applied_gain_db = min(desired_gain_db, ceiling - tp_db)
        return audio * (10.0 ** (applied_gain_db / 20.0)), {
            "target_lufs": finite_or_none(target_lufs, digits=2),
            "applied_gain_db": finite_or_none(applied_gain_db, digits=3) or 0.0,
            "peak_limited": False,
        }

    rel_ms = float(limiter_release_ms)
    out = _true_peak_limiter(
        audio * (10.0 ** (desired_gain_db / 20.0)),
        sample_rate,
        ceiling_dbtp=ceiling,
        release_ms=rel_ms,
    )
    total_gain_db = desired_gain_db
    lufs_1, tp_1 = _measure_lufs_and_tp(out, sample_rate)
    for _ in range(3):
        if lufs_1 is None or not math.isfinite(lufs_1):
            break
        deficit = float(target_lufs) - lufs_1
        if deficit <= 0.25:
            break
        topup_db = min(18.0, deficit)
        out = _true_peak_limiter(
            out * (10.0 ** (topup_db / 20.0)),
            sample_rate,
            ceiling_dbtp=ceiling,
            release_ms=rel_ms,
        )
        total_gain_db += topup_db
        lufs_1, tp_1 = _measure_lufs_and_tp(out, sample_rate)

    tp_1_db = 20.0 * math.log10(max(tp_1, 1e-12))
    if tp_1_db > ceiling:
        trim_db = ceiling - tp_1_db
        out = out * (10.0 ** (trim_db / 20.0))
        total_gain_db += trim_db

    return out, {
        "target_lufs": finite_or_none(target_lufs, digits=2),
        "applied_gain_db": finite_or_none(total_gain_db, digits=3) or 0.0,
        "peak_limited": True,
        "ceiling_dbtp": finite_or_none(ceiling, digits=2),
    }


def _loudness_normalize(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = -18.0,
    true_peak_ceiling_dbtp: float = -1.0,
    limit_peaks: bool = False,
    limiter_release_ms: float = 80.0,
) -> np.ndarray:
    out, _ = _loudness_normalize_with_info(
        audio,
        sample_rate,
        target_lufs=target_lufs,
        true_peak_ceiling_dbtp=true_peak_ceiling_dbtp,
        limit_peaks=limit_peaks,
        limiter_release_ms=limiter_release_ms,
    )
    return out


# ---------------------------------------------------------------------------
# v0.5.0 Nature, Hardware Ingest & Psychoacoustic Filters
# ---------------------------------------------------------------------------


def _repair_clipped_channel_slice(
    main_ch: np.ndarray,
    b_matched_ch: np.ndarray,
    clip_mask_ch: np.ndarray,
    backup_raw_ch: np.ndarray,
    backup_offset: float,
    sample_rate: int,
    xf_n: int,
    ar_fallback: bool,
    active_lo: int = 0,
    active_hi: int | None = None,
) -> tuple[np.ndarray, int, int, int]:
    """Repair merged clipped spans on a 1D slice via matched backup + optional AR fallback."""
    n_samples = main_ch.size
    a_hi = n_samples if active_hi is None else min(n_samples, active_hi)
    out = main_ch.copy()
    idx_c = np.flatnonzero(clip_mask_ch)
    if idx_c.size == 0:
        return out, 0, 0, 0

    spans_c: list[tuple[int, int]] = []
    s = e = int(idx_c[0])
    merge_gap = max(2, int(0.001 * sample_rate))
    for idx in idx_c[1:]:
        ii = int(idx)
        if ii - e <= merge_gap:
            e = ii
        else:
            spans_c.append((s, e))
            s = e = ii
    spans_c.append((s, e))

    repaired_regions = 0
    repaired_samples = 0
    ar_fallback_regions = 0
    for s_raw, e_raw in spans_c:
        mid = (s_raw + e_raw) // 2
        in_active = active_lo <= mid < a_hi
        s_i = max(0, s_raw - xf_n)
        e_i = min(n_samples - 1, e_raw + xf_n)
        b_lo = max(0, int(round(s_raw + backup_offset)) - 4)
        b_hi = min(backup_raw_ch.size, int(round(e_raw + backup_offset)) + 5)
        core_b_raw = backup_raw_ch[b_lo:b_hi] if b_hi > b_lo else np.zeros(0)
        backup_clipped = (
            float(np.max(np.abs(core_b_raw))) >= 0.995 if core_b_raw.size else False
        )
        if backup_clipped and ar_fallback:
            out[s_raw : e_raw + 1] = ar_interpolate_gap(out, s_raw, e_raw, order=16)
            if in_active:
                ar_fallback_regions += 1
                repaired_regions += 1
                repaired_samples += e_raw - s_raw + 1
            continue

        seg_b = b_matched_ch[s_i : e_i + 1].copy()
        seg_len = seg_b.size
        if seg_len <= 0:
            continue
        xf = min(xf_n, max(1, seg_len // 3))
        if xf > 0 and s_i > 0 and e_i < n_samples - 1:
            w_l = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, xf, endpoint=True)))
            w_r = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, xf, endpoint=True)))
            seg_b[:xf] = main_ch[s_i : s_i + xf] * (1.0 - w_l) + seg_b[:xf] * w_l
            seg_b[-xf:] = seg_b[-xf:] * w_r + main_ch[e_i + 1 - xf : e_i + 1] * (1.0 - w_r)
        out[s_i : e_i + 1] = seg_b
        if in_active:
            repaired_regions += 1
            repaired_samples += e_raw - s_raw + 1
    return out, repaired_regions, repaired_samples, ar_fallback_regions


def _declip_from_backup_with_info(
    audio: np.ndarray,
    sample_rate: int,
    backup_audio_path: str,
    clip_threshold: float = 0.985,
    crossfade_ms: float = 2.0,
    fir_taps: int = 31,
    max_lag_ms: float = 250.0,
    ar_fallback: bool = True,
    head_pad_frames: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Repair main-track digital/analog clipping via sub-sample aligned LS-FIR backup splice."""
    full_main = ensure_2d_audio(audio)
    total_samples, n_ch = full_main.shape
    hp_frames = max(0, min(total_samples, int(head_pad_frames)))
    main = full_main[hp_frames:]
    n_samples = main.shape[0]
    if n_samples < 32:
        return full_main.copy(), {"repaired_regions": 0, "repaired_samples": 0}

    b_path = Path(str(backup_audio_path))
    if not b_path.exists():
        raise FileNotFoundError(f"Backup audio file not found: {b_path}")
    b_info = probe_audio(b_path)
    if b_info.sample_rate != sample_rate or b_info.channels != n_ch:
        raise ValueError(
            f"Backup format mismatch ({b_info.sample_rate}Hz/{b_info.channels}ch) "
            f"vs main ({sample_rate}Hz/{n_ch}ch)."
        )

    backup_raw, _ = load_audio_slice(
        b_path, start_s=0.0, end_s=n_samples / float(max(sample_rate, 1)), dtype="float64"
    )
    active_len = min(n_samples, backup_raw.shape[0])
    if active_len < 32:
        return full_main.copy(), {"repaired_regions": 0, "repaired_samples": 0}
    main_active = main[:active_len]
    backup_raw = backup_raw[:active_len]

    clip_mask_2d = detect_clipping_mask(main_active, threshold=clip_threshold)
    any_clipped = np.any(clip_mask_2d, axis=1)
    repaired_regions = 0
    repaired_samples = 0
    ar_fallback_regions = 0

    lag_samples, coh = estimate_gcc_phat_lag(
        main_active[:, 0], backup_raw[:, 0], sample_rate, max_lag_ms=max_lag_ms
    )

    out_full = full_main.copy()
    out = out_full[hp_frames : hp_frames + active_len]
    gain_dbs: list[float | None] = []
    null_dbs: list[float | None] = []
    xf_n = max(1, int(round(max(0.2, float(crossfade_ms)) * 0.001 * sample_rate)))

    for ch in range(n_ch):
        b_aligned = fractional_shift_channel(backup_raw[:, ch], -lag_samples)
        valid_fit = (~any_clipped) & (np.abs(main_active[:, ch]) > 1e-5)
        if int(np.sum(valid_fit)) >= max(64, int(fir_taps) * 4):
            coeffs, g_db, null_db = fit_least_squares_fir(
                b_aligned,
                main_active[:, ch],
                taps=fir_taps,
                valid_mask=~clip_mask_2d[:, ch],
            )
            b_matched = signal.fftconvolve(b_aligned, coeffs, mode="same")
        else:
            rms_m = float(np.sqrt(np.mean(main_active[:, ch] ** 2))) + 1e-12
            rms_b = float(np.sqrt(np.mean(b_aligned**2))) + 1e-12
            g = max(0.05, min(20.0, rms_m / rms_b))
            b_matched = b_aligned * g
            g_db = finite_or_none(20.0 * math.log10(g), digits=3) or 0.0
            null_db = 0.0
        gain_dbs.append(g_db)
        null_dbs.append(null_db)

        rep_ch, reg_c, samp_c, ar_c = _repair_clipped_channel_slice(
            main_active[:, ch],
            b_matched,
            clip_mask_2d[:, ch],
            backup_raw[:, ch],
            lag_samples,
            sample_rate,
            xf_n,
            ar_fallback,
        )
        out[:, ch] = rep_ch
        repaired_regions += reg_c
        repaired_samples += samp_c
        ar_fallback_regions += ar_c

    return out_full, {
        "lag_samples": finite_or_none(lag_samples, digits=3) or 0.0,
        "alignment_coherence": finite_or_none(coh, digits=3) or 0.0,
        "fitted_gain_db": gain_dbs[0] if len(gain_dbs) == 1 else gain_dbs,
        "fit_null_residual_db": null_dbs[0] if len(null_dbs) == 1 else null_dbs,
        "repaired_regions": repaired_regions,
        "repaired_samples": repaired_samples,
        "ar_fallback_regions": ar_fallback_regions,
    }


def _declip_from_backup(
    audio: np.ndarray,
    sample_rate: int,
    backup_audio_path: str,
    clip_threshold: float = 0.985,
    crossfade_ms: float = 2.0,
    fir_taps: int = 31,
    max_lag_ms: float = 250.0,
    ar_fallback: bool = True,
) -> np.ndarray:
    out, _ = _declip_from_backup_with_info(
        audio,
        sample_rate,
        backup_audio_path=backup_audio_path,
        clip_threshold=clip_threshold,
        crossfade_ms=crossfade_ms,
        fir_taps=fir_taps,
        max_lag_ms=max_lag_ms,
        ar_fallback=ar_fallback,
    )
    return out


def _resolve_spatial_mix_globals(
    cal_audio: np.ndarray,
    sample_rate: int,
    mode: str,
    directional_channel: int,
    align_delay: bool,
    crossover_hz: float,
    auto_balance: bool,
    balance_db: float,
    decorrelate_ms: float,
) -> dict[str, Any]:
    """Resolve deterministic global mode, GCC-PHAT lag, and L/R trim before processing."""
    arr = ensure_2d_audio(cal_audio)
    ch = arr.shape[1]
    layout_info = detect_channel_layout(arr, sample_rate)
    req_mode = str(mode).strip().lower()
    eff_mode = req_mode
    if req_mode == "auto":
        hint = layout_info["layout"]
        if hint == "raw_ms":
            eff_mode = "ms_decode"
        elif hint == "directional_plus_surround":
            eff_mode = "dual_mic_blend"
        elif hint == "dual_mono" and float(decorrelate_ms) > 0.0:
            eff_mode = "dual_mono_to_stereo"
        else:
            eff_mode = "lr_adjust"

    lag_samples = 0.0
    dir_idx = 0 if int(directional_channel) <= 0 else 1
    if eff_mode == "dual_mic_blend" and ch >= 2 and float(crossover_hz) <= 0.0 and align_delay:
        lag_samples, _ = estimate_gcc_phat_lag(
            arr[:, dir_idx], arr[:, 1 - dir_idx], sample_rate, max_lag_ms=50.0
        )

    eff_balance_db = float(balance_db)
    if auto_balance and ch >= 2 and eff_mode in {"lr_adjust", "ms_decode", "dual_mic_blend"}:
        l_r = float(np.sqrt(np.mean(arr[:, 0] ** 2))) + 1e-12
        r_r = float(np.sqrt(np.mean(arr[:, 1] ** 2))) + 1e-12
        bal = 20.0 * math.log10(l_r / r_r)
        eff_balance_db = max(-6.0, min(6.0, -bal))

    return {
        "requested_mode": req_mode,
        "effective_mode": eff_mode,
        "detected_layout": layout_info["layout"],
        "layout_confidence": layout_info["confidence"],
        "lag_samples": round(float(lag_samples), 3),
        "effective_balance_db": round(float(eff_balance_db), 3),
    }


def _apply_spatial_mix_chunk(
    audio: np.ndarray,
    sample_rate: int,
    globals_cfg: dict[str, Any],
    side_gain_db: float = 0.0,
    directional_weight: float = 0.5,
    directional_channel: int = 0,
    crossover_hz: float = 0.0,
    low_mono_hz: float = 0.0,
    side_hpf_hz: float = 0.0,
    decorrelate_ms: float = 0.0,
) -> np.ndarray:
    arr = ensure_2d_audio(audio)
    n_samples, n_ch = arr.shape
    eff_mode = globals_cfg["effective_mode"]
    side_lin = 10.0 ** (max(-24.0, min(12.0, float(side_gain_db))) / 20.0)
    nyq = sample_rate * 0.5

    if n_ch == 1:
        if eff_mode == "dual_mono_to_stereo" and n_samples >= 32:
            m = arr[:, 0]
            dec_ms = max(2.0, float(decorrelate_ms) or 6.0)
            delay_n = max(4, int(round(dec_ms * 0.001 * sample_rate)))
            sos_s = signal.butter(
                2, min(250.0, nyq * 0.6), btype="highpass", fs=sample_rate, output="sos"
            )
            hf_m = _safe_sosfiltfilt(sos_s, m)
            delayed = np.zeros_like(hf_m)
            if delay_n < n_samples:
                delayed[delay_n:] = hf_m[:-delay_n]
            s = 0.25 * (hf_m - delayed)
            out = np.column_stack([m + s, m - s])
        else:
            return arr.copy()
    else:
        ch0, ch1 = arr[:, 0], arr[:, 1]
        if eff_mode == "ms_decode":
            m = ch0
            s = ch1 * side_lin
            if side_hpf_hz > 0.0 and side_hpf_hz < nyq * 0.85 and n_samples >= 24:
                sos_sh = signal.butter(
                    4, float(side_hpf_hz), btype="highpass", fs=sample_rate, output="sos"
                )
                s = _safe_sosfiltfilt(sos_sh, s)
            left = (m + s) * 0.7071067811865475
            right = (m - s) * 0.7071067811865475
            out = np.column_stack([left, right])
        elif eff_mode == "dual_mic_blend":
            dir_idx = 0 if int(directional_channel) <= 0 else 1
            d_ch = ch0 if dir_idx == 0 else ch1
            s_ch = ch1 if dir_idx == 0 else ch0
            w_dir = max(0.0, min(1.0, float(directional_weight)))
            if float(crossover_hz) > 10.0 and float(crossover_hz) < nyq * 0.8 and n_samples >= 32:
                fc = float(crossover_hz)
                sos_lp = signal.butter(4, fc, btype="lowpass", fs=sample_rate, output="sos")
                sos_hp = signal.butter(4, fc, btype="highpass", fs=sample_rate, output="sos")
                low_part = _safe_sosfiltfilt(sos_lp, s_ch)
                high_part = _safe_sosfiltfilt(sos_hp, d_ch)
                center = low_part * (1.0 - 0.5 * w_dir) + high_part * (0.5 + 0.5 * w_dir)
                amb_side = 0.35 * (low_part - _safe_sosfiltfilt(sos_lp, d_ch)) * side_lin
                out = np.column_stack([center + amb_side, center - amb_side])
            else:
                s_aligned = fractional_shift_channel(s_ch, -float(globals_cfg["lag_samples"]))
                center = w_dir * d_ch + (1.0 - w_dir) * s_aligned
                diff = 0.30 * (s_aligned - d_ch) * side_lin
                out = np.column_stack([center + diff, center - diff])
        elif eff_mode == "dual_mono_to_stereo":
            m = 0.5 * (ch0 + ch1)
            dec_ms = max(2.0, float(decorrelate_ms) or 6.0)
            delay_n = max(4, int(round(dec_ms * 0.001 * sample_rate)))
            sos_s = signal.butter(
                2, min(250.0, nyq * 0.6), btype="highpass", fs=sample_rate, output="sos"
            )
            hf_m = _safe_sosfiltfilt(sos_s, m)
            delayed = np.zeros_like(hf_m)
            if delay_n < n_samples:
                delayed[delay_n:] = hf_m[:-delay_n]
            s = 0.25 * (hf_m - delayed)
            out = np.column_stack([m + s, m - s])
        else:
            m = 0.5 * (ch0 + ch1)
            s = 0.5 * (ch0 - ch1) * side_lin
            hpf_cut = max(float(side_hpf_hz), float(low_mono_hz))
            if hpf_cut > 0.0 and hpf_cut < nyq * 0.85 and n_samples >= 24:
                sos_sh = signal.butter(4, hpf_cut, btype="highpass", fs=sample_rate, output="sos")
                s = _safe_sosfiltfilt(sos_sh, s)
            out = np.column_stack([m + s, m - s])

    if (
        eff_mode != "lr_adjust"
        and max(float(low_mono_hz), float(side_hpf_hz)) > 0.0
        and n_samples >= 24
    ):
        cut = min(nyq * 0.85, max(float(low_mono_hz), float(side_hpf_hz)))
        m = 0.5 * (out[:, 0] + out[:, 1])
        s = 0.5 * (out[:, 0] - out[:, 1])
        sos_lm = signal.butter(4, cut, btype="highpass", fs=sample_rate, output="sos")
        s = _safe_sosfiltfilt(sos_lm, s)
        out = np.column_stack([m + s, m - s])

    eff_bal = float(globals_cfg["effective_balance_db"])
    if abs(eff_bal) > 1e-4:
        g_l = 10.0 ** ((0.5 * eff_bal) / 20.0)
        g_r = 10.0 ** ((-0.5 * eff_bal) / 20.0)
        out[:, 0] *= g_l
        out[:, 1] *= g_r
    return out


def _spatial_mix_with_info(
    audio: np.ndarray,
    sample_rate: int,
    mode: str = "lr_adjust",
    side_gain_db: float = 0.0,
    directional_weight: float = 0.5,
    directional_channel: int = 0,
    crossover_hz: float = 0.0,
    align_delay: bool = True,
    low_mono_hz: float = 0.0,
    side_hpf_hz: float = 0.0,
    balance_db: float = 0.0,
    auto_balance: bool = False,
    decorrelate_ms: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = ensure_2d_audio(audio)
    globals_cfg = _resolve_spatial_mix_globals(
        arr,
        sample_rate,
        mode=mode,
        directional_channel=directional_channel,
        align_delay=align_delay,
        crossover_hz=crossover_hz,
        auto_balance=auto_balance,
        balance_db=balance_db,
        decorrelate_ms=decorrelate_ms,
    )
    out = _apply_spatial_mix_chunk(
        arr,
        sample_rate,
        globals_cfg,
        side_gain_db=side_gain_db,
        directional_weight=directional_weight,
        directional_channel=directional_channel,
        crossover_hz=crossover_hz,
        low_mono_hz=low_mono_hz,
        side_hpf_hz=side_hpf_hz,
        decorrelate_ms=decorrelate_ms,
    )
    return out, globals_cfg


def _spatial_mix(
    audio: np.ndarray,
    sample_rate: int,
    mode: str = "lr_adjust",
    side_gain_db: float = 0.0,
    directional_weight: float = 0.5,
    directional_channel: int = 0,
    crossover_hz: float = 0.0,
    align_delay: bool = True,
    low_mono_hz: float = 0.0,
    side_hpf_hz: float = 0.0,
    balance_db: float = 0.0,
    auto_balance: bool = False,
    decorrelate_ms: float = 0.0,
) -> np.ndarray:
    out, _ = _spatial_mix_with_info(
        audio,
        sample_rate,
        mode=mode,
        side_gain_db=side_gain_db,
        directional_weight=directional_weight,
        directional_channel=directional_channel,
        crossover_hz=crossover_hz,
        align_delay=align_delay,
        low_mono_hz=low_mono_hz,
        side_hpf_hz=side_hpf_hz,
        balance_db=balance_db,
        auto_balance=auto_balance,
        decorrelate_ms=decorrelate_ms,
    )
    return out


def _resolve_harmonic_track_spans(
    duration_s: float,
    sample_rate: int,
    start_s: float | None,
    end_s: float | None,
    f0_min_hz: float,
    f0_max_hz: float,
    max_harmonics: int,
    bandwidth_hz: float,
    strength: float,
    max_attenuation_db: float,
    broadband_assist: float,
    spatial_bias: bool,
    protect_birds: bool,
    texture_fill: float,
    context_s: float,
    context_gap_s: float,
    fade_ms: float,
    n_fft: int,
    spans: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    raw_list = (
        spans
        if spans
        else (
            [{"start_s": start_s, "end_s": end_s}]
            if start_s is not None and end_s is not None
            else ([{"start_s": 0.0, "end_s": duration_s}] if duration_s > 0.0 else [])
        )
    )
    resolved: list[dict[str, Any]] = []
    for item in raw_list:
        s_t = max(0.0, float(item["start_s"]))
        e_t = min(duration_s, float(item["end_s"]))
        if e_t <= s_t:
            continue
        f_lo = max(30.0, float(item.get("f0_min_hz", f0_min_hz)))
        f_hi_max = min(
            sample_rate * 0.45, max(1000.0, float(item.get("f0_max_hz", f0_max_hz)) * 12.0)
        )
        resolved.append(
            {
                "start_s": s_t,
                "end_s": e_t,
                "low_hz": f_lo,
                "high_hz": f_hi_max,
                "f0_min_hz": f_lo,
                "f0_max_hz": max(f_lo + 10.0, float(item.get("f0_max_hz", f0_max_hz))),
                "max_harmonics": max(3, min(20, int(item.get("max_harmonics", max_harmonics)))),
                "bandwidth_hz": max(6.0, min(60.0, float(item.get("bandwidth_hz", bandwidth_hz)))),
                "strength": max(0.1, min(1.5, float(item.get("strength", strength)))),
                "max_attenuation_db": max(
                    -60.0, min(-3.0, float(item.get("max_attenuation_db", max_attenuation_db)))
                ),
                "broadband_assist": max(
                    0.0, min(1.0, float(item.get("broadband_assist", broadband_assist)))
                ),
                "spatial_bias": bool(item.get("spatial_bias", spatial_bias)),
                "protect_birds": bool(item.get("protect_birds", protect_birds)),
                "texture_fill": max(0.0, min(1.0, float(item.get("texture_fill", texture_fill)))),
                "context_s": max(1.0, min(20.0, float(item.get("context_s", context_s)))),
                "context_gap_s": max(
                    0.0, min(15.0, float(item.get("context_gap_s", context_gap_s)))
                ),
                "fade_ms": max(20.0, min(5000.0, float(item.get("fade_ms", fade_ms)))),
                "n_fft": max(1024, min(8192, int(item.get("n_fft", n_fft)))),
            }
        )
    return resolved


def _harmonic_track_suppress_with_info(
    audio: np.ndarray,
    sample_rate: int,
    start_s: float | None = None,
    end_s: float | None = None,
    f0_min_hz: float = 55.0,
    f0_max_hz: float = 300.0,
    max_harmonics: int = 12,
    bandwidth_hz: float = 18.0,
    strength: float = 1.0,
    max_attenuation_db: float = -28.0,
    broadband_assist: float = 0.35,
    spatial_bias: bool = True,
    protect_birds: bool = True,
    texture_fill: float = 0.55,
    context_s: float = 6.0,
    context_gap_s: float = 1.5,
    fade_ms: float = 1000.0,
    n_fft: int = 4096,
    spans: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    dur_s = audio.shape[0] / float(max(sample_rate, 1))
    resolved = _resolve_harmonic_track_spans(
        dur_s,
        sample_rate,
        start_s,
        end_s,
        f0_min_hz,
        f0_max_hz,
        max_harmonics,
        bandwidth_hz,
        strength,
        max_attenuation_db,
        broadband_assist,
        spatial_bias,
        protect_birds,
        texture_fill,
        context_s,
        context_gap_s,
        fade_ms,
        n_fft,
        spans,
    )
    if not resolved:
        return audio.copy(), []
    return apply_masked_stft_transform(audio, sample_rate, resolved, mode="harmonic_comb")


def _harmonic_track_suppress(
    audio: np.ndarray,
    sample_rate: int,
    start_s: float | None = None,
    end_s: float | None = None,
    f0_min_hz: float = 55.0,
    f0_max_hz: float = 300.0,
    max_harmonics: int = 12,
    bandwidth_hz: float = 18.0,
    strength: float = 1.0,
    max_attenuation_db: float = -28.0,
    broadband_assist: float = 0.35,
    spatial_bias: bool = True,
    protect_birds: bool = True,
    texture_fill: float = 0.55,
    context_s: float = 6.0,
    context_gap_s: float = 1.5,
    fade_ms: float = 1000.0,
    n_fft: int = 4096,
    spans: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    out, _ = _harmonic_track_suppress_with_info(
        audio,
        sample_rate,
        start_s=start_s,
        end_s=end_s,
        f0_min_hz=f0_min_hz,
        f0_max_hz=f0_max_hz,
        max_harmonics=max_harmonics,
        bandwidth_hz=bandwidth_hz,
        strength=strength,
        max_attenuation_db=max_attenuation_db,
        broadband_assist=broadband_assist,
        spatial_bias=spatial_bias,
        protect_birds=protect_birds,
        texture_fill=texture_fill,
        context_s=context_s,
        context_gap_s=context_gap_s,
        fade_ms=fade_ms,
        n_fft=n_fft,
        spans=spans,
    )
    return out


def _resolve_distance_soften_spans(
    duration_s: float,
    sample_rate: int,
    start_s: float | None,
    end_s: float | None,
    mode: str,
    low_hz: float,
    high_hz: float,
    attenuation_db: float,
    air_absorption: float,
    attack_smear_ms: float,
    formant_blur: float,
    narrow_stereo: float,
    protect_birds: bool,
    context_s: float,
    context_gap_s: float,
    fade_ms: float,
    n_fft: int,
    spans: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    raw_list = (
        spans
        if spans
        else (
            [{"start_s": start_s, "end_s": end_s}]
            if start_s is not None and end_s is not None
            else ([{"start_s": 0.0, "end_s": duration_s}] if duration_s > 0.0 else [])
        )
    )
    nyq = sample_rate * 0.5
    resolved: list[dict[str, Any]] = []
    for item in raw_list:
        s_t = max(0.0, float(item["start_s"]))
        e_t = min(duration_s, float(item["end_s"]))
        if e_t <= s_t:
            continue
        m_mode = str(item.get("mode", mode)).strip().lower()
        def_lo = 220.0 if m_mode == "voice_muffle" and "low_hz" not in item else low_hz
        def_hi = 3500.0 if m_mode == "voice_muffle" and "high_hz" not in item else high_hz
        f_lo = max(20.0, float(item.get("low_hz", def_lo)))
        f_hi = max(f_lo + 40.0, min(nyq, float(item.get("high_hz", def_hi))))
        resolved.append(
            {
                "start_s": s_t,
                "end_s": e_t,
                "mode": m_mode,
                "low_hz": f_lo,
                "high_hz": f_hi,
                "max_attenuation_db": max(
                    -36.0, min(-2.0, float(item.get("attenuation_db", attenuation_db)))
                ),
                "air_absorption": max(
                    0.0, min(1.0, float(item.get("air_absorption", air_absorption)))
                ),
                "attack_smear_ms": max(
                    0.0, min(30.0, float(item.get("attack_smear_ms", attack_smear_ms)))
                ),
                "formant_blur": max(0.0, min(1.0, float(item.get("formant_blur", formant_blur)))),
                "narrow_stereo": max(
                    0.0, min(1.0, float(item.get("narrow_stereo", narrow_stereo)))
                ),
                "protect_birds": bool(item.get("protect_birds", protect_birds)),
                "context_s": max(1.0, min(20.0, float(item.get("context_s", context_s)))),
                "context_gap_s": max(
                    0.0, min(15.0, float(item.get("context_gap_s", context_gap_s)))
                ),
                "fade_ms": max(20.0, min(5000.0, float(item.get("fade_ms", fade_ms)))),
                "n_fft": max(512, min(8192, int(item.get("n_fft", n_fft)))),
                "texture_fill": 0.0,
            }
        )
    return resolved


def _distance_soften_with_info(
    audio: np.ndarray,
    sample_rate: int,
    start_s: float | None = None,
    end_s: float | None = None,
    mode: str = "distance",
    low_hz: float = 150.0,
    high_hz: float = 6000.0,
    attenuation_db: float = -14.0,
    air_absorption: float = 0.65,
    attack_smear_ms: float = 12.0,
    formant_blur: float = 0.5,
    narrow_stereo: float = 0.35,
    protect_birds: bool = True,
    context_s: float = 6.0,
    context_gap_s: float = 1.5,
    fade_ms: float = 500.0,
    n_fft: int = 2048,
    spans: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    dur_s = audio.shape[0] / float(max(sample_rate, 1))
    resolved = _resolve_distance_soften_spans(
        dur_s,
        sample_rate,
        start_s,
        end_s,
        mode,
        low_hz,
        high_hz,
        attenuation_db,
        air_absorption,
        attack_smear_ms,
        formant_blur,
        narrow_stereo,
        protect_birds,
        context_s,
        context_gap_s,
        fade_ms,
        n_fft,
        spans,
    )
    if not resolved:
        return audio.copy(), []
    return apply_masked_stft_transform(audio, sample_rate, resolved, mode="distance_soften")


def _distance_soften(
    audio: np.ndarray,
    sample_rate: int,
    start_s: float | None = None,
    end_s: float | None = None,
    mode: str = "distance",
    low_hz: float = 150.0,
    high_hz: float = 6000.0,
    attenuation_db: float = -14.0,
    air_absorption: float = 0.65,
    attack_smear_ms: float = 12.0,
    formant_blur: float = 0.5,
    narrow_stereo: float = 0.35,
    protect_birds: bool = True,
    context_s: float = 6.0,
    context_gap_s: float = 1.5,
    fade_ms: float = 500.0,
    n_fft: int = 2048,
    spans: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    out, _ = _distance_soften_with_info(
        audio,
        sample_rate,
        start_s=start_s,
        end_s=end_s,
        mode=mode,
        low_hz=low_hz,
        high_hz=high_hz,
        attenuation_db=attenuation_db,
        air_absorption=air_absorption,
        attack_smear_ms=attack_smear_ms,
        formant_blur=formant_blur,
        narrow_stereo=narrow_stereo,
        protect_birds=protect_birds,
        context_s=context_s,
        context_gap_s=context_gap_s,
        fade_ms=fade_ms,
        n_fft=n_fft,
        spans=spans,
    )
    return out


def _bed_shape(
    audio: np.ndarray,
    sample_rate: int,
    sub_rumble_cut_db: float = -6.0,
    harshness_dip_db: float = -3.0,
    warmth_tilt_db: float = -1.5,
    floor_preserve: float = 0.85,
    bed_width_boost: float = 0.0,
    deharsh_dynamic: bool = True,
    protect_birds: bool = True,
    n_fft: int = 2048,
) -> np.ndarray:
    """Reshape slow stationary background bed toward a warmer psychoacoustic slope."""
    arr = ensure_2d_audio(audio)
    n_samples, n_ch = arr.shape
    if n_samples < 64:
        return arr.copy()

    freqs, times, z_list, nperseg, noverlap = _stft_channels(arr, sample_rate, n_fft)
    mag_mean = np.sqrt(np.maximum(np.mean([np.abs(z) ** 2 for z in z_list], axis=0), 1e-24))
    dt = float(times[1] - times[0]) if times.size > 1 else 0.02

    k_pre = max(3, int(round(0.45 / max(dt, 1e-4))))
    k_win = max(5, int(round(4.0 / max(dt, 1e-4))))
    k_post = max(3, int(round(1.5 / max(dt, 1e-4))))
    mag_sm = ndimage.uniform_filter1d(mag_mean, size=k_pre, axis=1, mode="nearest")
    bed_min = ndimage.grey_erosion(mag_sm, size=(1, min(k_win, mag_mean.shape[1])))
    bed_lin = ndimage.uniform_filter1d(
        bed_min, size=min(k_post, mag_mean.shape[1]), axis=1, mode="nearest"
    )

    w_bed = np.clip(bed_lin / (mag_mean + 1e-12), 0.0, 1.0) ** 2
    if protect_birds:
        bird_mask = compute_bird_protect_mask(freqs, times, mag_mean)
        w_bed = w_bed * (1.0 - 0.90 * bird_mask)

    sub_cut = max(-24.0, min(0.0, float(sub_rumble_cut_db)))
    w_sub = np.clip((110.0 - freqs) / 60.0, 0.0, 1.0)
    curve_db = sub_cut * w_sub

    tilt = max(-6.0, min(6.0, float(warmth_tilt_db)))
    oct_offset = np.log2(np.maximum(freqs, 40.0) / 600.0)
    tilt_w = np.clip((freqs - 80.0) / 100.0, 0.0, 1.0) * np.exp(
        -np.maximum(0.0, freqs - 10000.0) / 6000.0
    )
    curve_db = curve_db + tilt * np.clip(oct_offset, -1.5, 2.5) * 0.45 * tilt_w

    harsh_dip = max(-12.0, min(0.0, float(harshness_dip_db)))
    bell_harsh = np.exp(-0.5 * ((freqs - 3200.0) / 1100.0) ** 2)
    if deharsh_dynamic and np.any((freqs >= 2000.0) & (freqs <= 5000.0)):
        m_hi = (freqs >= 2000.0) & (freqs <= 5000.0)
        m_mid = (freqs >= 300.0) & (freqs <= 1000.0)
        e_hi = float(np.mean(bed_lin[m_hi] ** 2)) + 1e-24
        e_mid = float(np.mean(bed_lin[m_mid] ** 2)) + 1e-24
        ratio_db = 10.0 * math.log10(e_hi / e_mid)
        dyn_scale = max(0.35, min(1.35, 1.0 + (ratio_db + 8.0) / 12.0))
        curve_db = curve_db + harsh_dip * dyn_scale * bell_harsh
    else:
        curve_db = curve_db + harsh_dip * bell_harsh

    fp = max(0.3, min(1.0, float(floor_preserve)))
    max_body_cut = -((1.0 - fp) * 24.0 + 3.0)
    curve_db = np.where(freqs >= 100.0, np.maximum(curve_db, max_body_cut), curve_db)

    applied_db = curve_db[:, np.newaxis] * w_bed
    gain_2d = 10.0 ** (applied_db / 20.0)
    z_out = [z * gain_2d for z in z_list]

    wb = max(-0.5, min(0.5, float(bed_width_boost)))
    if abs(wb) > 1e-4 and n_ch >= 2:
        f_side_w = cosine_edge_mask(freqs, 250.0, 9000.0, 100.0)[:, np.newaxis]
        mid_z = 0.5 * (z_out[0] + z_out[1])
        side_z = 0.5 * (z_out[0] - z_out[1]) * (1.0 + wb * f_side_w * w_bed)
        z_out[0] = mid_z + side_z
        z_out[1] = mid_z - side_z

    return _istft_channels(z_out, sample_rate, nperseg, noverlap, n_samples)


_FILTER_HANDLERS = {
    "highpass_rumble": _highpass_rumble,
    "dehum_notch": _dehum_notch,
    "drone_notch": _drone_notch,
    "declick_impulse": _declick_impulse,
    "spectral_denoise": _spectral_denoise,
    "spectral_match_suppress": _spectral_match_suppress,
    "band_attenuate": _band_attenuate,
    "ambience_patch": _ambience_patch,
    "trim_range": _trim_range,
    "cut_regions": _cut_regions,
    "fade_edges": _fade_edges,
    "gain": _gain,
    "parametric_eq": _parametric_eq,
    "soft_compress": _soft_compress,
    "true_peak_limiter": _true_peak_limiter,
    "loudness_normalize": _loudness_normalize,
    "declip_from_backup": _declip_from_backup,
    "spatial_mix": _spatial_mix,
    "harmonic_track_suppress": _harmonic_track_suppress,
    "distance_soften": _distance_soften,
    "bed_shape": _bed_shape,
}

_LOCAL_STREAMABLE_FILTERS = {
    "highpass_rumble",
    "dehum_notch",
    "drone_notch",
    "declick_impulse",
    "spectral_denoise",
    "band_attenuate",
    "gain",
    "parametric_eq",
    "soft_compress",
    "true_peak_limiter",
    "bed_shape",
}


def _array_rms_peak_db(audio: np.ndarray) -> tuple[float | None, float | None]:
    if audio.size == 0:
        return None, None
    rms_lin = float(np.sqrt(np.mean(audio * audio)))
    pk_lin = float(np.max(np.abs(audio)))
    rms_db = (
        round(20.0 * math.log10(rms_lin), 4)
        if rms_lin > 0.0 and math.isfinite(rms_lin)
        else None
    )
    pk_db = (
        round(20.0 * math.log10(pk_lin), 4)
        if pk_lin > 0.0 and math.isfinite(pk_lin)
        else None
    )
    return (
        rms_db if rms_db is not None and math.isfinite(rms_db) else None,
        pk_db if pk_db is not None and math.isfinite(pk_db) else None,
    )


def apply_recipe(
    audio: np.ndarray,
    sample_rate: int,
    steps: list[FilterStep] | list[dict[str, Any]],
) -> np.ndarray:
    """Apply an ordered list of FilterStep items non-destructively in memory."""
    out = ensure_2d_audio(audio).copy()
    for item in steps:
        step = item if isinstance(item, FilterStep) else FilterStep.model_validate(item)
        handler = _FILTER_HANDLERS.get(step.filter_name)
        if handler is None:
            known = ", ".join(sorted(_FILTER_HANDLERS))
            raise ValueError(f"Unknown filter '{step.filter_name}'. Available: {known}")
        out = ensure_2d_audio(handler(out, int(sample_rate), **step.params))
    return out


def apply_recipe_with_telemetry(
    audio: np.ndarray,
    sample_rate: int,
    steps: list[FilterStep] | list[dict[str, Any]],
    exclude_events: list[tuple[float, float]] | None = None,
    head_pad_frames: int = 0,
) -> tuple[np.ndarray, list[StepTelemetry]]:
    """Apply FilterStep chain in memory and return per-step level + provenance telemetry."""
    sr = int(sample_rate)
    out = ensure_2d_audio(audio).copy()
    telemetry: list[StepTelemetry] = []

    for idx, item in enumerate(steps):
        step = item if isinstance(item, FilterStep) else FilterStep.model_validate(item)
        handler = _FILTER_HANDLERS.get(step.filter_name)
        if handler is None:
            known = ", ".join(sorted(_FILTER_HANDLERS))
            raise ValueError(f"Unknown filter '{step.filter_name}'. Available: {known}")

        in_rms_db, in_pk_db = _array_rms_peak_db(out)
        in_lufs = None
        if out.shape[0] / float(max(sr, 1)) >= 0.45:
            in_lufs, _ = _measure_lufs_and_tp(out, sr)

        details: dict[str, Any] = {}
        if step.filter_name == "ambience_patch":
            out, patch_log = _ambience_patch_with_info(
                out, sr, **step.params, exclude_events=exclude_events
            )
            details = {"patched_spans": patch_log}
        elif step.filter_name == "band_attenuate":
            out, att_log = _band_attenuate_with_info(out, sr, **step.params)
            details = {"attenuated_spans": att_log}
        elif step.filter_name == "spectral_match_suppress":
            out, sup_log = _spectral_match_suppress_with_info(out, sr, **step.params)
            details = {"suppressed_spans": sup_log}
        elif step.filter_name == "harmonic_track_suppress":
            out, track_log = _harmonic_track_suppress_with_info(out, sr, **step.params)
            details = {"tracked_spans": track_log}
        elif step.filter_name == "distance_soften":
            out, dist_log = _distance_soften_with_info(out, sr, **step.params)
            details = {"softened_spans": dist_log}
        elif step.filter_name == "declip_from_backup":
            out, dc_info = _declip_from_backup_with_info(
                out, sr, **step.params, head_pad_frames=head_pad_frames
            )
            details = dc_info
        elif step.filter_name == "spatial_mix":
            out, sp_info = _spatial_mix_with_info(out, sr, **step.params)
            details = sp_info
        elif step.filter_name == "loudness_normalize":
            out, norm_info = _loudness_normalize_with_info(out, sr, **step.params)
            details = norm_info
        else:
            out = ensure_2d_audio(handler(out, sr, **step.params))

        out_rms_db, out_pk_db = _array_rms_peak_db(out)
        out_lufs = None
        if out.shape[0] / float(max(sr, 1)) >= 0.45:
            out_lufs, _ = _measure_lufs_and_tp(out, sr)

        telemetry.append(
            StepTelemetry(
                step_index=idx,
                filter_name=step.filter_name,
                params=copy.deepcopy(step.params),
                input_rms_dbfs=in_rms_db,
                output_rms_dbfs=out_rms_db,
                input_peak_dbfs=in_pk_db,
                output_peak_dbfs=out_pk_db,
                input_lufs=in_lufs,
                output_lufs=out_lufs,
                details=details,
            )
        )
    return out, telemetry


def _shift_step_times(step: FilterStep, offset_s: float) -> FilterStep:
    """Shift time-based parameters on streamable steps relative to a chunk buffer start."""
    params = copy.deepcopy(step.params)
    if step.filter_name in {
        "band_attenuate",
        "ambience_patch",
        "spectral_match_suppress",
        "harmonic_track_suppress",
        "distance_soften",
        "declick_impulse",
    }:
        for key in ("start_s", "end_s", "source_start_s", "ref_start_s", "ref_end_s"):
            if params.get(key) is not None:
                params[key] = float(params[key]) - offset_s
        if params.get("spans"):
            for sp in params["spans"]:
                for key in ("start_s", "end_s", "source_start_s", "ref_start_s", "ref_end_s"):
                    if sp.get(key) is not None:
                        sp[key] = float(sp[key]) - offset_s
    elif step.filter_name == "spectral_denoise":
        if params.get("noise_start_s") is not None:
            params["noise_start_s"] = float(params["noise_start_s"]) - offset_s
        if params.get("noise_end_s") is not None:
            params["noise_end_s"] = float(params["noise_end_s"]) - offset_s
    return FilterStep(filter_name=step.filter_name, params=params)


def _adjust_span_telemetry(
    span_logs: list[dict[str, Any]], head_pad_s: float
) -> list[dict[str, Any]]:
    """Shift span telemetry coordinates back from padded buffer seconds to file-local seconds."""
    if head_pad_s <= 0.0:
        return span_logs
    time_keys = (
        "start_s",
        "end_s",
        "target_start_s",
        "target_end_s",
        "donor_start_s",
        "donor_end_s",
    )
    adjusted: list[dict[str, Any]] = []
    for item in span_logs:
        entry = dict(item)
        for key in time_keys:
            if entry.get(key) is not None:
                entry[key] = round(max(0.0, float(entry[key]) - head_pad_s), 3)
        adjusted.append(entry)
    return adjusted


def _prepare_seam_context(
    sr: int,
    ch: int,
    parsed: list[FilterStep],
    prev_audio_path: str | Path | None,
    next_audio_path: str | Path | None,
    seam_context_s: float,
) -> tuple[Path | None, int, float, Path | None, int, float]:
    """Validate seam-padding neighbor files and compute exact sample counts."""
    if prev_audio_path is None and next_audio_path is None:
        return None, 0, 0.0, None, 0, 0.0
    for st in parsed:
        if st.filter_name in {"trim_range", "cut_regions"}:
            raise ValueError(
                "Boundary seam padding (prev_audio_path / next_audio_path) cannot be combined "
                f"with '{st.filter_name}'."
            )
    ctx_s = max(0.5, min(30.0, float(seam_context_s)))

    prev_p: Path | None = None
    head_frames = 0
    prev_dur_s = 0.0
    if prev_audio_path is not None:
        prev_p = Path(prev_audio_path)
        p_info = probe_audio(prev_p)
        if p_info.sample_rate != sr or p_info.channels != ch:
            raise ValueError(
                f"prev_audio_path format mismatch ({p_info.sample_rate}Hz/{p_info.channels}ch) "
                f"vs source ({sr}Hz/{ch}ch)."
            )
        prev_dur_s = p_info.duration_s
        head_frames = min(p_info.frames, max(0, int(round(min(ctx_s, p_info.duration_s) * sr))))

    next_p: Path | None = None
    tail_frames = 0
    next_dur_s = 0.0
    if next_audio_path is not None:
        next_p = Path(next_audio_path)
        n_info = probe_audio(next_p)
        if n_info.sample_rate != sr or n_info.channels != ch:
            raise ValueError(
                f"next_audio_path format mismatch ({n_info.sample_rate}Hz/{n_info.channels}ch) "
                f"vs source ({sr}Hz/{ch}ch)."
            )
        next_dur_s = n_info.duration_s
        tail_frames = min(n_info.frames, max(0, int(round(min(ctx_s, n_info.duration_s) * sr))))

    return prev_p, head_frames, prev_dur_s, next_p, tail_frames, next_dur_s


def _read_exact_edge_frames(
    path: Path, n_frames: int, from_end: bool, fallback_dur_s: float, sr: int
) -> np.ndarray:
    """Read exact frame count from the start or end of a neighbor audio file."""
    if n_frames <= 0:
        return np.zeros((0, 1), dtype=np.float64)
    try:
        with sf.SoundFile(str(path), mode="r") as f:
            avail = int(f.frames)
            take = min(avail, n_frames)
            f.seek(max(0, avail - take) if from_end else 0)
            data = f.read(take, always_2d=True, dtype="float64")
            return ensure_2d_audio(data, dtype="float64")
    except sf.LibsndfileError:
        s0 = max(0.0, fallback_dur_s - n_frames / float(sr)) if from_end else 0.0
        s1 = fallback_dur_s if from_end else min(fallback_dur_s, n_frames / float(sr))
        arr, _ = load_audio_slice(path, start_s=s0, end_s=s1, dtype="float64")
        return arr[-n_frames:] if from_end else arr[:n_frames]


def apply_recipe_with_seam_padding(
    audio: np.ndarray,
    sample_rate: int,
    steps: list[FilterStep] | list[dict[str, Any]],
    prev_audio_path: str | Path | None = None,
    next_audio_path: str | Path | None = None,
    seam_context_s: float = 8.0,
    exclude_events: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, list[StepTelemetry]]:
    """Apply an in-memory recipe with neighbor seam padding and return telemetry."""
    sr = int(sample_rate)
    src_arr = ensure_2d_audio(audio)
    parsed = [s if isinstance(s, FilterStep) else FilterStep.model_validate(s) for s in steps]
    prev_p, head_req, prev_dur, next_p, tail_req, next_dur = _prepare_seam_context(
        sr, src_arr.shape[1], parsed, prev_audio_path, next_audio_path, seam_context_s
    )
    if (prev_p is None and next_p is None) or not parsed or (head_req == 0 and tail_req == 0):
        return apply_recipe_with_telemetry(src_arr, sr, parsed, exclude_events=exclude_events)

    head_pad = (
        _read_exact_edge_frames(prev_p, head_req, from_end=True, fallback_dur_s=prev_dur, sr=sr)
        if prev_p is not None and head_req > 0
        else np.zeros((0, src_arr.shape[1]), dtype=np.float64)
    )
    tail_pad = (
        _read_exact_edge_frames(next_p, tail_req, from_end=False, fallback_dur_s=next_dur, sr=sr)
        if next_p is not None and tail_req > 0
        else np.zeros((0, src_arr.shape[1]), dtype=np.float64)
    )
    head_frames = head_pad.shape[0]
    head_pad_s = head_frames / float(sr)
    orig_frames = src_arr.shape[0]
    parts = [p for p in (head_pad, src_arr, tail_pad) if p.shape[0] > 0]
    padded = np.vstack(parts) if parts else src_arr

    working_steps = (
        [_shift_step_times(st, -head_pad_s) for st in parsed] if head_pad_s > 0.0 else parsed
    )
    shifted_ev = (
        [(es + head_pad_s, ee + head_pad_s) for es, ee in exclude_events]
        if exclude_events and head_pad_s > 0.0
        else exclude_events
    )
    processed, telemetry = apply_recipe_with_telemetry(
        padded,
        sr,
        working_steps,
        exclude_events=shifted_ev,
        head_pad_frames=head_frames,
    )
    sliced = processed[head_frames : head_frames + orig_frames].copy()

    for idx, st_orig in enumerate(parsed):
        if idx < len(telemetry):
            telemetry[idx].params = copy.deepcopy(st_orig.params)
            for det_key in (
                "patched_spans",
                "attenuated_spans",
                "suppressed_spans",
                "tracked_spans",
                "softened_spans",
            ):
                if det_key in telemetry[idx].details:
                    telemetry[idx].details[det_key] = _adjust_span_telemetry(
                        telemetry[idx].details[det_key], head_pad_s
                    )
    return sliced, telemetry


def _stats_from_accum(
    sum_sq: float, peak_lin: float, count: int
) -> tuple[float | None, float | None]:
    if count <= 0:
        return None, None
    if not math.isfinite(sum_sq) or sum_sq <= 0.0:
        rms_db = None
    else:
        rms_lin = math.sqrt(sum_sq / float(count))
        rms_db = round(20.0 * math.log10(rms_lin), 4) if rms_lin > 0.0 else None
    pk_db = (
        round(20.0 * math.log10(peak_lin), 4)
        if peak_lin > 0.0 and math.isfinite(peak_lin)
        else None
    )
    return (
        rms_db if rms_db is not None and math.isfinite(rms_db) else None,
        pk_db if pk_db is not None and math.isfinite(pk_db) else None,
    )


def _probe_stream_levels(wav_path: Path) -> tuple[float | None, float | None, float | None]:
    info = probe_audio(wav_path)
    sum_sq = 0.0
    peak_lin = 0.0
    count = 0
    lufs_acc = StreamingLUFS(info.sample_rate, info.channels)
    for _, block in iter_audio_blocks(wav_path, block_seconds=30.0, dtype="float64"):
        if block.size == 0:
            continue
        sum_sq += float(np.sum(block * block))
        peak_lin = max(peak_lin, float(np.max(np.abs(block))))
        count += block.size
        lufs_acc.update(block)
    rms_db, pk_db = _stats_from_accum(sum_sq, peak_lin, count)
    lufs_val = lufs_acc.finalize() if count > 0 and sum_sq > 0.0 else None
    return rms_db, pk_db, lufs_val


def _stream_local_batch(
    src_wav: Path,
    dst_wav: Path,
    steps: list[FilterStep],
    block_seconds: float,
    step_telemetry_out: list[StepTelemetry] | None = None,
    base_step_index: int = 0,
) -> None:
    info = probe_audio(src_wav)
    sr = info.sample_rate
    ch = info.channels
    total = info.frames

    max_fade_s = 0.0
    has_bed_shape = False
    for st in steps:
        if st.filter_name == "band_attenuate":
            max_fade_s = max(max_fade_s, float(st.params.get("fade_ms", 30.0)) * 0.001)
            for sp in st.params.get("spans") or []:
                max_fade_s = max(max_fade_s, float(sp.get("fade_ms", 30.0)) * 0.001)
        elif st.filter_name == "bed_shape":
            has_bed_shape = True
    pad_s = min(8.0, max(2.0, max_fade_s + 1.0))
    if has_bed_shape:
        pad_s = min(10.0, max(pad_s, 4.5))

    n_steps = len(steps)
    stage_sum_sq = [0.0] * (n_steps + 1)
    stage_peak = [0.0] * (n_steps + 1)
    stage_count = [0] * (n_steps + 1)
    lufs_accs = (
        [StreamingLUFS(sr, ch) for _ in range(n_steps + 1)]
        if step_telemetry_out is not None
        else []
    )
    global_att_spans: list[list[dict[str, Any]]] = []
    att_accum: list[list[dict[str, float]]] = []
    if step_telemetry_out is not None:
        for st in steps:
            if st.filter_name == "band_attenuate":
                sp_list = _resolve_band_spans(
                    info.duration_s,
                    st.params.get("start_s"),
                    st.params.get("end_s"),
                    float(st.params.get("low_hz", 40.0)),
                    float(st.params.get("high_hz", 250.0)),
                    float(st.params.get("attenuation_db", -18.0)),
                    float(st.params.get("fade_ms", 30.0)),
                    str(st.params.get("envelope_mode", "flat")),
                    st.params.get("spans"),
                )
                global_att_spans.append(sp_list)
                att_accum.append(
                    [{"max_cut": 0.0, "sum_mean_w": 0.0, "sum_w": 0.0} for _ in sp_list]
                )
            else:
                global_att_spans.append([])
                att_accum.append([])

    block_frames = max(1024, int(round(block_seconds * sr)))
    with sf.SoundFile(
        str(dst_wav),
        mode="w",
        samplerate=sr,
        channels=ch,
        subtype="FLOAT",
        format=_STREAM_TEMP_FORMAT,
    ) as fout:
        pos = 0
        while pos < total:
            core_end = min(total, pos + block_frames)
            core_s0 = pos / float(sr)
            core_s1 = core_end / float(sr)
            buf_s0 = max(0.0, core_s0 - pad_s)
            buf_s1 = min(info.duration_s, core_s1 + pad_s)

            buf_arr, _ = load_audio_slice(src_wav, start_s=buf_s0, end_s=buf_s1, dtype="float64")
            shifted = [_shift_step_times(st, buf_s0) for st in steps]
            trim_l = int(round((core_s0 - buf_s0) * sr))
            core_len = core_end - pos

            cur = ensure_2d_audio(buf_arr)
            if step_telemetry_out is not None:
                c0 = cur[trim_l : trim_l + core_len]
                if c0.size:
                    stage_sum_sq[0] += float(np.sum(c0 * c0))
                    stage_peak[0] = max(stage_peak[0], float(np.max(np.abs(c0))))
                    stage_count[0] += c0.size
                    lufs_accs[0].update(c0)
                for j, sh in enumerate(shifted):
                    if sh.filter_name == "band_attenuate":
                        cur, att_log = _band_attenuate_with_info(cur, sr, **sh.params)
                        cur = ensure_2d_audio(cur)
                        for entry in att_log:
                            abs_s0 = float(entry["start_s"]) + buf_s0
                            abs_s1 = float(entry["end_s"]) + buf_s0
                            for k, g in enumerate(global_att_spans[j]):
                                g_hi = round(min(sr * 0.5, g["high_hz"]), 1)
                                if (
                                    abs(round(g["low_hz"], 1) - float(entry["low_hz"])) > 0.2
                                    or abs(g_hi - float(entry["high_hz"])) > 0.2
                                ):
                                    continue
                                ov_buf = min(abs_s1, g["end_s"]) - max(abs_s0, g["start_s"])
                                ov_core = min(core_s1, g["end_s"]) - max(core_s0, g["start_s"])
                                if ov_buf > 0.0 and ov_core > 0.0:
                                    acc = att_accum[j][k]
                                    acc["max_cut"] = max(acc["max_cut"], float(entry["max_cut_db"]))
                                    acc["sum_mean_w"] += float(entry["mean_cut_db"]) * ov_core
                                    acc["sum_w"] += ov_core
                    else:
                        fn = _FILTER_HANDLERS[sh.filter_name]
                        cur = ensure_2d_audio(fn(cur, sr, **sh.params))
                    cj = cur[trim_l : trim_l + core_len]
                    if cj.size:
                        stage_sum_sq[j + 1] += float(np.sum(cj * cj))
                        stage_peak[j + 1] = max(stage_peak[j + 1], float(np.max(np.abs(cj))))
                        stage_count[j + 1] += cj.size
                        lufs_accs[j + 1].update(cj)
                processed = cur
            else:
                processed = apply_recipe(cur, sr, shifted)

            fout.write(processed[trim_l : trim_l + core_len])
            pos = core_end

    if step_telemetry_out is not None:
        stage_lufs = [acc.finalize() for acc in lufs_accs]
        for j, st in enumerate(steps):
            in_rms, in_pk = _stats_from_accum(stage_sum_sq[j], stage_peak[j], stage_count[j])
            out_rms, out_pk = _stats_from_accum(
                stage_sum_sq[j + 1], stage_peak[j + 1], stage_count[j + 1]
            )
            details: dict[str, Any] = {}
            if st.filter_name == "band_attenuate" and global_att_spans[j]:
                span_logs: list[dict[str, Any]] = []
                for g, acc in zip(global_att_spans[j], att_accum[j], strict=True):
                    f_hi = min(sr * 0.5, g["high_hz"])
                    if f_hi <= g["low_hz"]:
                        continue
                    mean_c = acc["sum_mean_w"] / acc["sum_w"] if acc["sum_w"] > 0.0 else 0.0
                    span_logs.append(
                        {
                            "start_s": round(g["start_s"], 3),
                            "end_s": round(g["end_s"], 3),
                            "low_hz": round(g["low_hz"], 1),
                            "high_hz": round(f_hi, 1),
                            "envelope_mode": g["envelope_mode"],
                            "max_cut_db": round(acc["max_cut"], 2),
                            "mean_cut_db": round(mean_c, 2),
                        }
                    )
                details = {"attenuated_spans": span_logs}
            step_telemetry_out.append(
                StepTelemetry(
                    step_index=base_step_index + j,
                    filter_name=st.filter_name,
                    params=copy.deepcopy(st.params),
                    input_rms_dbfs=in_rms,
                    output_rms_dbfs=out_rms,
                    input_peak_dbfs=in_pk,
                    output_peak_dbfs=out_pk,
                    input_lufs=stage_lufs[j],
                    output_lufs=stage_lufs[j + 1],
                    details=details,
                )
            )


def _stream_fade_edges(src_wav: Path, dst_wav: Path, params: dict[str, Any]) -> None:
    info = probe_audio(src_wav)
    sr = info.sample_rate
    ch = info.channels
    total = info.frames
    fade_in_n = min(total, max(0, int(round(float(params.get("fade_in_s", 2.0)) * sr))))
    fade_out_n = min(total, max(0, int(round(float(params.get("fade_out_s", 3.0)) * sr))))
    curve = str(params.get("curve", "cosine")).strip().lower()

    with sf.SoundFile(
        str(dst_wav),
        mode="w",
        samplerate=sr,
        channels=ch,
        subtype="FLOAT",
        format=_STREAM_TEMP_FORMAT,
    ) as fout:
        for start_frame, block in iter_audio_blocks(src_wav, block_seconds=30.0, dtype="float64"):
            n_b = block.shape[0]
            end_frame = start_frame + n_b
            if fade_in_n > 0 and start_frame < fade_in_n:
                lo = start_frame
                hi = min(end_frame, fade_in_n)
                t = np.arange(lo, hi, dtype=np.float64) / max(fade_in_n - 1, 1)
                ramp = t if curve == "linear" else 0.5 * (1.0 - np.cos(np.pi * t))
                block[: hi - lo] *= ramp[:, np.newaxis]
            out_start = total - fade_out_n
            if fade_out_n > 0 and end_frame > out_start:
                lo = max(start_frame, out_start)
                hi = end_frame
                t = 1.0 - (np.arange(lo, hi, dtype=np.float64) - out_start) / max(
                    fade_out_n - 1, 1
                )
                ramp = t if curve == "linear" else 0.5 * (1.0 - np.cos(np.pi * t))
                block[lo - start_frame : hi - start_frame] *= ramp[:, np.newaxis]
            fout.write(block)


def _stream_trim_range(src_wav: Path, dst_wav: Path, params: dict[str, Any]) -> None:
    info = probe_audio(src_wav)
    s0 = max(0.0, float(params.get("start_s", 0.0)))
    end_raw = params.get("end_s")
    s1 = info.duration_s if end_raw is None else min(info.duration_s, float(end_raw))
    with sf.SoundFile(
        str(dst_wav),
        mode="w",
        samplerate=info.sample_rate,
        channels=info.channels,
        subtype="FLOAT",
        format=_STREAM_TEMP_FORMAT,
    ) as fout:
        if s1 > s0:
            for _, block in iter_audio_blocks(
                src_wav, block_seconds=30.0, start_s=s0, end_s=s1, dtype="float64"
            ):
                fout.write(block)


def _stream_cut_regions(src_wav: Path, dst_wav: Path, params: dict[str, Any]) -> None:
    info = probe_audio(src_wav)
    sr = info.sample_rate
    ch = info.channels
    regions = params.get("regions") or []
    kept = _compute_kept_intervals(info.frames, sr, regions)
    xf_desired = max(
        1, int(max(10.0, min(5000.0, float(params.get("crossfade_ms", 1000.0)))) * 0.001 * sr)
    )

    with (
        sf.SoundFile(str(src_wav), mode="r") as fin,
        sf.SoundFile(
            str(dst_wav),
            mode="w",
            samplerate=sr,
            channels=ch,
            subtype="FLOAT",
            format=_STREAM_TEMP_FORMAT,
        ) as fout,
    ):
        tail = np.zeros((0, ch), dtype=np.float64)
        block_frames = max(4096, sr * 30)
        for ks, ke in kept:
            seg_len = ke - ks
            if seg_len <= 0:
                continue
            fin.seek(ks)
            first_of_seg = True
            rem = seg_len
            while rem > 0:
                to_read = min(block_frames, rem)
                chunk = ensure_2d_audio(fin.read(to_read, always_2d=True, dtype="float64"))
                rem -= chunk.shape[0]
                if first_of_seg:
                    first_of_seg = False
                    if tail.shape[0] > 0:
                        xf = min(tail.shape[0], chunk.shape[0])
                        t = np.linspace(0.0, math.pi * 0.5, xf, endpoint=True)[:, np.newaxis]
                        joined = tail[-xf:] * np.cos(t) + chunk[:xf] * np.sin(t)
                        if tail.shape[0] > xf:
                            fout.write(tail[:-xf])
                        fout.write(joined)
                        chunk = chunk[xf:]
                        tail = np.zeros((0, ch), dtype=np.float64)
                if rem == 0:
                    hold = min(xf_desired, chunk.shape[0] // 2)
                    if hold > 0:
                        fout.write(chunk[:-hold])
                        tail = chunk[-hold:].copy()
                    else:
                        fout.write(chunk)
                elif chunk.size:
                    fout.write(chunk)
        if tail.size:
            fout.write(tail)


def _read_exact_slice(
    src_wav: Path, start_frame: int, n_frames: int, sr: int
) -> np.ndarray:
    """Read exact frame range [start_frame, start_frame + n_frames) from src_wav."""
    if n_frames <= 0:
        return np.zeros((0, 1), dtype=np.float64)
    try:
        with sf.SoundFile(str(src_wav), mode="r") as f:
            s0 = max(0, min(int(f.frames), int(start_frame)))
            take = min(int(n_frames), max(0, int(f.frames) - s0))
            if take <= 0:
                return np.zeros((0, int(f.channels)), dtype=np.float64)
            f.seek(s0)
            return ensure_2d_audio(f.read(take, always_2d=True, dtype="float64"))
    except sf.LibsndfileError:
        s0_s = max(0.0, start_frame / float(max(sr, 1)))
        s1_s = max(s0_s, (start_frame + n_frames) / float(max(sr, 1)))
        arr, _ = load_audio_slice(src_wav, start_s=s0_s, end_s=s1_s, dtype="float64")
        return arr[:n_frames]


def _stream_window_stats_array(
    src_wav: Path, win_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-pass streaming RMS, peak, and sub-120 Hz energy ratio scan in float32 RAM."""
    info = probe_audio(src_wav)
    sr = info.sample_rate
    total = info.frames
    if total < win_frames:
        return np.zeros(0), np.zeros(0), np.zeros(0)

    blk_mult = max(1, (sr * 30) // win_frames)
    blk_frames = blk_mult * win_frames
    use_low = sr >= 400 and total >= 64
    if use_low:
        sos_low = signal.butter(
            2, min(120.0, sr * 0.35), btype="lowpass", fs=sr, output="sos"
        )
        zi = np.zeros((sos_low.shape[0], 2), dtype=np.float64)
    else:
        sos_low = None
        zi = None

    rms_parts: list[np.ndarray] = []
    pk_parts: list[np.ndarray] = []
    low_parts: list[np.ndarray] = []

    for _, blk in iter_audio_blocks(
        src_wav, block_seconds=blk_frames / float(max(sr, 1)), dtype="float32"
    ):
        n_w = blk.shape[0] // win_frames
        if n_w <= 0:
            continue
        usable = n_w * win_frames
        mono = np.mean(blk[:usable].astype(np.float64), axis=1)
        reshaped = mono.reshape(n_w, win_frames)
        w_rms = np.sqrt(np.mean(reshaped * reshaped, axis=1))
        w_pk = np.max(np.abs(reshaped), axis=1)
        if use_low and sos_low is not None and zi is not None:
            low_m, zi = signal.sosfilt(sos_low, mono, zi=zi)
            low_res = low_m.reshape(n_w, win_frames)
            w_low_rms = np.sqrt(np.mean(low_res * low_res, axis=1))
            w_low_ratio = w_low_rms / np.maximum(w_rms, 1e-9)
        else:
            w_low_ratio = np.zeros(n_w, dtype=np.float64)
        rms_parts.append(w_rms)
        pk_parts.append(w_pk)
        low_parts.append(w_low_ratio)

    if not rms_parts:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    return np.concatenate(rms_parts), np.concatenate(pk_parts), np.concatenate(low_parts)


def _pick_streaming_donor_with_span(
    src_wav: Path,
    exclude_spans: list[tuple[int, int]],
    target_len: int,
    xfade: int,
    exclude_events_s: list[tuple[float, float]] | None = None,
    used_spans: list[tuple[int, int]] | None = None,
    target_mid_frame: float | None = None,
    target_rms: float | None = None,
    stats_cache: tuple[int, np.ndarray, np.ndarray, np.ndarray] | None = None,
    min_donor_distance_s: float = 5.0,
    max_donor_crest: float = 6.0,
) -> tuple[np.ndarray, float, float, int, int]:
    """Locate an unrepeated, event-screened clean donor slice on disk with bounded RAM."""
    info = probe_audio(src_wav)
    sr = info.sample_rate
    ch = info.channels
    total = info.frames
    patch_margin = int(0.15 * sr)
    dist_margin = max(patch_margin, int(max(0.0, float(min_donor_distance_s)) * sr))
    event_margin = int(5.0 * sr)
    win_frames = stats_cache[0] if stats_cache else max(64, int(round(0.25 * sr)))
    min_region = max(win_frames * 2, int(0.25 * sr))

    ex_patch = [(s - patch_margin, e + patch_margin) for s, e in exclude_spans]
    ex_dist = [(s - dist_margin, e + dist_margin) for s, e in exclude_spans]
    ex_used = [(s - patch_margin, e + patch_margin) for s, e in (used_spans or [])]
    ex_events = [
        (int(round(es * sr)) - event_margin, int(round(ee * sr)) + event_margin)
        for es, ee in (exclude_events_s or [])
    ]

    free_spans = _complement_sample_intervals(ex_dist + ex_used + ex_events, total, min_region)
    if not free_spans:
        free_spans = _complement_sample_intervals(ex_patch + ex_used + ex_events, total, min_region)
    if not free_spans:
        free_spans = _complement_sample_intervals(ex_patch + ex_used, total, min_region)
    if not free_spans:
        free_spans = _complement_sample_intervals(ex_patch, total, min_region)
    if not free_spans:
        ex0, ex1 = exclude_spans[0] if exclude_spans else (0, total)
        fb0, fb1 = (0, ex0) if ex0 >= (total - ex1) else (ex1, total)
        raw_fb = _read_exact_slice(src_wav, fb0, fb1 - fb0, sr)
        if raw_fb.shape[0] == 0:
            return np.zeros((target_len, ch), dtype=np.float64), 0.0, 0.0, 0, 1
        tiled, loops = _build_tiled_donor_with_info(raw_fb, target_len, xfade)
        return (
            tiled,
            round(fb0 / float(sr), 3),
            round(fb1 / float(sr), 3),
            loops,
            1,
        )

    if stats_cache is None:
        w_rms, w_pk, w_low = _stream_window_stats_array(src_wav, win_frames)
    else:
        _, w_rms, w_pk, w_low = stats_cache

    ref_rms = (
        float(target_rms)
        if target_rms and target_rms > 1e-9
        else (float(np.median(w_rms)) if w_rms.size else 0.01)
    )
    mid_f = float(target_mid_frame) if target_mid_frame is not None else 0.5 * total

    picked = _score_donor_windows_from_stats(
        free_spans,
        win_frames,
        w_rms,
        w_pk,
        w_low,
        target_len,
        mid_f,
        ref_rms,
        sr,
        max_donor_crest=max_donor_crest,
    )
    if picked is None:
        fs0, fs1 = free_spans[0]
        b0, b1 = fs0, min(fs1, fs0 + target_len)
    else:
        b0, b1, _ = picked

    if used_spans is not None:
        used_spans.append((b0, b1))

    if b1 - b0 >= target_len:
        exact = _read_exact_slice(src_wav, b0, target_len, sr)
        return (
            exact,
            round(b0 / float(sr), 3),
            round((b0 + exact.shape[0]) / float(sr), 3),
            0,
            1,
        )

    chunks = [_read_exact_slice(src_wav, b0, b1 - b0, sr)]
    got = chunks[0].shape[0]
    local_used = list(used_spans or [])
    if (b0, b1) not in local_used:
        local_used.append((b0, b1))
    while got < target_len:
        cur_free = _complement_sample_intervals(
            ex_dist + [(s - patch_margin, e + patch_margin) for s, e in local_used] + ex_events,
            total,
            min_region,
        )
        if not cur_free:
            cur_free = _complement_sample_intervals(
                ex_patch
                + [(s - patch_margin, e + patch_margin) for s, e in local_used]
                + ex_events,
                total,
                min_region,
            )
        if not cur_free:
            cur_free = _complement_sample_intervals(
                ex_patch + [(s - patch_margin, e + patch_margin) for s, e in local_used],
                total,
                min_region,
            )
        if not cur_free:
            break
        nxt = _score_donor_windows_from_stats(
            cur_free,
            win_frames,
            w_rms,
            w_pk,
            w_low,
            target_len - got,
            mid_f,
            ref_rms,
            sr,
            max_donor_crest=max_donor_crest,
        )
        if nxt is None:
            break
        nb0, nb1, _ = nxt
        seg = _read_exact_slice(src_wav, nb0, nb1 - nb0, sr)
        if seg.shape[0] == 0:
            break
        chunks.append(seg)
        local_used.append((nb0, nb1))
        if used_spans is not None:
            used_spans.append((nb0, nb1))
        got += seg.shape[0]

    assembled, loops = _stitch_non_repeating_chunks(chunks, target_len, xfade)
    return assembled, round(b0 / float(sr), 3), round(b1 / float(sr), 3), loops, len(chunks)


def _stream_ambience_patch(
    src_wav: Path,
    dst_wav: Path,
    params: dict[str, Any],
    patch_log_out: list[dict[str, Any]] | None = None,
    exclude_events: list[tuple[float, float]] | None = None,
) -> None:
    info = probe_audio(src_wav)
    sr = info.sample_rate
    ch = info.channels
    total = info.frames
    default_xf_ms = float(params.get("crossfade_ms", 40.0))
    min_donor_distance_s = float(params.get("min_donor_distance_s", 5.0))
    max_donor_crest = float(params.get("max_donor_crest", 6.0))

    resolved = _resolve_patch_spans(
        total,
        sr,
        params.get("start_s"),
        params.get("end_s"),
        params.get("source_start_s"),
        default_xf_ms,
        params.get("spans"),
    )
    if not resolved:
        shutil.copy2(src_wav, dst_wav)
        return

    exclude_spans = [(r["p0"], r["p1"]) for r in resolved]
    used_spans: list[tuple[int, int]] = []
    win_frames = max(64, int(round(0.25 * sr)))
    has_auto = any(r["source_start_s"] is None for r in resolved)
    stats_cache = (
        (win_frames, *_stream_window_stats_array(src_wav, win_frames)) if has_auto else None
    )

    patches: list[tuple[int, int, np.ndarray]] = []
    ctx_frames = int(0.5 * sr)
    for r in resolved:
        p0, p1 = r["p0"], r["p1"]
        patch_len = p1 - p0
        orig_slice = _read_exact_slice(src_wav, p0, patch_len, sr)
        ctx_parts = []
        if p0 > 16:
            c0 = max(0, p0 - ctx_frames)
            ctx_parts.append(_read_exact_slice(src_wav, c0, p0 - c0, sr))
        if total - p1 > 16:
            c1 = min(total, p1 + ctx_frames)
            ctx_parts.append(_read_exact_slice(src_wav, p1, c1 - p1, sr))
        ctx = np.vstack(ctx_parts) if ctx_parts else None
        ctx_rms = float(np.sqrt(np.mean(ctx * ctx))) if ctx is not None and ctx.size else None

        if r["source_start_s"] is not None:
            s_val = max(0.0, float(r["source_start_s"]))
            src_i0 = max(0, min(total - 1, int(round(s_val * sr))))
            raw_d = _read_exact_slice(src_wav, src_i0, min(patch_len, total - src_i0), sr)
            if raw_d.size:
                donor, loops = _build_tiled_donor_with_info(raw_d, patch_len, r["desired_xf"])
            else:
                donor = np.zeros((patch_len, ch), dtype=np.float64)
                loops = 0
            donor_s0 = round(src_i0 / float(sr), 3)
            donor_s1 = round((src_i0 + raw_d.shape[0]) / float(sr), 3)
            donor_chunks = 1
            src_mode = "manual"
        else:
            donor, donor_s0, donor_s1, loops, donor_chunks = _pick_streaming_donor_with_span(
                src_wav,
                exclude_spans,
                patch_len,
                r["desired_xf"],
                exclude_events_s=exclude_events,
                used_spans=used_spans,
                target_mid_frame=0.5 * (p0 + p1),
                target_rms=ctx_rms,
                stats_cache=stats_cache,
                min_donor_distance_s=min_donor_distance_s,
                max_donor_crest=max_donor_crest,
            )
            src_mode = "auto"

        matched = _match_donor_rms_and_tilt(donor, ctx, sr)
        blended = matched.copy()
        xf_l, xf_r = r["xf_l"], r["xf_r"]
        if xf_l > 0 and orig_slice.shape[0] >= xf_l and blended.shape[0] >= xf_l:
            t_l = np.linspace(0.0, math.pi * 0.5, xf_l, endpoint=True)[:, np.newaxis]
            blended[:xf_l] = orig_slice[:xf_l] * np.cos(t_l) + matched[:xf_l] * np.sin(t_l)
        if xf_r > 0 and orig_slice.shape[0] >= xf_r and blended.shape[0] >= xf_r:
            t_r = np.linspace(0.0, math.pi * 0.5, xf_r, endpoint=True)[:, np.newaxis]
            blended[-xf_r:] = matched[-xf_r:] * np.cos(t_r) + orig_slice[-xf_r:] * np.sin(t_r)
        patches.append((p0, p1, blended))
        if patch_log_out is not None:
            quality = _donor_quality_report(
                matched, ctx_rms, donor_s0, donor_s1, exclude_events
            )
            patch_log_out.append(
                {
                    "target_start_s": round(p0 / float(sr), 3),
                    "target_end_s": round(p1 / float(sr), 3),
                    "donor_start_s": donor_s0,
                    "donor_end_s": donor_s1,
                    "source_mode": src_mode,
                    "crossfade_ms": round(float(r.get("crossfade_ms", default_xf_ms)), 1),
                    "loop_count": int(loops),
                    "donor_chunks": int(donor_chunks),
                    "donor_quality": quality,
                }
            )

    # Coalesce any overlapping back-to-back patch intervals with equal-power crossfades
    merged_patches: list[tuple[int, int, np.ndarray]] = []
    for p0, p1, blended in patches:
        if merged_patches and p0 < merged_patches[-1][1]:
            m0, m1, m_buf = merged_patches[-1]
            if p1 <= m1:
                continue
            ov = m1 - p0
            if 0 < ov <= m_buf.shape[0] and ov <= blended.shape[0]:
                t_ov = np.linspace(0.0, math.pi * 0.5, ov, endpoint=True)[:, np.newaxis]
                joined = m_buf[-ov:] * np.cos(t_ov) + blended[:ov] * np.sin(t_ov)
                combined = np.vstack([m_buf[:-ov], joined, blended[ov:]])
            else:
                combined = np.vstack([m_buf, blended[max(0, m1 - p0) :]])
            merged_patches[-1] = (m0, p1, combined)
        else:
            merged_patches.append((p0, p1, blended))

    with (
        sf.SoundFile(str(src_wav), mode="r") as fin,
        sf.SoundFile(
            str(dst_wav),
            mode="w",
            samplerate=sr,
            channels=ch,
            subtype="FLOAT",
            format=_STREAM_TEMP_FORMAT,
        ) as fout,
    ):
        cursor = 0
        block_frames = max(4096, sr * 30)
        for p0, p1, blended in merged_patches:
            if p0 > cursor:
                fin.seek(cursor)
                rem = p0 - cursor
                while rem > 0:
                    c = fin.read(min(block_frames, rem), always_2d=True, dtype="float64")
                    if c.size == 0:
                        break
                    fout.write(c)
                    rem -= c.shape[0]
            fout.write(blended)
            cursor = p1

        if cursor < total:
            fin.seek(cursor)
            while True:
                c = fin.read(block_frames, always_2d=True, dtype="float64")
                if c.size == 0:
                    break
                fout.write(c)


def _stream_window_masked_transform(
    src_wav: Path,
    dst_wav: Path,
    resolved: list[dict[str, Any]],
    mode: str,
    log_out: list[dict[str, Any]] | None = None,
) -> None:
    """Stream any unified masked STFT transform across bounded window buffers."""
    if not resolved:
        shutil.copy2(src_wav, dst_wav)
        return

    info = probe_audio(src_wav)
    sr = info.sample_rate
    ch = info.channels
    total = info.frames

    resolved_sorted = sorted(resolved, key=lambda sp: sp["start_s"])
    span_items: list[dict[str, Any]] = []
    for sp in resolved_sorted:
        fade_s = float(sp.get("fade_ms", 1000.0)) * 0.001
        fft_s = int(sp.get("n_fft", 2048)) / float(max(sr, 1))
        gap_s = float(sp.get("context_gap_s", 0.0))
        act_pad = fade_s + fft_s + 0.1
        win_pad = max(act_pad, fade_s + float(sp.get("context_s", 6.0)) + gap_s + 1.0)
        act0 = max(0, int(math.floor((sp["start_s"] - act_pad) * sr)))
        act1 = min(total, int(math.ceil((sp["end_s"] + act_pad) * sr)))
        if act1 <= act0:
            continue
        win0 = max(0, int(math.floor((sp["start_s"] - win_pad) * sr)))
        win1 = min(total, int(math.ceil((sp["end_s"] + win_pad) * sr)))
        span_items.append({"sp": sp, "act0": act0, "act1": act1, "win0": win0, "win1": win1})

    if not span_items:
        shutil.copy2(src_wav, dst_wav)
        return

    groups: list[dict[str, Any]] = []
    for item in span_items:
        if not groups:
            groups.append(
                {
                    "act0": item["act0"],
                    "act1": item["act1"],
                    "win0": item["win0"],
                    "win1": item["win1"],
                    "spans": [item["sp"]],
                }
            )
            continue
        cur = groups[-1]
        comb_win0 = min(cur["win0"], item["win0"])
        comb_win1 = max(cur["win1"], item["win1"])
        comb_dur_s = (comb_win1 - comb_win0) / float(max(sr, 1))
        if (item["act0"] <= cur["act1"] and comb_dur_s <= 90.0) or (
            item["win0"] <= cur["win1"] and comb_dur_s <= 60.0
        ):
            cur["act0"] = min(cur["act0"], item["act0"])
            cur["act1"] = max(cur["act1"], item["act1"])
            cur["win0"] = comb_win0
            cur["win1"] = comb_win1
            cur["spans"].append(item["sp"])
        else:
            groups.append(
                {
                    "act0": item["act0"],
                    "act1": item["act1"],
                    "win0": item["win0"],
                    "win1": item["win1"],
                    "spans": [item["sp"]],
                }
            )

    with (
        sf.SoundFile(str(src_wav), mode="r") as fin,
        sf.SoundFile(
            str(dst_wav),
            mode="w",
            samplerate=sr,
            channels=ch,
            subtype="FLOAT",
            format=_STREAM_TEMP_FORMAT,
        ) as fout,
    ):
        cursor = 0
        block_frames = max(4096, sr * 30)
        for grp in groups:
            g_act0 = max(cursor, grp["act0"])
            g_act1 = grp["act1"]
            if g_act1 <= g_act0:
                continue
            g_win0, g_win1 = grp["win0"], grp["win1"]
            if g_act0 > cursor:
                fin.seek(cursor)
                rem = g_act0 - cursor
                while rem > 0:
                    c = fin.read(min(block_frames, rem), always_2d=True, dtype="float64")
                    if c.size == 0:
                        break
                    fout.write(c)
                    rem -= c.shape[0]
            win_buf = _read_exact_slice(src_wav, g_win0, g_win1 - g_win0, sr)
            win0_s = g_win0 / float(sr)
            shifted_grp: list[dict[str, Any]] = []
            ref_overrides: dict[int, np.ndarray] = {}
            for s_idx, sp in enumerate(grp["spans"]):
                sh_sp = dict(sp)
                sh_sp["start_s"] = sp["start_s"] - win0_s
                sh_sp["end_s"] = sp["end_s"] - win0_s
                if sp.get("ref_start_s") is not None and sp.get("ref_end_s") is not None:
                    rs0 = max(0, int(round(float(sp["ref_start_s"]) * sr)))
                    rs1 = min(total, int(round(float(sp["ref_end_s"]) * sr)))
                    if rs1 - rs0 >= 64:
                        ref_overrides[s_idx] = _read_exact_slice(src_wav, rs0, rs1 - rs0, sr)
                    sh_sp["ref_start_s"] = None
                    sh_sp["ref_end_s"] = None
                shifted_grp.append(sh_sp)

            proc_win, grp_logs = apply_masked_stft_transform(
                win_buf,
                sr,
                shifted_grp,
                ref_overrides=ref_overrides or None,
                mode=mode,
            )
            sl0 = max(0, g_act0 - g_win0)
            sl1 = min(proc_win.shape[0], g_act1 - g_win0)
            if sl1 > sl0:
                fout.write(proc_win[sl0:sl1])
            cursor = max(cursor, g_act1)
            if log_out is not None:
                for entry in grp_logs:
                    e_copy = dict(entry)
                    e_copy["start_s"] = round(float(e_copy["start_s"]) + win0_s, 3)
                    e_copy["end_s"] = round(float(e_copy["end_s"]) + win0_s, 3)
                    log_out.append(e_copy)

        if cursor < total:
            fin.seek(cursor)
            while True:
                c = fin.read(block_frames, always_2d=True, dtype="float64")
                if c.size == 0:
                    break
                fout.write(c)


def _stream_spectral_match_suppress(
    src_wav: Path,
    dst_wav: Path,
    params: dict[str, Any],
    sup_log_out: list[dict[str, Any]] | None = None,
) -> None:
    """Stream per-span adaptive spectral excess suppression on bounded window buffers."""
    info = probe_audio(src_wav)
    resolved = _resolve_match_suppress_spans(
        info.duration_s,
        info.sample_rate,
        params.get("start_s"),
        params.get("end_s"),
        float(params.get("low_hz", 25.0)),
        float(params.get("high_hz", 4000.0)),
        float(params.get("strength", 1.0)),
        float(params.get("max_attenuation_db", -32.0)),
        float(params.get("texture_fill", 0.65)),
        float(params.get("context_s", 6.0)),
        float(params.get("context_gap_s", 1.5)),
        float(params.get("fade_ms", 1500.0)),
        int(params.get("n_fft", 4096)),
        params.get("ref_start_s"),
        params.get("ref_end_s"),
        bool(params.get("protect_birds", False)),
        str(params.get("spatial_mode", "linked")),
        bool(params.get("narrowband_mode", False)),
        params.get("spans"),
    )
    _stream_window_masked_transform(
        src_wav, dst_wav, resolved, mode="spectral_match", log_out=sup_log_out
    )


def _tile_long_spans(
    resolved: list[dict[str, Any]], max_span_s: float = 45.0
) -> list[dict[str, Any]]:
    """Split oversized whole-file spans so streaming STFT windows stay RAM-bounded."""
    tiled: list[dict[str, Any]] = []
    for sp in resolved:
        s0 = float(sp["start_s"])
        s1 = float(sp["end_s"])
        if (s1 - s0) <= max_span_s:
            tiled.append(sp)
            continue
        cur = s0
        while cur < s1 - 1e-6:
            nxt = min(s1, cur + max_span_s)
            part = dict(sp)
            part["start_s"] = cur
            part["end_s"] = nxt
            tiled.append(part)
            cur = nxt
    return tiled


def _stream_harmonic_track_suppress(
    src_wav: Path,
    dst_wav: Path,
    params: dict[str, Any],
    log_out: list[dict[str, Any]] | None = None,
) -> None:
    """Stream variable-RPM harmonic comb suppression across bounded window buffers."""
    info = probe_audio(src_wav)
    resolved = _resolve_harmonic_track_spans(
        info.duration_s,
        info.sample_rate,
        params.get("start_s"),
        params.get("end_s"),
        float(params.get("f0_min_hz", 55.0)),
        float(params.get("f0_max_hz", 300.0)),
        int(params.get("max_harmonics", 12)),
        float(params.get("bandwidth_hz", 18.0)),
        float(params.get("strength", 1.0)),
        float(params.get("max_attenuation_db", -28.0)),
        float(params.get("broadband_assist", 0.35)),
        bool(params.get("spatial_bias", True)),
        bool(params.get("protect_birds", True)),
        float(params.get("texture_fill", 0.55)),
        float(params.get("context_s", 6.0)),
        float(params.get("context_gap_s", 1.5)),
        float(params.get("fade_ms", 1000.0)),
        int(params.get("n_fft", 4096)),
        params.get("spans"),
    )
    _stream_window_masked_transform(
        src_wav,
        dst_wav,
        _tile_long_spans(resolved),
        mode="harmonic_comb",
        log_out=log_out,
    )


def _stream_distance_soften(
    src_wav: Path,
    dst_wav: Path,
    params: dict[str, Any],
    log_out: list[dict[str, Any]] | None = None,
) -> None:
    """Stream distance softening / voice muffling across bounded window buffers."""
    info = probe_audio(src_wav)
    resolved = _resolve_distance_soften_spans(
        info.duration_s,
        info.sample_rate,
        params.get("start_s"),
        params.get("end_s"),
        str(params.get("mode", "distance")),
        float(params.get("low_hz", 150.0)),
        float(params.get("high_hz", 6000.0)),
        float(params.get("attenuation_db", -14.0)),
        float(params.get("air_absorption", 0.65)),
        float(params.get("attack_smear_ms", 12.0)),
        float(params.get("formant_blur", 0.5)),
        float(params.get("narrow_stereo", 0.35)),
        bool(params.get("protect_birds", True)),
        float(params.get("context_s", 6.0)),
        float(params.get("context_gap_s", 1.5)),
        float(params.get("fade_ms", 500.0)),
        int(params.get("n_fft", 2048)),
        params.get("spans"),
    )
    _stream_window_masked_transform(
        src_wav,
        dst_wav,
        _tile_long_spans(resolved),
        mode="distance_soften",
        log_out=log_out,
    )


def _stream_declip_from_backup(
    src_wav: Path,
    dst_wav: Path,
    params: dict[str, Any],
    info_out: dict[str, Any] | None = None,
    head_pad_frames: int = 0,
) -> None:
    """Stream main-track declipping against a backup WAV in bounded RAM."""
    info = probe_audio(src_wav)
    sr = info.sample_rate
    ch = info.channels
    total = info.frames
    b_path = Path(str(params["backup_audio_path"]))
    if not b_path.exists():
        raise FileNotFoundError(f"Backup audio file not found: {b_path}")
    b_info = probe_audio(b_path)
    if b_info.sample_rate != sr or b_info.channels != ch:
        raise ValueError(
            f"Backup format mismatch ({b_info.sample_rate}Hz/{b_info.channels}ch) "
            f"vs main ({sr}Hz/{ch}ch)."
        )

    clip_threshold = float(params.get("clip_threshold", 0.985))
    crossfade_ms = float(params.get("crossfade_ms", 2.0))
    fir_taps = int(params.get("fir_taps", 31))
    max_lag_ms = float(params.get("max_lag_ms", 250.0))
    ar_fallback = bool(params.get("ar_fallback", True))

    active_end = min(total, head_pad_frames + b_info.frames)
    overlap_len = max(0, active_end - head_pad_frames)
    if overlap_len < 32:
        shutil.copy2(src_wav, dst_wav)
        if info_out is not None:
            info_out.update({"repaired_regions": 0, "repaired_samples": 0})
        return

    cal_len = min(int(60 * sr), overlap_len)
    main_cal = _read_exact_slice(src_wav, head_pad_frames, cal_len, sr)
    bk_cal = _read_exact_slice(b_path, 0, cal_len, sr)
    lag_samples, coh = estimate_gcc_phat_lag(
        main_cal[:, 0], bk_cal[:, 0], sr, max_lag_ms=max_lag_ms
    )

    cal_clip_2d = detect_clipping_mask(main_cal, threshold=clip_threshold)
    any_cal_clip = np.any(cal_clip_2d, axis=1)
    ch_coeffs: list[np.ndarray | None] = []
    ch_gains: list[float] = []
    gain_dbs: list[float] = []
    null_dbs: list[float] = []
    for c_idx in range(ch):
        b_al = fractional_shift_channel(bk_cal[:, c_idx], -lag_samples)
        valid_fit = (~any_cal_clip) & (np.abs(main_cal[:, c_idx]) > 1e-5)
        if int(np.sum(valid_fit)) >= max(64, int(fir_taps) * 4):
            coeffs, g_db, null_db = fit_least_squares_fir(
                b_al,
                main_cal[:, c_idx],
                taps=fir_taps,
                valid_mask=~cal_clip_2d[:, c_idx],
            )
            ch_coeffs.append(coeffs)
            ch_gains.append(1.0)
        else:
            rms_m = float(np.sqrt(np.mean(main_cal[:, c_idx] ** 2))) + 1e-12
            rms_b = float(np.sqrt(np.mean(b_al**2))) + 1e-12
            g = max(0.05, min(20.0, rms_m / rms_b))
            ch_coeffs.append(None)
            ch_gains.append(g)
            g_db = round(20.0 * math.log10(g), 3)
            null_db = 0.0
        gain_dbs.append(g_db)
        null_dbs.append(null_db)

    xf_n = max(1, int(round(max(0.2, crossfade_ms) * 0.001 * sr)))
    pad_n = max(fir_taps * 4, int(0.05 * sr) + xf_n + 64)
    lag_extra = int(math.ceil(abs(lag_samples))) + fir_taps + 32
    block_frames = max(4096, sr * 30)

    repaired_regions = 0
    repaired_samples = 0
    ar_fallback_regions = 0

    with (
        sf.SoundFile(str(src_wav), mode="r") as fin,
        sf.SoundFile(
            str(dst_wav),
            mode="w",
            samplerate=sr,
            channels=ch,
            subtype="FLOAT",
            format=_STREAM_TEMP_FORMAT,
        ) as fout,
    ):
        cursor = 0
        if head_pad_frames > 0:
            fin.seek(0)
            rem = head_pad_frames
            while rem > 0:
                c = fin.read(min(block_frames, rem), always_2d=True, dtype="float64")
                if c.size == 0:
                    break
                fout.write(c)
                rem -= c.shape[0]
            cursor = head_pad_frames

        while cursor < active_end:
            b0 = cursor
            b1 = min(active_end, cursor + block_frames)
            w0 = max(head_pad_frames, b0 - pad_n)
            w1 = min(active_end, b1 + pad_n)
            main_win = _read_exact_slice(src_wav, w0, w1 - w0, sr)
            clip_win = detect_clipping_mask(main_win, threshold=clip_threshold)

            chk0 = max(0, (b0 - w0) - xf_n)
            chk1 = min(w1 - w0, (b1 - w0) + xf_n)
            if not np.any(clip_win[chk0:chk1]):
                fout.write(main_win[b0 - w0 : b1 - w0])
                cursor = b1
                continue

            bk_target0 = w0 - head_pad_frames
            bk_target1 = w1 - head_pad_frames
            bk_r0 = max(0, bk_target0 - lag_extra)
            bk_r1 = min(b_info.frames, bk_target1 + lag_extra)
            bk_slice = _read_exact_slice(b_path, bk_r0, bk_r1 - bk_r0, sr)
            win_len = w1 - w0
            rep_win = main_win.copy()

            for c_idx in range(ch):
                shifted = fractional_shift_channel(bk_slice[:, c_idx], -lag_samples)
                coeffs = ch_coeffs[c_idx]
                if coeffs is not None:
                    matched_raw = signal.fftconvolve(shifted, coeffs, mode="same")
                else:
                    matched_raw = shifted * ch_gains[c_idx]

                # Map window indices [bk_target0 .. bk_target1) out of [bk_r0 .. bk_r1)
                idx_grid = np.arange(bk_target0, bk_target1, dtype=np.int64) - bk_r0
                in_b = (idx_grid >= 0) & (idx_grid < matched_raw.size)
                b_matched_win = np.zeros(win_len, dtype=np.float64)
                b_raw_aligned = np.zeros(win_len, dtype=np.float64)
                b_matched_win[in_b] = matched_raw[idx_grid[in_b]]

                raw_idx_grid = (
                    np.arange(bk_target0, bk_target1, dtype=np.int64)
                    + int(round(lag_samples))
                    - bk_r0
                )
                in_raw = (raw_idx_grid >= 0) & (raw_idx_grid < bk_slice.shape[0])
                b_raw_aligned[in_raw] = bk_slice[raw_idx_grid[in_raw], c_idx]

                rep_c, reg_c, samp_c, ar_c = _repair_clipped_channel_slice(
                    main_win[:, c_idx],
                    b_matched_win,
                    clip_win[:, c_idx],
                    b_raw_aligned,
                    0.0,
                    sr,
                    xf_n,
                    ar_fallback,
                    active_lo=b0 - w0,
                    active_hi=b1 - w0,
                )
                rep_win[:, c_idx] = rep_c
                repaired_regions += reg_c
                repaired_samples += samp_c
                ar_fallback_regions += ar_c

            fout.write(rep_win[b0 - w0 : b1 - w0])
            cursor = b1

        if cursor < total:
            fin.seek(cursor)
            while True:
                c = fin.read(block_frames, always_2d=True, dtype="float64")
                if c.size == 0:
                    break
                fout.write(c)

    if info_out is not None:
        info_out.update(
            {
                "lag_samples": finite_or_none(lag_samples, digits=3) or 0.0,
                "alignment_coherence": finite_or_none(coh, digits=3) or 0.0,
                "fitted_gain_db": gain_dbs[0] if len(gain_dbs) == 1 else gain_dbs,
                "fit_null_residual_db": null_dbs[0] if len(null_dbs) == 1 else null_dbs,
                "repaired_regions": repaired_regions,
                "repaired_samples": repaired_samples,
                "ar_fallback_regions": ar_fallback_regions,
            }
        )


def _stream_spatial_mix(
    src_wav: Path,
    dst_wav: Path,
    params: dict[str, Any],
    info_out: dict[str, Any] | None = None,
) -> None:
    """Stream deterministic M/S decode, directional blend, L/R balance, or mono stereoizer."""
    info = probe_audio(src_wav)
    sr = info.sample_rate
    ch = info.channels
    total = info.frames
    if total < 32:
        shutil.copy2(src_wav, dst_wav)
        if info_out is not None:
            info_out.update({"effective_mode": str(params.get("mode", "lr_adjust"))})
        return

    cal_len = min(int(45 * sr), total)
    cal_start = max(0, (total - cal_len) // 2) if total > cal_len else 0
    cal_audio = _read_exact_slice(src_wav, cal_start, cal_len, sr)

    mode = str(params.get("mode", "lr_adjust"))
    side_gain_db = float(params.get("side_gain_db", 0.0))
    directional_weight = float(params.get("directional_weight", 0.5))
    directional_channel = int(params.get("directional_channel", 0))
    crossover_hz = float(params.get("crossover_hz", 0.0))
    align_delay = bool(params.get("align_delay", True))
    low_mono_hz = float(params.get("low_mono_hz", 0.0))
    side_hpf_hz = float(params.get("side_hpf_hz", 0.0))
    balance_db = float(params.get("balance_db", 0.0))
    auto_balance = bool(params.get("auto_balance", False))
    decorrelate_ms = float(params.get("decorrelate_ms", 0.0))

    globals_cfg = _resolve_spatial_mix_globals(
        cal_audio,
        sr,
        mode=mode,
        directional_channel=directional_channel,
        align_delay=align_delay,
        crossover_hz=crossover_hz,
        auto_balance=auto_balance,
        balance_db=balance_db,
        decorrelate_ms=decorrelate_ms,
    )
    out_ch = (
        2
        if (ch >= 2 or globals_cfg["effective_mode"] == "dual_mono_to_stereo")
        else ch
    )
    pad_n = max(64, sr)
    block_frames = max(4096, sr * 30)

    with (
        sf.SoundFile(str(src_wav), mode="r") as fin,
        sf.SoundFile(
            str(dst_wav),
            mode="w",
            samplerate=sr,
            channels=out_ch,
            subtype="FLOAT",
            format=_STREAM_TEMP_FORMAT,
        ) as fout,
    ):
        cursor = 0
        while cursor < total:
            b0 = cursor
            b1 = min(total, cursor + block_frames)
            w0 = max(0, b0 - pad_n)
            w1 = min(total, b1 + pad_n)
            fin.seek(w0)
            win = fin.read(w1 - w0, always_2d=True, dtype="float64")
            proc = _apply_spatial_mix_chunk(
                win,
                sr,
                globals_cfg,
                side_gain_db=side_gain_db,
                directional_weight=directional_weight,
                directional_channel=directional_channel,
                crossover_hz=crossover_hz,
                low_mono_hz=low_mono_hz,
                side_hpf_hz=side_hpf_hz,
                decorrelate_ms=decorrelate_ms,
            )
            sl0 = b0 - w0
            sl1 = b1 - w0
            fout.write(proc[sl0:sl1])
            cursor = b1

    if info_out is not None:
        info_out.update(globals_cfg)


def _probe_stream_lufs_tp(wav_path: Path) -> tuple[float | None, float]:
    info = probe_audio(wav_path)
    lufs_acc = StreamingLUFS(info.sample_rate, info.channels)
    tp_acc = StreamingTruePeak(info.channels)
    for _, block in iter_audio_blocks(wav_path, block_seconds=30.0, dtype="float64"):
        lufs_acc.update(block)
        tp_acc.update(block)
    return lufs_acc.finalize(), max(tp_acc.peak_lin, 1e-12)


def _stream_loudness_normalize(
    src_wav: Path,
    dst_wav: Path,
    params: dict[str, Any],
    info_out: dict[str, Any] | None = None,
) -> None:
    info = probe_audio(src_wav)
    sr = info.sample_rate
    ch = info.channels
    target_lufs = float(params.get("target_lufs", -18.0))
    ceiling_dbtp = float(params.get("true_peak_ceiling_dbtp", -1.0))
    limit_peaks = bool(params.get("limit_peaks", False))
    rel_ms = float(params.get("limiter_release_ms", 80.0))

    cur_lufs, tp_lin = _probe_stream_lufs_tp(src_wav)
    if cur_lufs is None or not math.isfinite(cur_lufs):
        shutil.copy2(src_wav, dst_wav)
        if info_out is not None:
            info_out.update(
                {
                    "target_lufs": round(target_lufs, 2),
                    "applied_gain_db": 0.0,
                    "peak_limited": False,
                }
            )
        return

    desired_db = max(-48.0, min(48.0, target_lufs - cur_lufs))
    if not limit_peaks:
        tp_db = 20.0 * math.log10(tp_lin)
        applied_db = min(desired_db, ceiling_dbtp - tp_db)
        gain_lin = 10.0 ** (applied_db / 20.0)
        with sf.SoundFile(
            str(dst_wav),
            mode="w",
            samplerate=sr,
            channels=ch,
            subtype="FLOAT",
            format=_STREAM_TEMP_FORMAT,
        ) as fout:
            for _, block in iter_audio_blocks(src_wav, block_seconds=30.0, dtype="float64"):
                fout.write(block * gain_lin if gain_lin != 1.0 else block)
        out_lufs = cur_lufs + applied_db
        if info_out is not None:
            info_out.update(
                {
                    "input_lufs": finite_or_none(cur_lufs, digits=3),
                    "output_lufs": finite_or_none(out_lufs, digits=3),
                    "target_lufs": finite_or_none(target_lufs, digits=2),
                    "applied_gain_db": finite_or_none(applied_db, digits=3) or 0.0,
                    "peak_limited": False,
                }
            )
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        pass1_wav = tmp / "norm_p1.wav"
        _stream_local_batch(
            src_wav,
            pass1_wav,
            [
                FilterStep(filter_name="gain", params={"gain_db": desired_db}),
                FilterStep(
                    filter_name="true_peak_limiter",
                    params={"ceiling_dbtp": ceiling_dbtp, "release_ms": rel_ms},
                ),
            ],
            block_seconds=30.0,
        )
        total_gain_db = desired_db
        cur_wav = pass1_wav
        lufs_1, tp_1 = _probe_stream_lufs_tp(cur_wav)
        for pass_idx in range(2, 5):
            if lufs_1 is None or not math.isfinite(lufs_1):
                break
            deficit = target_lufs - lufs_1
            if deficit <= 0.25:
                break
            topup_db = min(18.0, deficit)
            next_pass_wav = tmp / f"norm_p{pass_idx}.wav"
            _stream_local_batch(
                cur_wav,
                next_pass_wav,
                [
                    FilterStep(filter_name="gain", params={"gain_db": topup_db}),
                    FilterStep(
                        filter_name="true_peak_limiter",
                        params={"ceiling_dbtp": ceiling_dbtp, "release_ms": rel_ms},
                    ),
                ],
                block_seconds=30.0,
            )
            prev_pass = cur_wav
            cur_wav = next_pass_wav
            prev_pass.unlink(missing_ok=True)
            total_gain_db += topup_db
            lufs_1, tp_1 = _probe_stream_lufs_tp(cur_wav)

        tp_1_db = 20.0 * math.log10(max(tp_1, 1e-12))
        if tp_1_db > ceiling_dbtp:
            trim_db = ceiling_dbtp - tp_1_db
            trim_lin = 10.0 ** (trim_db / 20.0)
            total_gain_db += trim_db
            if lufs_1 is not None:
                lufs_1 += trim_db
            with sf.SoundFile(
                str(dst_wav),
                mode="w",
                samplerate=sr,
                channels=ch,
                subtype="FLOAT",
                format=_STREAM_TEMP_FORMAT,
            ) as fout:
                for _, block in iter_audio_blocks(cur_wav, block_seconds=30.0, dtype="float64"):
                    fout.write(block * trim_lin)
        else:
            shutil.copy2(cur_wav, dst_wav)

    if info_out is not None:
        info_out.update(
            {
                "input_lufs": finite_or_none(cur_lufs, digits=3),
                "output_lufs": finite_or_none(lufs_1, digits=3) if lufs_1 is not None else None,
                "target_lufs": finite_or_none(target_lufs, digits=2),
                "applied_gain_db": finite_or_none(total_gain_db, digits=3) or 0.0,
                "peak_limited": True,
                "ceiling_dbtp": finite_or_none(ceiling_dbtp, digits=2),
            }
        )


def apply_recipe_streaming(
    src_path: str | Path,
    dst_path: str | Path,
    steps: list[FilterStep] | list[dict[str, Any]],
    subtype: str = "PCM_24",
    block_seconds: float = 60.0,
    telemetry_out: list[StepTelemetry] | None = None,
    prev_audio_path: str | Path | None = None,
    next_audio_path: str | Path | None = None,
    seam_context_s: float = 8.0,
    exclude_events: list[tuple[float, float]] | None = None,
) -> AudioInfo:
    """Apply an ordered DSP chain file-to-file in bounded RAM for long recordings."""
    src = Path(src_path)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    parsed = [s if isinstance(s, FilterStep) else FilterStep.model_validate(s) for s in steps]
    for st in parsed:
        if st.filter_name not in _FILTER_HANDLERS:
            known = ", ".join(sorted(_FILTER_HANDLERS))
            raise ValueError(f"Unknown filter '{st.filter_name}'. Available: {known}")

    orig_info = probe_audio(src)
    sr = orig_info.sample_rate
    ch = orig_info.channels
    prev_p, head_req, prev_dur, next_p, tail_req, next_dur = _prepare_seam_context(
        sr, ch, parsed, prev_audio_path, next_audio_path, seam_context_s
    )

    if not parsed:
        shutil.copy2(src, dst)
        return probe_audio(dst)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cur_wav = src
        prev_temp: Path | None = None
        head_frames = 0
        tail_frames = 0
        head_pad_s = 0.0
        working_steps = parsed
        shifted_ev = exclude_events

        if (prev_p is not None and head_req > 0) or (next_p is not None and tail_req > 0):
            padded_wav = tmp / "stage_padded.wav"
            head_pad = (
                _read_exact_edge_frames(
                    prev_p, head_req, from_end=True, fallback_dur_s=prev_dur, sr=sr
                )
                if prev_p is not None and head_req > 0
                else np.zeros((0, ch), dtype=np.float64)
            )
            tail_pad = (
                _read_exact_edge_frames(
                    next_p, tail_req, from_end=False, fallback_dur_s=next_dur, sr=sr
                )
                if next_p is not None and tail_req > 0
                else np.zeros((0, ch), dtype=np.float64)
            )
            head_frames = head_pad.shape[0]
            tail_frames = tail_pad.shape[0]
            head_pad_s = head_frames / float(sr)
            with sf.SoundFile(
                str(padded_wav),
                mode="w",
                samplerate=sr,
                channels=ch,
                subtype="FLOAT",
                format=_STREAM_TEMP_FORMAT,
            ) as fp:
                if head_frames > 0:
                    fp.write(head_pad)
                for _, block in iter_audio_blocks(src, block_seconds=30.0, dtype="float64"):
                    fp.write(block)
                if tail_frames > 0:
                    fp.write(tail_pad)
            cur_wav = padded_wav
            prev_temp = padded_wav
            if head_pad_s > 0.0:
                working_steps = [_shift_step_times(st, -head_pad_s) for st in parsed]
                if exclude_events:
                    shifted_ev = [(es + head_pad_s, ee + head_pad_s) for es, ee in exclude_events]

        stage_idx = 0
        i = 0
        prev_levels: tuple[float | None, float | None, float | None] | None = None
        while i < len(working_steps):
            st = working_steps[i]
            if st.filter_name in _LOCAL_STREAMABLE_FILTERS:
                batch_base = i
                batch: list[FilterStep] = []
                while (
                    i < len(working_steps)
                    and working_steps[i].filter_name in _LOCAL_STREAMABLE_FILTERS
                ):
                    batch.append(working_steps[i])
                    i += 1
                next_wav = tmp / f"stage_{stage_idx:03d}.wav"
                batch_tele: list[StepTelemetry] | None = [] if telemetry_out is not None else None
                _stream_local_batch(
                    cur_wav,
                    next_wav,
                    batch,
                    block_seconds=block_seconds,
                    step_telemetry_out=batch_tele,
                    base_step_index=batch_base,
                )
                if telemetry_out is not None and batch_tele:
                    for k, bt in enumerate(batch_tele):
                        bt.params = copy.deepcopy(parsed[batch_base + k].params)
                        if "attenuated_spans" in bt.details and head_pad_s > 0.0:
                            bt.details["attenuated_spans"] = _adjust_span_telemetry(
                                bt.details["attenuated_spans"], head_pad_s
                            )
                    telemetry_out.extend(batch_tele)
                    prev_levels = (
                        batch_tele[-1].output_rms_dbfs,
                        batch_tele[-1].output_peak_dbfs,
                        batch_tele[-1].output_lufs,
                    )
                cur_wav = next_wav
                if prev_temp is not None and prev_temp != src:
                    prev_temp.unlink(missing_ok=True)
                prev_temp = next_wav
                stage_idx += 1
                continue

            if telemetry_out is not None and prev_levels is None:
                prev_levels = _probe_stream_levels(cur_wav)

            next_wav = tmp / f"stage_{stage_idx:03d}.wav"
            step_details: dict[str, Any] = {}
            in_lufs = out_lufs = None
            if st.filter_name == "fade_edges":
                _stream_fade_edges(cur_wav, next_wav, st.params)
            elif st.filter_name == "trim_range":
                _stream_trim_range(cur_wav, next_wav, st.params)
            elif st.filter_name == "cut_regions":
                _stream_cut_regions(cur_wav, next_wav, st.params)
            elif st.filter_name == "ambience_patch":
                patch_log: list[dict[str, Any]] = []
                _stream_ambience_patch(
                    cur_wav,
                    next_wav,
                    st.params,
                    patch_log_out=patch_log,
                    exclude_events=shifted_ev,
                )
                step_details = {"patched_spans": _adjust_span_telemetry(patch_log, head_pad_s)}
            elif st.filter_name == "spectral_match_suppress":
                sup_log: list[dict[str, Any]] = []
                _stream_spectral_match_suppress(
                    cur_wav, next_wav, st.params, sup_log_out=sup_log
                )
                step_details = {"suppressed_spans": _adjust_span_telemetry(sup_log, head_pad_s)}
            elif st.filter_name == "harmonic_track_suppress":
                harm_log: list[dict[str, Any]] = []
                _stream_harmonic_track_suppress(
                    cur_wav, next_wav, st.params, log_out=harm_log
                )
                step_details = {"tracked_spans": _adjust_span_telemetry(harm_log, head_pad_s)}
            elif st.filter_name == "distance_soften":
                dist_log: list[dict[str, Any]] = []
                _stream_distance_soften(
                    cur_wav, next_wav, st.params, log_out=dist_log
                )
                step_details = {"softened_spans": _adjust_span_telemetry(dist_log, head_pad_s)}
            elif st.filter_name == "declip_from_backup":
                declip_info: dict[str, Any] = {}
                _stream_declip_from_backup(
                    cur_wav,
                    next_wav,
                    st.params,
                    info_out=declip_info,
                    head_pad_frames=head_frames,
                )
                step_details = declip_info
            elif st.filter_name == "spatial_mix":
                spatial_info: dict[str, Any] = {}
                _stream_spatial_mix(cur_wav, next_wav, st.params, info_out=spatial_info)
                step_details = spatial_info
            elif st.filter_name == "loudness_normalize":
                norm_info: dict[str, Any] = {}
                _stream_loudness_normalize(cur_wav, next_wav, st.params, info_out=norm_info)
                in_lufs = norm_info.pop("input_lufs", None)
                out_lufs = norm_info.pop("output_lufs", None)
                step_details = norm_info

            if telemetry_out is not None:
                out_rms, out_pk, probed_lufs = _probe_stream_levels(next_wav)
                in_rms, in_pk, prev_lufs = prev_levels or (None, None, None)
                eff_in_lufs = in_lufs if in_lufs is not None else prev_lufs
                eff_out_lufs = out_lufs if out_lufs is not None else probed_lufs
                telemetry_out.append(
                    StepTelemetry(
                        step_index=i,
                        filter_name=st.filter_name,
                        params=copy.deepcopy(parsed[i].params),
                        input_rms_dbfs=in_rms,
                        output_rms_dbfs=out_rms,
                        input_peak_dbfs=in_pk,
                        output_peak_dbfs=out_pk,
                        input_lufs=eff_in_lufs,
                        output_lufs=eff_out_lufs,
                        details=step_details,
                    )
                )
                prev_levels = (out_rms, out_pk, eff_out_lufs)

            cur_wav = next_wav
            if prev_temp is not None and prev_temp != src:
                prev_temp.unlink(missing_ok=True)
            prev_temp = next_wav
            stage_idx += 1
            i += 1

        # Materialize final output in requested subtype (slicing off any boundary padding)
        info = probe_audio(cur_wav)
        out_ch = info.channels
        padded_run = head_frames > 0 or tail_frames > 0
        out_frames = (
            min(orig_info.frames, max(0, info.frames - head_frames)) if padded_run else info.frames
        )
        bytes_per_sample = 4 if subtype == "FLOAT" else 3
        est_bytes = out_frames * out_ch * bytes_per_sample + 4096
        out_format = "RF64" if est_bytes >= 3_800_000_000 else "WAV"

        if (
            not padded_run
            and cur_wav.suffix.lower() == ".wav"
            and info.subtype == subtype
            and info.format_name.upper() == out_format
        ):
            shutil.copy2(cur_wav, dst)
        else:
            block_frames = max(4096, sr * 30)
            rem = out_frames
            with (
                sf.SoundFile(str(cur_wav), mode="r") as fin,
                sf.SoundFile(
                    str(dst),
                    mode="w",
                    samplerate=sr,
                    channels=out_ch,
                    subtype=subtype,
                    format=out_format,
                ) as fout,
            ):
                if head_frames > 0:
                    fin.seek(head_frames)
                while rem > 0:
                    to_read = min(block_frames, rem)
                    blk = fin.read(to_read, always_2d=True, dtype="float64")
                    if blk.size == 0:
                        break
                    if subtype != "FLOAT":
                        blk = np.clip(blk, -1.0, 0.999999)
                    fout.write(blk)
                    rem -= blk.shape[0]

    return probe_audio(dst)


def compute_residual(parent_audio: np.ndarray, processed_audio: np.ndarray) -> np.ndarray:
    """Return exact sample-level difference removed by processing (parent - processed)."""
    parent = ensure_2d_audio(parent_audio)
    processed = ensure_2d_audio(processed_audio)
    if parent.shape != processed.shape:
        raise ValueError(
            f"Shape mismatch computing residual: {parent.shape} vs {processed.shape}"
        )
    return parent - processed
