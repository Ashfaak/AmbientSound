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
from ambient_sound.io_audio import (
    ensure_2d_audio,
    iter_audio_blocks,
    load_audio_slice,
    probe_audio,
)
from ambient_sound.models import AudioInfo, FilterStep, StepTelemetry

FILTER_SPECS: list[dict[str, Any]] = [
    {
        "name": "highpass_rumble",
        "summary": (
            "Zero-phase Butterworth high-pass filter to remove subsonic wind and mic rumble."
        ),
        "ambient_guidance": (
            "Use 45-90 Hz for outdoor rain/wind/urban; 80-120 Hz clears traffic rumble "
            "below 200-300 Hz bee buzz."
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
            "MAD spike detector + monotonicity-preserving PCHIP interpolation for mic clicks."
        ),
        "ambient_guidance": (
            "Ideal for rain drips on mic housing, static pops, or branch snaps cleanly."
        ),
        "params": {
            "threshold_sigma": {"type": "float", "default": 6.0, "min": 3.0, "max": 20.0},
            "max_width_ms": {"type": "float", "default": 3.0, "min": 0.5, "max": 12.0},
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
            "Replaces distracted time span(s) with matched clean ambient texture via crossfades."
        ),
        "ambient_guidance": (
            "Supports crossfade_ms up to 5000 ms and batch spans=[{start_s, end_s, "
            "crossfade_ms?, source_start_s?}] for per-span crossfades and explicit donors. "
            "Auto-selects clean donor outside all target spans and records donor provenance."
        ),
        "params": {
            "start_s": {"type": "float | null", "default": None},
            "end_s": {"type": "float | null", "default": None},
            "source_start_s": {"type": "float | null", "default": None},
            "crossfade_ms": {"type": "float", "default": 40.0, "min": 5.0, "max": 5000.0},
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
            "Place after loudness_normalize (e.g. -1.0 dBTP) for YouTube readiness."
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
) -> np.ndarray:
    n_samples, n_channels = audio.shape
    if n_samples < 32:
        return audio.copy()

    out = audio.copy()
    cutoff = min(1500.0, sample_rate * 0.35)
    sos = signal.butter(4, cutoff, btype="highpass", fs=sample_rate, output="sos")
    hp = _safe_sosfiltfilt(sos, out)

    max_width_samples = max(4, int(float(max_width_ms) * 0.001 * sample_rate))
    merge_gap = max(1, int(0.001 * sample_rate))
    dilate = max(1, int(0.0005 * sample_rate))

    for ch in range(n_channels):
        env = np.abs(hp[:, ch])
        med = float(np.median(env))
        mad = float(np.median(np.abs(env - med))) * 1.4826
        sigma = max(mad, 1e-7)
        hits = np.flatnonzero(((env - med) / sigma > float(threshold_sigma)) & (env > 2e-3))
        if hits.size == 0:
            continue

        spans: list[tuple[int, int]] = []
        s = e = int(hits[0])
        for idx in hits[1:]:
            i = int(idx)
            if i - e <= merge_gap:
                e = i
            else:
                spans.append((s, e))
                s = e = i
        spans.append((s, e))

        for s_raw, e_raw in spans:
            s_i = max(1, s_raw - dilate)
            e_i = min(n_samples - 2, e_raw + dilate)
            if (e_i - s_i + 1) > max_width_samples:
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
    n_samples, _ = audio.shape
    dur_s = n_samples / float(max(sample_rate, 1))
    resolved = _resolve_band_spans(
        dur_s, start_s, end_s, low_hz, high_hz, attenuation_db, fade_ms, envelope_mode, spans
    )
    if n_samples < 64 or not resolved:
        return audio.copy()

    freqs, times, z_list, nperseg, noverlap = _stft_channels(audio, sample_rate, n_fft)
    gain_total = np.ones((freqs.size, times.size), dtype=np.float64)

    active_ranges: list[tuple[int, int]] = []
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

        pad_s = fade_s + (nperseg / float(sample_rate))
        i0 = max(0, int((sp["start_s"] - pad_s) * sample_rate))
        i1 = min(n_samples, int((sp["end_s"] + pad_s) * sample_rate))
        active_ranges.append((i0, i1))

    if not active_ranges:
        return audio.copy()

    masked = [z * gain_total for z in z_list]
    processed = _istft_channels(masked, sample_rate, nperseg, noverlap, n_samples)

    out = audio.copy()
    for i0, i1 in active_ranges:
        out[i0:i1] = processed[i0:i1]
    return out


def _build_tiled_donor(donor: np.ndarray, target_len: int, xfade: int) -> np.ndarray:
    """Create a seamless chunk of length target_len by crossfade-tiling donor."""
    if donor.shape[0] >= target_len:
        return donor[:target_len].copy()

    n_channels = donor.shape[1]
    if donor.shape[0] <= 2 * max(xfade, 1):
        reps = math.ceil(target_len / donor.shape[0])
        return np.tile(donor, (reps, 1))[:target_len]

    xf = min(xfade, donor.shape[0] // 4)
    step = donor.shape[0] - xf
    out = np.zeros((target_len + donor.shape[0], n_channels), dtype=np.float64)

    t_xf = np.linspace(0.0, math.pi * 0.5, xf, endpoint=True)[:, np.newaxis]
    fade_in = np.sin(t_xf)
    fade_out = np.cos(t_xf)

    pos = 0
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
    return out[:target_len]


def _pick_clean_donor_with_span(
    audio: np.ndarray,
    sample_rate: int,
    exclude_spans: list[tuple[int, int]],
    target_len: int,
    xfade: int,
) -> tuple[np.ndarray, float, float]:
    """Select the lowest-crest clean segment outside excluded intervals and return span (s)."""
    n_samples, _ = audio.shape
    margin = int(0.05 * sample_rate)
    merged_ex: list[tuple[int, int]] = []
    for s, e in sorted(exclude_spans):
        s_m = max(0, s - margin)
        e_m = min(n_samples, e + margin)
        if merged_ex and s_m <= merged_ex[-1][1]:
            merged_ex[-1] = (merged_ex[-1][0], max(merged_ex[-1][1], e_m))
        else:
            merged_ex.append((s_m, e_m))

    candidates: list[tuple[int, int]] = []
    cur = 0
    for s_m, e_m in merged_ex:
        if s_m - cur > int(0.15 * sample_rate):
            candidates.append((cur, s_m))
        cur = max(cur, e_m)
    if n_samples - cur > int(0.15 * sample_rate):
        candidates.append((cur, n_samples))

    if not candidates:
        ex0, ex1 = merged_ex[0] if merged_ex else (0, n_samples)
        fb0, fb1 = (0, ex0) if ex0 >= (n_samples - ex1) else (ex1, n_samples)
        fallback = audio[fb0:fb1]
        if fallback.shape[0] == 0:
            return np.zeros((target_len, audio.shape[1]), dtype=np.float64), 0.0, 0.0
        return (
            _build_tiled_donor(fallback, target_len, xfade),
            round(fb0 / float(sample_rate), 3),
            round(fb1 / float(sample_rate), 3),
        )

    win_rms = max(32, int(0.1 * sample_rate))
    mono = np.mean(audio, axis=1)
    hop_rms = max(16, win_rms // 2)
    frame_rms = [
        float(np.sqrt(np.mean(mono[p : p + win_rms] ** 2)))
        for p in range(0, max(1, mono.size - win_rms + 1), hop_rms)
    ]
    global_med_rms = (
        float(np.median(frame_rms))
        if frame_rms
        else float(np.sqrt(np.mean(mono * mono)))
    )

    best_score = float("inf")
    best_bounds = candidates[0]
    probe_len = min(target_len, max(int(0.25 * sample_rate), min(e - s for s, e in candidates)))

    for r_s, r_e in candidates:
        region_len = r_e - r_s
        if region_len <= probe_len:
            chunk = audio[r_s:r_e]
            mono_c = np.mean(chunk, axis=1)
            rms = float(np.sqrt(np.mean(mono_c * mono_c))) + 1e-12
            crest = float(np.max(np.abs(mono_c))) / rms
            score = crest + 3.0 * abs(rms - global_med_rms) / max(global_med_rms, 1e-6)
            if score < best_score:
                best_score = score
                best_bounds = (r_s, r_e)
            continue

        step = max(32, probe_len // 4)
        for p in range(0, region_len - probe_len + 1, step):
            abs_s = r_s + p
            abs_e = abs_s + probe_len
            chunk = audio[abs_s:abs_e]
            mono_c = np.mean(chunk, axis=1)
            rms = float(np.sqrt(np.mean(mono_c * mono_c))) + 1e-12
            crest = float(np.max(np.abs(mono_c))) / rms
            hf_spike = float(np.max(np.abs(np.diff(mono_c)))) / rms if mono_c.size > 1 else 0.0
            score = (
                crest
                + 0.35 * hf_spike
                + 3.0 * abs(rms - global_med_rms) / max(global_med_rms, 1e-6)
            )
            if score < best_score:
                best_score = score
                best_bounds = (abs_s, abs_e)

    b0, b1 = best_bounds
    return (
        _build_tiled_donor(audio[b0:b1], target_len, xfade),
        round(b0 / float(sample_rate), 3),
        round(b1 / float(sample_rate), 3),
    )


def _pick_clean_donor(
    audio: np.ndarray,
    sample_rate: int,
    exclude_spans: list[tuple[int, int]],
    target_len: int,
    xfade: int,
) -> np.ndarray:
    """Select the lowest-crest, smoothest clean segment outside all excluded sample intervals."""
    donor, _, _ = _pick_clean_donor_with_span(audio, sample_rate, exclude_spans, target_len, xfade)
    return donor


def _apply_patch_slice(
    out: np.ndarray,
    ref_audio: np.ndarray,
    sample_rate: int,
    p0: int,
    p1: int,
    xf_l: int,
    xf_r: int,
    donor: np.ndarray,
) -> None:
    """Splice donor into out[p0:p1] with RMS boundary match and equal-power crossfades."""
    n_samples = ref_audio.shape[0]
    ctx_len = int(0.1 * sample_rate)
    ctx_parts = []
    if p0 > 16:
        ctx_parts.append(ref_audio[max(0, p0 - ctx_len) : p0])
    if n_samples - p1 > 16:
        ctx_parts.append(ref_audio[p1 : min(n_samples, p1 + ctx_len)])
    if ctx_parts:
        ctx = np.vstack(ctx_parts)
        ctx_rms = float(np.sqrt(np.mean(ctx * ctx)))
        donor_rms = float(np.sqrt(np.mean(donor * donor)))
        if donor_rms > 1e-9 and ctx_rms > 1e-9:
            scale = max(0.5, min(2.0, ctx_rms / donor_rms))
            donor = donor * scale

    blended = donor.copy()
    if xf_l > 0:
        t_l = np.linspace(0.0, math.pi * 0.5, xf_l, endpoint=True)[:, np.newaxis]
        blended[:xf_l] = out[p0 : p0 + xf_l] * np.cos(t_l) + donor[:xf_l] * np.sin(t_l)
    if xf_r > 0:
        t_r = np.linspace(0.0, math.pi * 0.5, xf_r, endpoint=True)[:, np.newaxis]
        blended[-xf_r:] = donor[-xf_r:] * np.cos(t_r) + out[p1 - xf_r : p1] * np.sin(t_r)

    out[p0:p1] = blended


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
    spans: list[dict[str, Any]] | None = None,
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
    out = audio.copy()
    patch_log: list[dict[str, Any]] = []

    for r in resolved:
        p0, p1 = r["p0"], r["p1"]
        patch_len = p1 - p0
        if r["source_start_s"] is not None:
            s_val = float(r["source_start_s"])
            src_i0 = max(0, min(n_samples - 1, int(round(s_val * sample_rate))))
            src_i1 = min(n_samples, src_i0 + patch_len)
            raw_donor = audio[src_i0:src_i1]
            if raw_donor.shape[0] == 0:
                raw_donor = audio
            donor = _build_tiled_donor(raw_donor, patch_len, r["desired_xf"])
            donor_s0 = round(src_i0 / float(sample_rate), 3)
            donor_s1 = round(src_i1 / float(sample_rate), 3)
            src_mode = "manual"
        else:
            donor, donor_s0, donor_s1 = _pick_clean_donor_with_span(
                audio, sample_rate, exclude_spans, patch_len, r["desired_xf"]
            )
            src_mode = "auto"

        _apply_patch_slice(out, audio, sample_rate, p0, p1, r["xf_l"], r["xf_r"], donor)
        patch_log.append(
            {
                "target_start_s": round(p0 / float(sample_rate), 3),
                "target_end_s": round(p1 / float(sample_rate), 3),
                "donor_start_s": donor_s0,
                "donor_end_s": donor_s1,
                "source_mode": src_mode,
                "crossfade_ms": round(float(r.get("crossfade_ms", crossfade_ms)), 1),
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
    spans: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    out, _ = _ambience_patch_with_info(
        audio, sample_rate, start_s, end_s, source_start_s, crossfade_ms, spans
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
            "target_lufs": round(float(target_lufs), 2),
            "applied_gain_db": round(applied_gain_db, 3),
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
        "target_lufs": round(float(target_lufs), 2),
        "applied_gain_db": round(total_gain_db, 3),
        "peak_limited": True,
        "ceiling_dbtp": round(ceiling, 2),
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


_FILTER_HANDLERS = {
    "highpass_rumble": _highpass_rumble,
    "dehum_notch": _dehum_notch,
    "drone_notch": _drone_notch,
    "declick_impulse": _declick_impulse,
    "spectral_denoise": _spectral_denoise,
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
}


def _array_rms_peak_db(audio: np.ndarray) -> tuple[float | None, float | None]:
    if audio.size == 0:
        return None, None
    rms_lin = float(np.sqrt(np.mean(audio * audio)))
    pk_lin = float(np.max(np.abs(audio)))
    rms_db = round(20.0 * math.log10(rms_lin), 4) if rms_lin > 0.0 else None
    pk_db = round(20.0 * math.log10(pk_lin), 4) if pk_lin > 0.0 else None
    return rms_db, pk_db


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
            out, patch_log = _ambience_patch_with_info(out, sr, **step.params)
            details = {"patched_spans": patch_log}
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
    if step.filter_name == "band_attenuate":
        if params.get("start_s") is not None:
            params["start_s"] = float(params["start_s"]) - offset_s
        if params.get("end_s") is not None:
            params["end_s"] = float(params["end_s"]) - offset_s
        if params.get("spans"):
            for sp in params["spans"]:
                sp["start_s"] = float(sp["start_s"]) - offset_s
                sp["end_s"] = float(sp["end_s"]) - offset_s
    elif step.filter_name == "spectral_denoise":
        if params.get("noise_start_s") is not None:
            params["noise_start_s"] = float(params["noise_start_s"]) - offset_s
        if params.get("noise_end_s") is not None:
            params["noise_end_s"] = float(params["noise_end_s"]) - offset_s
    return FilterStep(filter_name=step.filter_name, params=params)


def _stats_from_accum(
    sum_sq: float, peak_lin: float, count: int
) -> tuple[float | None, float | None]:
    if count <= 0:
        return None, None
    rms_lin = math.sqrt(sum_sq / float(count))
    rms_db = round(20.0 * math.log10(rms_lin), 4) if rms_lin > 0.0 else None
    pk_db = round(20.0 * math.log10(peak_lin), 4) if peak_lin > 0.0 else None
    return rms_db, pk_db


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
    for st in steps:
        if st.filter_name == "band_attenuate":
            max_fade_s = max(max_fade_s, float(st.params.get("fade_ms", 30.0)) * 0.001)
            for sp in st.params.get("spans") or []:
                max_fade_s = max(max_fade_s, float(sp.get("fade_ms", 30.0)) * 0.001)
    pad_s = min(8.0, max(2.0, max_fade_s + 1.0))

    n_steps = len(steps)
    stage_sum_sq = [0.0] * (n_steps + 1)
    stage_peak = [0.0] * (n_steps + 1)
    stage_count = [0] * (n_steps + 1)
    lufs_accs = (
        [StreamingLUFS(sr, ch) for _ in range(n_steps + 1)]
        if step_telemetry_out is not None
        else []
    )

    block_frames = max(1024, int(round(block_seconds * sr)))
    with sf.SoundFile(
        str(dst_wav),
        mode="w",
        samplerate=sr,
        channels=ch,
        subtype="FLOAT",
        format="WAV",
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
                    cur = ensure_2d_audio(_FILTER_HANDLERS[sh.filter_name](cur, sr, **sh.params))
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
                )
            )


def _stream_fade_edges(src_wav: Path, dst_wav: Path, params: dict[str, Any]) -> None:
    info = probe_audio(src_wav)
    sr = info.sample_rate
    ch = info.channels
    total = info.frames
    fade_in_n = min(total, max(0, int(round(float(params.get("fade_in_s", 2.0)) * sr))))
    fade_out_n = min(total, max(0, int(round(float(params.get("fade_out_s", 3.0)) * sr))))
    curve = str(params.get("curve", "cosine"))

    with sf.SoundFile(
        str(dst_wav),
        mode="w",
        samplerate=sr,
        channels=ch,
        subtype="FLOAT",
        format="WAV",
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
        format="WAV",
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
            str(dst_wav), mode="w", samplerate=sr, channels=ch, subtype="FLOAT", format="WAV"
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


def _pick_streaming_donor_with_span(
    src_wav: Path,
    exclude_spans: list[tuple[int, int]],
    target_len: int,
    xfade: int,
) -> tuple[np.ndarray, float, float]:
    """Locate a clean donor slice on disk with bounded RAM and return (donor, start_s, end_s)."""
    info = probe_audio(src_wav)
    sr = info.sample_rate
    ch = info.channels
    total = info.frames
    margin = int(0.05 * sr)

    merged_ex: list[tuple[int, int]] = []
    for s, e in sorted(exclude_spans):
        sm, em = max(0, s - margin), min(total, e + margin)
        if merged_ex and sm <= merged_ex[-1][1]:
            merged_ex[-1] = (merged_ex[-1][0], max(merged_ex[-1][1], em))
        else:
            merged_ex.append((sm, em))

    free_spans: list[tuple[int, int]] = []
    cur = 0
    for sm, em in merged_ex:
        if sm - cur > int(0.25 * sr):
            free_spans.append((cur, sm))
        cur = max(cur, em)
    if total - cur > int(0.25 * sr):
        free_spans.append((cur, total))

    if not free_spans:
        return np.zeros((target_len, ch), dtype=np.float64), 0.0, 0.0

    probe_len = min(target_len, max(int(0.5 * sr), min(e - s for s, e in free_spans)))
    # Stream 0.25 s window RMS & crest across free spans
    best_score = float("inf")
    best_sample_start = free_spans[0][0]
    win_n = max(64, int(0.25 * sr))
    rms_samples: list[float] = []
    candidates: list[tuple[int, float, float]] = []

    for fs0, fs1 in free_spans:
        for abs_f, blk in iter_audio_blocks(
            src_wav,
            block_seconds=30.0,
            start_s=fs0 / float(sr),
            end_s=fs1 / float(sr),
            dtype="float32",
        ):
            mono = np.mean(blk, axis=1)
            n_w = mono.size // win_n
            for j in range(n_w):
                seg = mono[j * win_n : (j + 1) * win_n]
                rms = float(np.sqrt(np.mean(seg * seg))) + 1e-12
                crest = float(np.max(np.abs(seg))) / rms
                rms_samples.append(rms)
                start_f = abs_f + j * win_n
                if start_f + probe_len <= fs1:
                    candidates.append((start_f, rms, crest))

    med_rms = float(np.median(rms_samples)) if rms_samples else 0.01
    for start_f, rms, crest in candidates:
        score = crest + 3.0 * abs(rms - med_rms) / max(med_rms, 1e-6)
        if score < best_score:
            best_score = score
            best_sample_start = start_f

    donor_s0 = best_sample_start / float(sr)
    donor_s1 = min(info.duration_s, (best_sample_start + probe_len) / float(sr))
    raw_donor, _ = load_audio_slice(
        src_wav,
        start_s=donor_s0,
        end_s=donor_s1,
        dtype="float64",
    )
    return (
        _build_tiled_donor(raw_donor, target_len, xfade),
        round(donor_s0, 3),
        round(donor_s1, 3),
    )


def _pick_streaming_donor(
    src_wav: Path,
    exclude_spans: list[tuple[int, int]],
    target_len: int,
    xfade: int,
) -> np.ndarray:
    """Locate a clean donor slice on disk with bounded RAM using a streaming RMS scan."""
    donor, _, _ = _pick_streaming_donor_with_span(src_wav, exclude_spans, target_len, xfade)
    return donor


def _stream_ambience_patch(
    src_wav: Path,
    dst_wav: Path,
    params: dict[str, Any],
    patch_log_out: list[dict[str, Any]] | None = None,
) -> None:
    info = probe_audio(src_wav)
    sr = info.sample_rate
    ch = info.channels
    total = info.frames
    default_xf_ms = float(params.get("crossfade_ms", 40.0))

    resolved = _resolve_patch_spans(
        total,
        sr,
        params.get("start_s"),
        params.get("end_s"),
        params.get("source_start_s"),
        default_xf_ms,
        params.get("spans"),
    )
    exclude_spans = [(r["p0"], r["p1"]) for r in resolved]
    # Merge or sequence sorted non-overlapping patches
    patches: list[tuple[int, int, np.ndarray]] = []
    for r in resolved:
        p0, p1 = r["p0"], r["p1"]
        patch_len = p1 - p0
        if r["source_start_s"] is not None:
            s_t = max(0.0, float(r["source_start_s"]))
            e_t = min(info.duration_s, s_t + patch_len / float(sr))
            raw_d, _ = load_audio_slice(
                src_wav,
                start_s=s_t,
                end_s=e_t,
                dtype="float64",
            )
            donor = (
                _build_tiled_donor(raw_d, patch_len, r["desired_xf"])
                if raw_d.size
                else np.zeros((patch_len, ch))
            )
            donor_s0, donor_s1, src_mode = round(s_t, 3), round(e_t, 3), "manual"
        else:
            donor, donor_s0, donor_s1 = _pick_streaming_donor_with_span(
                src_wav, exclude_spans, patch_len, r["desired_xf"]
            )
            src_mode = "auto"

        # Boundary context slice around [p0, p1]
        orig_slice, _ = load_audio_slice(
            src_wav, start_s=p0 / float(sr), end_s=p1 / float(sr), dtype="float64"
        )
        ctx_l, _ = load_audio_slice(
            src_wav, start_s=max(0.0, (p0 - int(0.1 * sr)) / float(sr)), end_s=p0 / float(sr)
        )
        ctx_r, _ = load_audio_slice(
            src_wav,
            start_s=p1 / float(sr),
            end_s=min(info.duration_s, (p1 + int(0.1 * sr)) / float(sr)),
        )
        ctx_parts = [c for c in (ctx_l, ctx_r) if c.shape[0] > 16]
        if ctx_parts:
            ctx = np.vstack(ctx_parts)
            ctx_rms = float(np.sqrt(np.mean(ctx * ctx)))
            d_rms = float(np.sqrt(np.mean(donor * donor)))
            if d_rms > 1e-9 and ctx_rms > 1e-9:
                donor = donor * max(0.5, min(2.0, ctx_rms / d_rms))

        blended = donor.copy()
        xf_l, xf_r = r["xf_l"], r["xf_r"]
        if xf_l > 0 and orig_slice.shape[0] >= xf_l:
            t_l = np.linspace(0.0, math.pi * 0.5, xf_l, endpoint=True)[:, np.newaxis]
            blended[:xf_l] = orig_slice[:xf_l] * np.cos(t_l) + donor[:xf_l] * np.sin(t_l)
        if xf_r > 0 and orig_slice.shape[0] >= xf_r:
            t_r = np.linspace(0.0, math.pi * 0.5, xf_r, endpoint=True)[:, np.newaxis]
            blended[-xf_r:] = donor[-xf_r:] * np.cos(t_r) + orig_slice[-xf_r:] * np.sin(t_r)
        patches.append((p0, p1, blended))
        if patch_log_out is not None:
            patch_log_out.append(
                {
                    "target_start_s": round(p0 / float(sr), 3),
                    "target_end_s": round(p1 / float(sr), 3),
                    "donor_start_s": donor_s0,
                    "donor_end_s": donor_s1,
                    "source_mode": src_mode,
                    "crossfade_ms": round(float(r.get("crossfade_ms", default_xf_ms)), 1),
                }
            )

    with (
        sf.SoundFile(str(src_wav), mode="r") as fin,
        sf.SoundFile(
            str(dst_wav), mode="w", samplerate=sr, channels=ch, subtype="FLOAT", format="WAV"
        ) as fout,
    ):
        cursor = 0
        block_frames = max(4096, sr * 30)
        for p0, p1, blended in patches:
            if p0 > cursor:
                fin.seek(cursor)
                rem = p0 - cursor
                while rem > 0:
                    c = fin.read(min(block_frames, rem), always_2d=True, dtype="float64")
                    if c.size == 0:
                        break
                    fout.write(c)
                    rem -= c.shape[0]
            elif p0 < cursor:
                # Overlapping patch: skip already-written prefix
                skip = cursor - p0
                if skip < blended.shape[0]:
                    fout.write(blended[skip:])
                cursor = max(cursor, p1)
                continue
            fout.write(blended)
            cursor = p1

        if cursor < total:
            fin.seek(cursor)
            while True:
                c = fin.read(block_frames, always_2d=True, dtype="float64")
                if c.size == 0:
                    break
                fout.write(c)


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
            str(dst_wav), mode="w", samplerate=sr, channels=ch, subtype="FLOAT", format="WAV"
        ) as fout:
            for _, block in iter_audio_blocks(src_wav, block_seconds=30.0, dtype="float64"):
                fout.write(block * gain_lin if gain_lin != 1.0 else block)
        out_lufs = cur_lufs + applied_db
        if info_out is not None:
            info_out.update(
                {
                    "input_lufs": round(cur_lufs, 3),
                    "output_lufs": round(out_lufs, 3),
                    "target_lufs": round(target_lufs, 2),
                    "applied_gain_db": round(applied_db, 3),
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
            cur_wav = next_pass_wav
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
                str(dst_wav), mode="w", samplerate=sr, channels=ch, subtype="FLOAT", format="WAV"
            ) as fout:
                for _, block in iter_audio_blocks(cur_wav, block_seconds=30.0, dtype="float64"):
                    fout.write(block * trim_lin)
        else:
            shutil.copy2(cur_wav, dst_wav)

    if info_out is not None:
        info_out.update(
            {
                "input_lufs": round(cur_lufs, 3),
                "output_lufs": round(lufs_1, 3) if lufs_1 is not None else None,
                "target_lufs": round(target_lufs, 2),
                "applied_gain_db": round(total_gain_db, 3),
                "peak_limited": True,
                "ceiling_dbtp": round(ceiling_dbtp, 2),
            }
        )


def apply_recipe_streaming(
    src_path: str | Path,
    dst_path: str | Path,
    steps: list[FilterStep] | list[dict[str, Any]],
    subtype: str = "PCM_24",
    block_seconds: float = 60.0,
    telemetry_out: list[StepTelemetry] | None = None,
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

    if not parsed:
        shutil.copy2(src, dst)
        return probe_audio(dst)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cur_wav = src
        stage_idx = 0
        i = 0
        prev_levels: tuple[float | None, float | None, float | None] | None = None
        while i < len(parsed):
            st = parsed[i]
            if st.filter_name in _LOCAL_STREAMABLE_FILTERS:
                batch_base = i
                batch: list[FilterStep] = []
                while i < len(parsed) and parsed[i].filter_name in _LOCAL_STREAMABLE_FILTERS:
                    batch.append(parsed[i])
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
                    telemetry_out.extend(batch_tele)
                    prev_levels = (
                        batch_tele[-1].output_rms_dbfs,
                        batch_tele[-1].output_peak_dbfs,
                        batch_tele[-1].output_lufs,
                    )
                cur_wav = next_wav
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
                _stream_ambience_patch(cur_wav, next_wav, st.params, patch_log_out=patch_log)
                step_details = {"patched_spans": patch_log}
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
                        params=copy.deepcopy(st.params),
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
            stage_idx += 1
            i += 1

        # Materialize final output in requested subtype
        info = probe_audio(cur_wav)
        if cur_wav.suffix.lower() == ".wav" and info.subtype == subtype:
            shutil.copy2(cur_wav, dst)
        else:
            with sf.SoundFile(
                str(dst),
                mode="w",
                samplerate=info.sample_rate,
                channels=info.channels,
                subtype=subtype,
                format="WAV",
            ) as fout:
                for _, block in iter_audio_blocks(cur_wav, block_seconds=30.0, dtype="float64"):
                    if subtype != "FLOAT" and block.size:
                        block = np.clip(block, -1.0, 0.999999)
                    fout.write(block)

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
