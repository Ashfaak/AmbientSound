"""Linked-stereo CPU DSP engine tailored for ambient soundscape restoration."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pyloudnorm as pyln
from scipy import interpolate, ndimage, signal

from ambient_sound.io_audio import ensure_2d_audio
from ambient_sound.models import FilterStep

FILTER_SPECS: list[dict[str, Any]] = [
    {
        "name": "highpass_rumble",
        "summary": (
            "Zero-phase Butterworth high-pass filter to remove subsonic wind and mic rumble."
        ),
        "ambient_guidance": (
            "Use 45-90 Hz for outdoor rain/wind/urban; keep low enough for thunder or surf."
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
            "Keep strength modest (0.2-0.45) on rain/surf to avoid metallic shimmer."
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
            "Feathered time- and frequency-localized linked-stereo STFT attenuation."
        ),
        "ambient_guidance": (
            "Use for horns, sirens, speech, or whines where surrounding ambience stays intact."
        ),
        "params": {
            "start_s": {"type": "float", "required": True},
            "end_s": {"type": "float", "required": True},
            "low_hz": {"type": "float", "required": True},
            "high_hz": {"type": "float", "required": True},
            "attenuation_db": {"type": "float", "default": -18.0, "min": -60.0, "max": 12.0},
            "fade_ms": {"type": "float", "default": 30.0, "min": 2.0, "max": 500.0},
            "n_fft": {"type": "int", "default": 2048, "min": 256, "max": 8192},
        },
    },
    {
        "name": "ambience_patch",
        "summary": (
            "Replaces a distracted span with matched clean ambient texture via crossfades."
        ),
        "ambient_guidance": (
            "Best when a cough, car pass, or bump dominates the full band."
        ),
        "params": {
            "start_s": {"type": "float", "required": True},
            "end_s": {"type": "float", "required": True},
            "source_start_s": {"type": "float | null", "default": None},
            "crossfade_ms": {"type": "float", "default": 40.0, "min": 5.0, "max": 500.0},
        },
    },
    {
        "name": "parametric_eq",
        "summary": "RBJ Audio EQ Cookbook biquad filter (peaking, lowshelf, or highshelf).",
        "ambient_guidance": (
            "Use subtle +/- 2 to 4 dB shelves/bells to tame harsh midrange or add warmth."
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
        "name": "loudness_normalize",
        "summary": "Integrated ITU-R BS.1770 LUFS normalization with true-peak ceiling.",
        "ambient_guidance": (
            "Recommended as final mastering step (-18 to -24 LUFS for ambient beds)."
        ),
        "params": {
            "target_lufs": {"type": "float", "default": -18.0, "min": -40.0, "max": -6.0},
            "true_peak_ceiling_dbtp": {"type": "float", "default": -1.0, "min": -6.0, "max": -0.1},
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
    # Smooth gain across time and frequency to avoid musical artifacts
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


def _band_attenuate(
    audio: np.ndarray,
    sample_rate: int,
    start_s: float,
    end_s: float,
    low_hz: float,
    high_hz: float,
    attenuation_db: float = -18.0,
    fade_ms: float = 30.0,
    n_fft: int = 2048,
) -> np.ndarray:
    n_samples, _ = audio.shape
    s_t = max(0.0, float(start_s))
    e_t = min(n_samples / float(sample_rate), float(end_s))
    f_lo = max(0.0, float(low_hz))
    f_hi = min(sample_rate * 0.5, float(high_hz))
    if n_samples < 64 or e_t <= s_t or f_hi <= f_lo:
        return audio.copy()

    freqs, times, z_list, nperseg, noverlap = _stft_channels(audio, sample_rate, n_fft)

    fade_s = max(0.002, float(fade_ms) * 0.001)
    freq_fade = max(25.0, 0.1 * (f_hi - f_lo))

    w_t = _cosine_edge_mask(times, s_t, e_t, fade_s)
    w_f = _cosine_edge_mask(freqs, f_lo, f_hi, freq_fade)
    mask_2d = np.outer(w_f, w_t)

    atten_clamped = max(-60.0, min(12.0, float(attenuation_db)))
    target_lin = 10.0 ** (atten_clamped / 20.0)
    gain = 1.0 - mask_2d * (1.0 - target_lin)

    masked = [z * gain for z in z_list]
    processed = _istft_channels(masked, sample_rate, nperseg, noverlap, n_samples)

    # Keep unedited regions bit-exact via soft time window blend
    pad_s = fade_s + (nperseg / float(sample_rate))
    i0 = max(0, int((s_t - pad_s) * sample_rate))
    i1 = min(n_samples, int((e_t + pad_s) * sample_rate))
    out = audio.copy()
    out[i0:i1] = processed[i0:i1]
    return out


def _build_tiled_donor(donor: np.ndarray, target_len: int, xfade: int) -> np.ndarray:
    """Create a seamless chunk of length target_len by crossfade-tiling donor."""
    if donor.shape[0] >= target_len:
        return donor[:target_len].copy()

    n_channels = donor.shape[1]
    if donor.shape[0] <= 2 * max(xfade, 1):
        # Simple wrap tile if donor is tiny
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


def _pick_clean_donor(
    audio: np.ndarray,
    sample_rate: int,
    exclude_start: int,
    exclude_end: int,
    target_len: int,
    xfade: int,
) -> np.ndarray:
    """Select the lowest-crest, smoothest clean segment outside [exclude_start, exclude_end]."""
    n_samples, _ = audio.shape
    margin = int(0.05 * sample_rate)
    ex0 = max(0, exclude_start - margin)
    ex1 = min(n_samples, exclude_end + margin)

    candidates: list[np.ndarray] = []
    if ex0 > int(0.15 * sample_rate):
        candidates.append(audio[:ex0])
    if n_samples - ex1 > int(0.15 * sample_rate):
        candidates.append(audio[ex1:])
    if not candidates:
        # Fallback if edited span covers almost the entire file
        left = audio[:exclude_start]
        right = audio[exclude_end:]
        fallback = left if left.shape[0] >= right.shape[0] else right
        if fallback.shape[0] == 0:
            return np.zeros((target_len, audio.shape[1]), dtype=np.float64)
        return _build_tiled_donor(fallback, target_len, xfade)

    # Compute reference ambient RMS from 100 ms frames across whole file
    win_rms = max(32, int(0.1 * sample_rate))
    mono = np.mean(audio, axis=1)
    hop_rms = max(16, win_rms // 2)
    frame_rms: list[float] = []
    for p in range(0, max(1, mono.size - win_rms + 1), hop_rms):
        seg = mono[p : p + win_rms]
        frame_rms.append(float(np.sqrt(np.mean(seg * seg))))
    global_med_rms = (
        float(np.median(frame_rms))
        if frame_rms
        else float(np.sqrt(np.mean(mono * mono)))
    )

    best_score = float("inf")
    best_chunk = candidates[0]
    probe_len = min(target_len, max(int(0.25 * sample_rate), min(c.shape[0] for c in candidates)))

    for region in candidates:
        if region.shape[0] <= probe_len:
            chunk = region
            mono_c = np.mean(chunk, axis=1)
            rms = float(np.sqrt(np.mean(mono_c * mono_c))) + 1e-12
            crest = float(np.max(np.abs(mono_c))) / rms
            score = crest + 3.0 * abs(rms - global_med_rms) / max(global_med_rms, 1e-6)
            if score < best_score:
                best_score = score
                best_chunk = chunk
            continue

        step = max(32, probe_len // 4)
        for p in range(0, region.shape[0] - probe_len + 1, step):
            chunk = region[p : p + probe_len]
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
                best_chunk = chunk

    return _build_tiled_donor(best_chunk, target_len, xfade)


def _ambience_patch(
    audio: np.ndarray,
    sample_rate: int,
    start_s: float,
    end_s: float,
    source_start_s: float | None = None,
    crossfade_ms: float = 40.0,
) -> np.ndarray:
    n_samples, _ = audio.shape
    i0 = max(0, min(n_samples, int(round(float(start_s) * sample_rate))))
    i1 = max(0, min(n_samples, int(round(float(end_s) * sample_rate))))
    if i1 <= i0 or n_samples < 32:
        return audio.copy()

    desired_xf = max(1, int(float(crossfade_ms) * 0.001 * sample_rate))
    xf_l = min(desired_xf, i0, max(1, (i1 - i0) // 2))
    xf_r = min(desired_xf, n_samples - i1, max(1, (i1 - i0) // 2))

    p0 = max(0, i0 - xf_l // 2)
    p1 = min(n_samples, i1 + xf_r // 2)
    patch_len = p1 - p0

    if source_start_s is not None:
        src_i0 = max(0, min(n_samples - 1, int(round(float(source_start_s) * sample_rate))))
        raw_donor = audio[src_i0 : min(n_samples, src_i0 + patch_len)]
        if raw_donor.shape[0] == 0:
            raw_donor = audio
        donor = _build_tiled_donor(raw_donor, patch_len, desired_xf)
    else:
        donor = _pick_clean_donor(audio, sample_rate, p0, p1, patch_len, desired_xf)

    # Match donor level gently to clean local boundaries around [p0, p1]
    ctx_len = int(0.1 * sample_rate)
    ctx_parts = []
    if p0 > 16:
        ctx_parts.append(audio[max(0, p0 - ctx_len) : p0])
    if n_samples - p1 > 16:
        ctx_parts.append(audio[p1 : min(n_samples, p1 + ctx_len)])
    if ctx_parts:
        ctx = np.vstack(ctx_parts)
        ctx_rms = float(np.sqrt(np.mean(ctx * ctx)))
        donor_rms = float(np.sqrt(np.mean(donor * donor)))
        if donor_rms > 1e-9 and ctx_rms > 1e-9:
            scale = max(0.5, min(2.0, ctx_rms / donor_rms))
            donor = donor * scale

    out = audio.copy()
    blended = donor.copy()

    if xf_l > 0:
        t_l = np.linspace(0.0, math.pi * 0.5, xf_l, endpoint=True)[:, np.newaxis]
        blended[:xf_l] = out[p0 : p0 + xf_l] * np.cos(t_l) + donor[:xf_l] * np.sin(t_l)
    if xf_r > 0:
        t_r = np.linspace(0.0, math.pi * 0.5, xf_r, endpoint=True)[:, np.newaxis]
        blended[-xf_r:] = donor[-xf_r:] * np.cos(t_r) + out[p1 - xf_r : p1] * np.sin(t_r)

    out[p0:p1] = blended
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
    f0 = max(20.0, min(float(freq_hz), sample_rate * 0.45))
    gain = max(-24.0, min(24.0, float(gain_db)))
    if abs(gain) < 1e-4:
        return audio.copy()

    q_val = max(0.1, min(20.0, float(q)))
    a_lin = 10.0 ** (gain / 40.0)
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
        # Shelf slope formulation using S derived from q_val
        sqrt_a = math.sqrt(a_lin)
        slope_term = (a_lin + 1.0 / a_lin) * (1.0 / max(q_val, 0.2) - 1.0) + 2.0
        alpha_s = (sin_w0 / 2.0) * math.sqrt(slope_term)
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
    sos = signal.tf2sos(b, a)
    return signal.sosfilt(sos, audio, axis=0)


def _loudness_normalize(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = -18.0,
    true_peak_ceiling_dbtp: float = -1.0,
) -> np.ndarray:
    duration_s = audio.shape[0] / float(max(sample_rate, 1))
    rms = float(np.sqrt(np.mean(audio * audio)))
    if duration_s < 0.45 or rms <= 1e-10:
        return audio.copy()

    meter = pyln.Meter(sample_rate)
    current_lufs = float(meter.integrated_loudness(audio))
    if not math.isfinite(current_lufs):
        return audio.copy()

    desired_gain_db = float(target_lufs) - current_lufs
    upsampled = signal.resample_poly(audio, 4, 1, axis=0) if audio.shape[0] >= 4 else audio
    tp_lin = max(float(np.max(np.abs(upsampled))), 1e-12)
    tp_db = 20.0 * math.log10(tp_lin)
    max_safe_gain_db = float(true_peak_ceiling_dbtp) - tp_db
    applied_gain_db = min(desired_gain_db, max_safe_gain_db)

    return audio * (10.0 ** (applied_gain_db / 20.0))


_FILTER_HANDLERS = {
    "highpass_rumble": _highpass_rumble,
    "dehum_notch": _dehum_notch,
    "declick_impulse": _declick_impulse,
    "spectral_denoise": _spectral_denoise,
    "band_attenuate": _band_attenuate,
    "ambience_patch": _ambience_patch,
    "parametric_eq": _parametric_eq,
    "loudness_normalize": _loudness_normalize,
}


def apply_recipe(
    audio: np.ndarray,
    sample_rate: int,
    steps: list[FilterStep] | list[dict[str, Any]],
) -> np.ndarray:
    """Apply an ordered list of FilterStep items non-destructively."""
    out = ensure_2d_audio(audio).copy()
    for item in steps:
        step = item if isinstance(item, FilterStep) else FilterStep.model_validate(item)
        handler = _FILTER_HANDLERS.get(step.filter_name)
        if handler is None:
            known = ", ".join(sorted(_FILTER_HANDLERS))
            raise ValueError(f"Unknown filter '{step.filter_name}'. Available: {known}")
        out = ensure_2d_audio(handler(out, int(sample_rate), **step.params))
    return out


def compute_residual(parent_audio: np.ndarray, processed_audio: np.ndarray) -> np.ndarray:
    """Return exact sample-level difference removed by processing (parent - processed)."""
    parent = ensure_2d_audio(parent_audio)
    processed = ensure_2d_audio(processed_audio)
    if parent.shape != processed.shape:
        raise ValueError(
            f"Shape mismatch computing residual: {parent.shape} vs {processed.shape}"
        )
    return parent - processed
