"""Shared DSP, psychoacoustic, spatial, ridge-tracking, and detection engines."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import linalg, ndimage, signal

from ambient_sound.io_audio import ensure_2d_audio
from ambient_sound.models import DistractionEvent


def finite_or_none(value: float | np.floating | None, digits: int = 4) -> float | None:
    """Convert numeric scalar to rounded finite float or None."""
    if value is None:
        return None
    val = float(value)
    if not math.isfinite(val):
        return None
    return round(val, digits)


def db_from_linear(value: float, digits: int = 4) -> float | None:
    """Convert positive finite linear amplitude to dBFS or return None."""
    if value <= 0.0 or not math.isfinite(value):
        return None
    return finite_or_none(20.0 * math.log10(value), digits=digits)


def cosine_edge_mask(coords: np.ndarray, start: float, end: float, fade: float) -> np.ndarray:
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


def _merge_bool_spans(
    mask: np.ndarray,
    times: np.ndarray,
    frame_dur_s: float,
    max_gap_s: float,
) -> list[tuple[int, int, float, float]]:
    """Merge True frames within max_gap_s and return (idx0, idx1, start_s, end_s)."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    spans: list[tuple[int, int, float, float]] = []
    i0 = int(idx[0])
    i1 = i0
    s_t = float(times[i0])
    e_t = s_t + frame_dur_s
    for k in idx[1:]:
        ik = int(k)
        tk_s = float(times[ik])
        tk_e = tk_s + frame_dur_s
        if tk_s - e_t <= max_gap_s:
            i1 = ik
            e_t = max(e_t, tk_e)
        else:
            spans.append((i0, i1, s_t, e_t))
            i0 = i1 = ik
            s_t, e_t = tk_s, tk_e
    spans.append((i0, i1, s_t, e_t))
    return spans


def _stft_2d(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], int, int]:
    arr = ensure_2d_audio(audio)
    n_samples, n_ch = arr.shape
    nperseg = max(64, min(int(n_fft), n_samples))
    noverlap = nperseg - max(1, nperseg // 4)
    z_list: list[np.ndarray] = []
    freqs = np.zeros(0, dtype=np.float64)
    times = np.zeros(0, dtype=np.float64)
    for ch in range(n_ch):
        freqs, times, zxx = signal.stft(
            arr[:, ch],
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            boundary="zeros",
            padded=True,
        )
        z_list.append(zxx)
    return freqs, times, z_list, nperseg, noverlap


def _istft_2d(
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


def _tiled_array(src: np.ndarray, target_len: int, xfade: int) -> np.ndarray:
    if src.shape[0] >= target_len:
        return src[:target_len].copy()
    if src.shape[0] <= 2 * max(xfade, 1):
        reps = math.ceil(target_len / max(src.shape[0], 1))
        return np.tile(src, (reps, 1))[:target_len]
    xf = min(xfade, src.shape[0] // 4)
    step = src.shape[0] - xf
    out = np.zeros((target_len + src.shape[0], src.shape[1]), dtype=np.float64)
    t_xf = np.linspace(0.0, math.pi * 0.5, xf, endpoint=True)[:, np.newaxis]
    fade_in = np.sin(t_xf)
    fade_out = np.cos(t_xf)
    pos = 0
    first = True
    while pos < target_len:
        if not first and xf > 0:
            out[pos : pos + xf] = out[pos : pos + xf] * fade_out + src[:xf] * fade_in
            out[pos + xf : pos + src.shape[0]] = src[xf:]
        else:
            out[pos : pos + src.shape[0]] = src
            first = False
        pos += step
    return out[:target_len]


# ---------------------------------------------------------------------------
# 1. Channel Layout, Alignment & Clipping Engine
# ---------------------------------------------------------------------------


def detect_channel_layout_from_stats(
    channels: int,
    stereo_correlation: float | None,
    mid_side_ratio_db: float | None,
    channel_balance_db: float | None,
    ch2_low_loss_db: float | None = None,
    low_freq_coherence: float | None = None,
    directivity_diff_db: float | None = None,
) -> dict[str, Any]:
    """Classify recording channel layout from channel statistics."""
    if channels <= 1:
        return {
            "layout": "mono",
            "confidence": 1.0,
            "channel_balance_db": None,
            "stereo_correlation": None,
            "low_freq_coherence": None,
            "ch2_low_loss_db": None,
            "evidence": "Single-channel mono recording.",
        }

    corr = 0.0 if stereo_correlation is None else float(stereo_correlation)
    bal = 0.0 if channel_balance_db is None else float(channel_balance_db)
    ms_ratio = 0.0 if mid_side_ratio_db is None else float(mid_side_ratio_db)
    low_loss = 0.0 if ch2_low_loss_db is None else float(ch2_low_loss_db)
    lf_coh = corr if low_freq_coherence is None else float(low_freq_coherence)
    dir_diff = 0.0 if directivity_diff_db is None else float(directivity_diff_db)

    if corr > 0.995 and abs(bal) < 0.35 and ms_ratio > 28.0:
        return {
            "layout": "dual_mono",
            "confidence": 0.95,
            "channel_balance_db": finite_or_none(bal, 3),
            "stereo_correlation": finite_or_none(corr, 4),
            "low_freq_coherence": finite_or_none(lf_coh, 4),
            "ch2_low_loss_db": finite_or_none(low_loss, 2),
            "evidence": "Channels are near-identical (e.g. dual-mono shotgun feed).",
        }

    # Raw M/S: Ch0=Mid, Ch1=Side -> near-zero correlation, Side quieter, figure-8 bass rolloff
    if abs(corr) < 0.25 and 0.8 <= bal <= 24.0 and low_loss >= 2.0:
        return {
            "layout": "raw_ms",
            "confidence": 0.88,
            "channel_balance_db": finite_or_none(bal, 3),
            "stereo_correlation": finite_or_none(corr, 4),
            "low_freq_coherence": finite_or_none(lf_coh, 4),
            "ch2_low_loss_db": finite_or_none(low_loss, 2),
            "evidence": (
                f"Low M/S correlation ({corr:.2f}), Ch2 quieter (-{bal:.1f} dB) with "
                f"{low_loss:.1f} dB figure-8 low-band rolloff -> raw Mid/Side."
            ),
        }

    if abs(bal) >= 4.0 or (abs(corr) < 0.85 and abs(dir_diff) >= 2.8):
        return {
            "layout": "directional_plus_surround",
            "confidence": 0.78,
            "channel_balance_db": finite_or_none(bal, 3),
            "stereo_correlation": finite_or_none(corr, 4),
            "low_freq_coherence": finite_or_none(lf_coh, 4),
            "ch2_low_loss_db": finite_or_none(low_loss, 2),
            "evidence": (
                f"Asymmetric channel level/crest ({bal:+.1f} dB, diff {dir_diff:.1f} dB) -> "
                "directional + surround pair."
            ),
        }

    if corr >= 0.18 and abs(bal) < 4.5:
        conf = 0.92 if lf_coh >= 0.3 else 0.80
        return {
            "layout": "stereo_lr",
            "confidence": conf,
            "channel_balance_db": finite_or_none(bal, 3),
            "stereo_correlation": finite_or_none(corr, 4),
            "low_freq_coherence": finite_or_none(lf_coh, 4),
            "ch2_low_loss_db": finite_or_none(low_loss, 2),
            "evidence": f"Positive L/R correlation ({corr:.2f}) and balanced channels.",
        }

    return {
        "layout": "ambiguous",
        "confidence": 0.45,
        "channel_balance_db": finite_or_none(bal, 3),
        "stereo_correlation": finite_or_none(corr, 4),
        "low_freq_coherence": finite_or_none(lf_coh, 4),
        "ch2_low_loss_db": finite_or_none(low_loss, 2),
        "evidence": "Uncorrelated or ambiguous channels; specify explicit spatial_mix mode.",
    }


def detect_channel_layout(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Inspect a multi-channel array and infer mono, dual_mono, raw_ms, dir+sur, or L/R."""
    arr = ensure_2d_audio(audio)
    n_samples, ch = arr.shape
    if ch <= 1 or n_samples < 32:
        return detect_channel_layout_from_stats(ch, None, None, None)

    l_ch, r_ch = arr[:, 0], arr[:, 1]
    l_zm = l_ch - np.mean(l_ch)
    r_zm = r_ch - np.mean(r_ch)
    denom = float(np.linalg.norm(l_zm) * np.linalg.norm(r_zm))
    corr = max(-1.0, min(1.0, float(np.dot(l_zm, r_zm) / denom))) if denom > 1e-12 else 0.0

    rms_l = float(np.sqrt(np.mean(l_ch * l_ch)))
    rms_r = float(np.sqrt(np.mean(r_ch * r_ch)))
    bal_db = 20.0 * math.log10(max(rms_l, 1e-12) / max(rms_r, 1e-12))

    mid = 0.5 * (l_ch + r_ch)
    side = 0.5 * (l_ch - r_ch)
    mid_rms = float(np.sqrt(np.mean(mid * mid)))
    side_rms = float(np.sqrt(np.mean(side * side)))
    ms_db = 20.0 * math.log10(max(mid_rms, 1e-12) / max(side_rms, 1e-12))

    ch2_low_loss_db = 0.0
    lf_coh = corr
    dir_diff = 0.0
    if sample_rate >= 1200 and n_samples >= 512:
        nperseg = min(4096, n_samples)
        freqs, p1 = signal.welch(l_ch, fs=sample_rate, nperseg=nperseg)
        _, p2 = signal.welch(r_ch, fs=sample_rate, nperseg=nperseg)
        _, p12 = signal.csd(l_ch, r_ch, fs=sample_rate, nperseg=nperseg)
        low_m = (freqs >= 30.0) & (freqs <= 250.0)
        mid_m = (freqs >= 500.0) & (freqs <= min(4000.0, sample_rate * 0.45))
        if np.any(low_m) and np.any(mid_m):
            c1_lm = 10.0 * math.log10(
                (float(np.mean(p1[low_m])) + 1e-24) / (float(np.mean(p1[mid_m])) + 1e-24)
            )
            c2_lm = 10.0 * math.log10(
                (float(np.mean(p2[low_m])) + 1e-24) / (float(np.mean(p2[mid_m])) + 1e-24)
            )
            ch2_low_loss_db = c1_lm - c2_lm
        lf_band = (freqs >= 30.0) & (freqs <= 200.0)
        if np.any(lf_band):
            num = float(np.sum(np.abs(p12[lf_band])))
            den = math.sqrt(float(np.sum(p1[lf_band])) * float(np.sum(p2[lf_band]))) + 1e-24
            lf_coh = max(0.0, min(1.0, num / den))

        # Crest / high-band directivity comparison between ch0 and ch1
        crest1 = 20.0 * math.log10(max(float(np.max(np.abs(l_ch))), 1e-12) / max(rms_l, 1e-12))
        crest2 = 20.0 * math.log10(max(float(np.max(np.abs(r_ch))), 1e-12) / max(rms_r, 1e-12))
        dir_diff = (crest1 - crest2) + abs(bal_db) * 0.5

    return detect_channel_layout_from_stats(
        ch, corr, ms_db, bal_db, ch2_low_loss_db, lf_coh, dir_diff
    )


def estimate_gcc_phat_lag(
    ref_ch: np.ndarray,
    target_ch: np.ndarray,
    sample_rate: int,
    max_lag_ms: float = 250.0,
) -> tuple[float, float]:
    """Estimate sub-sample delay (lag > 0 means target occurs AFTER ref) via GCC-PHAT."""
    r = np.asarray(ref_ch, dtype=np.float64).ravel()
    y = np.asarray(target_ch, dtype=np.float64).ravel()
    n = min(r.size, y.size)
    if n < 32:
        return 0.0, 0.0
    r = r[:n] - np.mean(r[:n])
    y = y[:n] - np.mean(y[:n])
    if float(np.max(np.abs(r))) < 1e-12 or float(np.max(np.abs(y))) < 1e-12:
        return 0.0, 0.0

    # Cap correlation slice to 12 s for speed
    max_n = min(n, int(sample_rate * 12))
    r = r[:max_n]
    y = y[:max_n]

    n_fft = 1 << math.ceil(math.log2(max(64, max_n * 2)))
    fr = np.fft.rfft(r, n=n_fft)
    fy = np.fft.rfft(y, n=n_fft)
    # R = conj(fr) * fy -> peak at +D when y[n] = r[n - D]
    cross = np.conj(fr) * fy
    mag = np.abs(cross)
    cc = np.fft.irfft(cross / np.maximum(mag, 1e-12), n=n_fft)

    max_shift = min(n_fft // 2 - 2, max(1, int(round(max_lag_ms * 0.001 * sample_rate))))
    neg_idx = np.arange(n_fft - max_shift, n_fft)
    pos_idx = np.arange(0, max_shift + 1)
    search_vals = np.concatenate([cc[neg_idx], cc[pos_idx]])
    search_lags = np.arange(-max_shift, max_shift + 1, dtype=np.float64)

    peak_i = int(np.argmax(search_vals))
    peak_val = float(search_vals[peak_i])
    lag = float(search_lags[peak_i])

    if 0 < peak_i < search_vals.size - 1:
        y0, y1, y2 = float(search_vals[peak_i - 1]), peak_val, float(search_vals[peak_i + 1])
        denom = y0 - 2.0 * y1 + y2
        if denom < -1e-12:
            delta = max(-0.5, min(0.5, 0.5 * (y0 - y2) / denom))
            lag += delta

    rms_cc = float(np.sqrt(np.mean(search_vals * search_vals))) + 1e-12
    coherence = max(0.0, min(1.0, peak_val / max(rms_cc * 6.0, 1e-6)))
    return lag, coherence


def fractional_shift_channel(x: np.ndarray, shift_samples: float) -> np.ndarray:
    """Shift a 1D signal right by +shift_samples (y[n] = x[n - shift]) with cubic interp."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    n = arr.size
    if n == 0 or abs(shift_samples) < 1e-4:
        return arr.copy()
    if abs(shift_samples) >= n:
        return np.zeros_like(arr)
    idx_orig = np.arange(n, dtype=np.float64)
    query = idx_orig - float(shift_samples)
    out = np.interp(query, idx_orig, arr, left=0.0, right=0.0)
    # Cubic Catmull-Rom refinement inside interior away from boundaries
    frac = query - np.floor(query)
    if np.any(np.abs(frac) > 1e-4):
        i0 = np.floor(query).astype(np.int64)
        valid = (i0 >= 1) & (i0 <= n - 3)
        if np.any(valid):
            k = i0[valid]
            t = frac[valid]
            p0, p1, p2, p3 = arr[k - 1], arr[k], arr[k + 1], arr[k + 2]
            t2 = t * t
            t3 = t2 * t
            out[valid] = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
    return out


def fit_least_squares_fir(
    src: np.ndarray,
    dst: np.ndarray,
    taps: int = 31,
    ridge: float = 1e-4,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """Fit a zero-centered least-squares FIR mapping aligned src -> dst."""
    s = np.asarray(src, dtype=np.float64).ravel()
    d = np.asarray(dst, dtype=np.float64).ravel()
    n = min(s.size, d.size)
    vm = np.asarray(valid_mask, dtype=bool).ravel()[:n] if valid_mask is not None else None
    k_taps = max(3, int(taps) | 1)
    half = k_taps // 2
    if n < k_taps + 16:
        s_v = s[:n][vm] if vm is not None and int(np.sum(vm)) >= 4 else s[:n]
        d_v = d[:n][vm] if vm is not None and int(np.sum(vm)) >= 4 else d[:n]
        rms_s = float(np.sqrt(np.mean(s_v * s_v))) + 1e-12
        rms_d = float(np.sqrt(np.mean(d_v * d_v))) + 1e-12
        gain = max(0.05, min(20.0, rms_d / rms_s))
        coeffs = np.zeros(k_taps, dtype=np.float64)
        coeffs[half] = gain
        resid = d_v - s_v * gain
        null_db = 20.0 * math.log10(
            max(float(np.sqrt(np.mean(resid * resid))), 1e-12) / max(rms_d, 1e-12)
        )
        return (
            coeffs,
            finite_or_none(20.0 * math.log10(gain), digits=3) or 0.0,
            finite_or_none(null_db, digits=2) or 0.0,
        )

    # Cap fit length while skipping excluded/clipped rows
    step = max(1, (n - k_taps) // 180_000)
    win = np.lib.stride_tricks.sliding_window_view(s[:n], k_taps)[::step, ::-1]
    target = d[half : n - half][::step]
    m = min(win.shape[0], target.size)
    win = win[:m]
    target = target[:m]
    if vm is not None and vm.size >= n:
        vm_sub = vm[half : n - half][::step][:m]
        if int(np.sum(vm_sub)) >= k_taps + 8:
            win = win[vm_sub]
            target = target[vm_sub]

    xtx = win.T @ win
    scale = float(np.mean(np.diag(xtx))) if k_taps > 0 else 1.0
    reg = max(1e-12, scale * float(ridge))
    xtx[np.diag_indices_from(xtx)] += reg
    xty = win.T @ target
    coeffs = np.linalg.solve(xtx, xty)

    pred = signal.fftconvolve(s[:n], coeffs, mode="same")
    resid = d[:n] - pred
    if vm is not None and vm.size >= n and int(np.sum(vm)) >= 16:
        dst_rms = float(np.sqrt(np.mean(d[:n][vm] ** 2))) + 1e-12
        res_rms = float(np.sqrt(np.mean(resid[vm] ** 2))) + 1e-12
    else:
        dst_rms = float(np.sqrt(np.mean(d[:n] ** 2))) + 1e-12
        res_rms = float(np.sqrt(np.mean(resid**2))) + 1e-12
    dc_gain = max(abs(float(np.sum(coeffs))), 1e-9)
    rms_gain = float(np.sqrt(np.sum(coeffs * coeffs)))
    gain_db = 20.0 * math.log10(max(dc_gain, rms_gain, 1e-9))
    null_db = 20.0 * math.log10(max(res_rms / dst_rms, 1e-9))
    return (
        coeffs,
        finite_or_none(gain_db, digits=3) or 0.0,
        finite_or_none(null_db, digits=2) or 0.0,
    )


def detect_clipping_mask(
    audio: np.ndarray,
    threshold: float = 0.985,
    plateau_min_len: int = 3,
    plateau_floor: float = 0.85,
) -> np.ndarray:
    """Detect both digital saturation (|x| >= threshold) and flat analog plateaus."""
    arr = ensure_2d_audio(audio)
    n_samples, n_ch = arr.shape
    mask = np.abs(arr) >= float(threshold)
    min_run = max(2, int(plateau_min_len))
    if n_samples >= min_run and plateau_floor < threshold:
        diff_flat = np.abs(np.diff(arr, axis=0)) <= 1e-4
        high_amp = (np.abs(arr[:-1]) >= float(plateau_floor)) & (
            np.abs(arr[1:]) >= float(plateau_floor)
        )
        step_flat = diff_flat & high_amp
        req_steps = min_run - 1
        kernel = np.ones(req_steps, dtype=np.int64)
        for ch in range(n_ch):
            conv = np.convolve(step_flat[:, ch].astype(np.int64), kernel, mode="valid")
            starts = np.flatnonzero(conv >= req_steps)
            for s in starts:
                mask[s : s + req_steps + 1, ch] = True
    return mask


# ---------------------------------------------------------------------------
# 2. Content-Defined Bird-Protection & Harmonic Comb Tracking Engine
# ---------------------------------------------------------------------------


def compute_bird_protect_mask(
    freqs: np.ndarray,
    times: np.ndarray,
    mag: np.ndarray,
) -> np.ndarray:
    """Build content-defined (F, T) bird/chirp protection mask in [0, 1].

    Distinguishes FM-modulated bird syllables and organic trills (across all
    avian frequencies 300 Hz - 11.5 kHz) from stationary zero-drift beeps,
    steady machinery drones, broadband noise, and low mechanical harmonic stacks.
    """
    n_f, n_t = mag.shape
    if n_f < 8 or n_t < 3:
        return np.zeros((n_f, n_t), dtype=np.float64)

    db = 20.0 * np.log10(np.maximum(mag, 1e-12))
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 20.0
    dt = float(times[1] - times[0]) if times.size > 1 else 0.02

    k_f = max(5, min(n_f | 1, int(round(450.0 / max(df, 1.0))) | 1))
    med_f = ndimage.median_filter(db, size=(k_f, 1), mode="nearest")
    prom_f = db - med_f

    bed_t = np.percentile(db, 25.0, axis=1, keepdims=True)
    excess_t = db - bed_t

    cand = (prom_f >= 7.5) & (excess_t >= 5.0)
    f_band = (freqs >= 300.0) & (freqs <= min(11500.0, float(freqs[-1]) * 0.9))
    cand &= f_band[:, np.newaxis]

    # Exclude persistent stationary lines (> 40% active frames across same narrow strip)
    active_frac = np.mean((prom_f >= 6.0) & (excess_t >= 4.0), axis=1, keepdims=True)
    cand &= active_frac < 0.38

    # Downweight frames with dense low-frequency (< 1.2 kHz) engine/bark harmonic stacks
    low_bins = freqs <= 1200.0
    if np.any(low_bins):
        low_peak_cnt = np.sum((prom_f[low_bins] >= 8.0) & (excess_t[low_bins] >= 6.0), axis=0)
        low_dense = low_peak_cnt >= 4
        if np.any(low_dense):
            cand[np.ix_(low_bins, low_dense)] = False

    # Connected component check: reject flat zero-drift rectangular beeps (df ~ 0 & flat amp)
    labeled, n_comp = ndimage.label(cand, structure=np.ones((3, 3), dtype=int))
    protect = np.zeros_like(db, dtype=np.float64)
    if n_comp > 0:
        slices = ndimage.find_objects(labeled)
        for comp_id, sl in enumerate(slices, start=1):
            if sl is None:
                continue
            f_sl, t_sl = sl
            sub = labeled[f_sl, t_sl] == comp_id
            f_span_bins = sub.shape[0]
            t_span_frames = sub.shape[1]
            dur_comp_s = t_span_frames * dt

            # Track per-frame peak bin inside component to check frequency modulation
            t_start_i = t_sl.start or 0
            f_start_i = f_sl.start or 0
            sub_db = db[f_sl, t_sl]
            peak_rows = []
            peak_amps = []
            peak_hz = []
            for tj in range(t_span_frames):
                rows = np.flatnonzero(sub[:, tj])
                if rows.size > 0:
                    best_r = int(rows[np.argmax(sub_db[rows, tj])])
                    peak_rows.append(best_r)
                    peak_amps.append(float(sub_db[best_r, tj]))
                    g_r = f_start_i + best_r
                    g_t = t_start_i + tj
                    peak_hz.append(_parabolic_peak_freq(freqs, db[:, g_t], g_r))

            if len(peak_rows) >= 4:
                row_ptp = max(peak_rows) - min(peak_rows)
                f_center = float(np.median(peak_hz))
                f_ptp = max(peak_hz) - min(peak_hz)
                amp_arr = np.asarray(peak_amps, dtype=np.float64)
                mid_lo = max(0, len(amp_arr) // 5)
                mid_hi = min(len(amp_arr), max(mid_lo + 2, (4 * len(amp_arr)) // 5))
                mid_std = float(np.std(amp_arr[mid_lo:mid_hi]))
                # Zero-drift flat piezo beep: stays at constant frequency (even with Hann skirt)
                if (
                    f_center >= 1400.0
                    and dur_comp_s >= 0.045
                    and row_ptp <= 1
                    and f_ptp <= max(18.0, 0.008 * f_center)
                ):
                    continue
                if row_ptp <= 1 and f_span_bins <= 3 and mid_std < 1.4 and dur_comp_s >= 0.04:
                    continue

            # Reject very long unmodulated tones (> 2.2 s)
            if dur_comp_s > 2.2 and f_span_bins <= 2:
                continue

            protect[f_sl, t_sl][sub] = 1.0

    # Dilate gently around bird syllables to preserve harmonic skirt & onset/release
    protect = ndimage.grey_dilation(protect, size=(3, 3))
    protect = ndimage.gaussian_filter(protect, sigma=(0.8, 1.2), mode="nearest")
    peak_p = float(np.max(protect))
    if peak_p > 1e-6:
        protect = np.clip(protect / max(peak_p, 1.0) * 1.15, 0.0, 1.0)
    return protect


def track_harmonic_comb(
    freqs: np.ndarray,
    times: np.ndarray,
    mag: np.ndarray,
    f0_min_hz: float = 55.0,
    f0_max_hz: float = 300.0,
    max_harmonics: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Track sweeping fundamental f0(t) and harmonic comb confidence across frames."""
    n_f, n_t = mag.shape
    f0_out = np.zeros(n_t, dtype=np.float64)
    conf_out = np.zeros(n_t, dtype=np.float64)
    if n_f < 16 or n_t < 3:
        return f0_out, conf_out

    max_f = min(4500.0, float(freqs[-1]) * 0.9)
    sub_mask = (freqs >= 30.0) & (freqs <= max_f)
    if np.sum(sub_mask) < 12:
        return f0_out, conf_out
    f_sub = freqs[sub_mask]
    db_sub = 20.0 * np.log10(np.maximum(mag[sub_mask], 1e-12))

    df = float(f_sub[1] - f_sub[0]) if f_sub.size > 1 else 10.0
    k_f = max(5, min(f_sub.size | 1, int(round(120.0 / max(df, 1.0))) | 1))
    med_f = ndimage.median_filter(db_sub, size=(k_f, 1), mode="nearest")
    bed_t = np.percentile(db_sub, 20.0, axis=1, keepdims=True)
    # Require both local spectral peak prominence AND time-domain excess over bed
    prom = np.maximum(0.0, db_sub - med_f)
    excess = np.maximum(0.0, db_sub - bed_t)
    comb_score_bins = np.minimum(prom, excess)  # (F_sub, T)

    f0_grid = np.geomspace(max(30.0, f0_min_hz), max(f0_min_hz + 5.0, f0_max_hz), 48)
    n_h = max(3, int(max_harmonics))
    scores = np.zeros((f0_grid.size, n_t), dtype=np.float64)
    active_counts = np.zeros((f0_grid.size, n_t), dtype=np.float64)

    for g_idx, f0 in enumerate(f0_grid):
        w_sum = 0.0
        for h in range(1, n_h + 1):
            fh = f0 * h
            if fh > f_sub[-1] - df:
                break
            b_idx = int(np.argmin(np.abs(f_sub - fh)))
            lo = max(0, b_idx - 1)
            hi = min(f_sub.size, b_idx + 2)
            band_peak = np.max(comb_score_bins[lo:hi], axis=0)
            w = 1.0 / math.sqrt(float(h))
            scores[g_idx] += w * np.minimum(band_peak, 25.0)
            active_counts[g_idx] += (band_peak >= 4.5).astype(np.float64)
            w_sum += w
        if w_sum > 0.0:
            scores[g_idx] /= w_sum

    best_idx = np.argmax(scores, axis=0)
    best_scores = scores[best_idx, np.arange(n_t)]
    best_cnt = active_counts[best_idx, np.arange(n_t)]
    raw_f0 = f0_grid[best_idx]

    # Confidence requires >= 3 harmonics clearly standing out over bed
    conf = np.clip((best_scores - 2.5) / 10.0, 0.0, 1.0) * np.clip((best_cnt - 2.0) / 3.0, 0.0, 1.0)

    dt = float(times[1] - times[0]) if times.size > 1 else 0.02
    k_med = max(3, min(n_t | 1, int(round(0.25 / max(dt, 1e-3))) | 1))
    active_t = conf >= 0.28
    f0_clean = raw_f0.copy()
    if np.sum(active_t) >= 3:
        act_idx = np.flatnonzero(active_t)
        n_act = int(act_idx.size)
        k_cap = n_act if (n_act % 2 == 1) else max(1, n_act - 1)
        k_sub = max(1, min(k_med, k_cap))
        if k_sub >= 3:
            f0_clean[act_idx] = signal.medfilt(raw_f0[act_idx], kernel_size=k_sub)
        if act_idx.size < n_t:
            f0_clean = np.interp(np.arange(n_t), act_idx, f0_clean[act_idx])

    conf_smooth = ndimage.uniform_filter1d(conf, size=max(3, k_med), mode="nearest")
    return f0_clean, np.clip(conf_smooth, 0.0, 1.0)


def build_harmonic_comb_mask(
    freqs: np.ndarray,
    f0_track_hz: np.ndarray,
    comb_conf: np.ndarray,
    max_harmonics: int = 12,
    bandwidth_hz: float = 18.0,
    conf_thresh: float = 0.30,
) -> np.ndarray:
    """Build smooth (F, T) weight mask around harmonics k * f0(t)."""
    n_f = freqs.size
    n_t = f0_track_hz.size
    mask = np.zeros((n_f, n_t), dtype=np.float64)
    if n_f == 0 or n_t == 0:
        return mask

    bw = max(6.0, float(bandwidth_hz))
    active = comb_conf >= float(conf_thresh)
    if not np.any(active):
        return mask

    freqs_col = freqs[:, np.newaxis]
    act_idx = np.flatnonzero(active)
    f0_act = f0_track_hz[act_idx][np.newaxis, :]
    for h in range(1, max(1, int(max_harmonics)) + 1):
        fh = h * f0_act
        eff_bw = bw + 0.015 * fh
        dist = np.abs(freqs_col - fh)
        contrib = np.exp(-0.5 * (dist / np.maximum(eff_bw * 0.5, 1e-3)) ** 2)
        mask[:, act_idx] = np.maximum(mask[:, act_idx], contrib)

    weight_t = np.clip((comb_conf - conf_thresh) / max(1e-3, 1.0 - conf_thresh), 0.0, 1.0)
    mask *= weight_t[np.newaxis, :]
    return np.clip(mask, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 3. Specialized Outdoor Event Classifiers
# ---------------------------------------------------------------------------


def _parabolic_peak_freq(freqs: np.ndarray, db_col: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= freqs.size - 1:
        return float(freqs[idx])
    y0, y1, y2 = float(db_col[idx - 1]), float(db_col[idx]), float(db_col[idx + 1])
    denom = y0 - 2.0 * y1 + y2
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
    if denom < -1e-9:
        delta = max(-0.5, min(0.5, 0.5 * (y0 - y2) / denom))
        return float(freqs[idx]) + delta * df
    return float(freqs[idx])


def detect_synthetic_beeps(
    mono: np.ndarray,
    sample_rate: int,
    time_offset_s: float = 0.0,
    sensitivity: str = "medium",
) -> list[DistractionEvent]:
    """Detect zero-drift synthetic piezo tones (e.g. Garmin watch beeps) while ignoring FM birds."""
    x = np.asarray(mono, dtype=np.float64).ravel()
    if x.size < 512 or sample_rate < 4000:
        return []

    nperseg = min(2048, x.size)
    noverlap = int(nperseg * 0.75)
    freqs, times, zxx = signal.stft(
        x, fs=sample_rate, window="hann", nperseg=nperseg, noverlap=noverlap, boundary="zeros"
    )
    if times.size < 4:
        return []

    mag = np.abs(zxx)
    db = 20.0 * np.log10(np.maximum(mag, 1e-12))
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 20.0
    dt = float(times[1] - times[0]) if times.size > 1 else 0.02

    k_f = max(5, min(freqs.size | 1, int(round(350.0 / max(df, 1.0))) | 1))
    med_f = ndimage.median_filter(db, size=(k_f, 1), mode="nearest")
    bed_t = np.percentile(db, 20.0, axis=1, keepdims=True)
    prom = db - med_f
    excess = db - bed_t

    prom_req = {"low": 16.0, "medium": 13.0, "high": 10.5}.get(sensitivity, 13.0)
    exc_req = {"low": 13.0, "medium": 10.5, "high": 8.0}.get(sensitivity, 10.5)

    band_m = (freqs >= 1400.0) & (freqs <= min(8000.0, sample_rate * 0.45))
    cand = (prom >= prom_req) & (excess >= exc_req) & (db > -88.0) & band_m[:, np.newaxis]
    if not np.any(cand):
        return []

    # Reject persistent continuous tones (> 35% of window)
    frac_active = np.mean(
        (prom >= prom_req - 3.0) & (excess >= exc_req - 2.0), axis=1, keepdims=True
    )
    cand &= frac_active < 0.35

    labeled, n_comp = ndimage.label(cand, structure=np.ones((3, 3), dtype=int))
    raw_cands: list[dict[str, Any]] = []
    if n_comp > 0:
        slices = ndimage.find_objects(labeled)
        for cid, sl in enumerate(slices, start=1):
            if sl is None:
                continue
            f_sl, t_sl = sl
            sub = labeled[f_sl, t_sl] == cid
            t_len = sub.shape[1]
            dur_s = t_len * dt
            if dur_s < 0.035 or dur_s > 2.0:
                continue
            f_start = f_sl.start
            sub_db = db[f_sl, t_sl]
            p_freqs: list[float] = []
            p_amps: list[float] = []
            for tj in range(t_len):
                rows = np.flatnonzero(sub[:, tj])
                if rows.size == 0:
                    continue
                r = int(rows[np.argmax(sub_db[rows, tj])])
                g_idx = f_start + r
                p_freqs.append(_parabolic_peak_freq(freqs, db[:, t_sl.start + tj], g_idx))
                p_amps.append(float(db[g_idx, t_sl.start + tj]))
            if len(p_freqs) < 2:
                continue
            f_arr = np.asarray(p_freqs, dtype=np.float64)
            f_med = float(np.median(f_arr))
            f_std = float(np.std(f_arr))
            f_ptp = float(np.ptp(f_arr))
            # Zero-drift crystal-oscillator piezo signature vs bird FM glide
            if f_std > 2.5 or f_ptp > max(6.0, 0.25 * df):
                continue

            amp_arr = np.asarray(p_amps, dtype=np.float64)
            mid_lo = max(0, amp_arr.size // 4)
            mid_hi = min(amp_arr.size, max(mid_lo + 1, (3 * amp_arr.size) // 4))
            mid_std = float(np.std(amp_arr[mid_lo:mid_hi])) if mid_hi > mid_lo else 0.0
            if mid_std > 2.2:
                continue

            # Check odd 3rd harmonic presence (square-wave piezo indicator)
            odd_harm = False
            f3 = 3.0 * f_med
            if f3 <= freqs[-1] - df:
                h_mask = (freqs >= f3 - 1.5 * df) & (freqs <= f3 + 1.5 * df)
                if np.any(h_mask):
                    odd_harm = float(np.max(prom[h_mask, t_sl])) >= 5.0

            s_t = max(0.0, float(times[t_sl.start]) - 0.5 * dt)
            e_t = float(times[t_sl.stop - 1]) + 0.5 * dt
            raw_cands.append(
                {
                    "start_s": time_offset_s + s_t,
                    "end_s": time_offset_s + e_t,
                    "freq_hz": f_med,
                    "peak_db": float(np.max(amp_arr)),
                    "f_std": f_std,
                    "odd_harm": odd_harm,
                    "mid_std": mid_std,
                }
            )

    if not raw_cands:
        return []

    # Cross-window frequency fingerprinting: cluster hits within +/- 3.5 Hz
    freqs_list = np.array([c["freq_hz"] for c in raw_cands], dtype=np.float64)
    events: list[DistractionEvent] = []
    for c in raw_cands:
        repeats = int(np.sum(np.abs(freqs_list - c["freq_hz"]) <= 3.5))
        conf = 0.65
        if c["f_std"] < 1.0:
            conf += 0.12
        if c["odd_harm"]:
            conf += 0.12
        if repeats >= 2:
            conf = min(0.98, conf + 0.15)
        f_hz = round(c["freq_hz"], 1)
        sev = "high" if (repeats >= 2 or conf >= 0.85) else "medium"
        events.append(
            DistractionEvent(
                id="pending",
                start_s=round(c["start_s"], 4),
                end_s=round(c["end_s"], 4),
                kind="synthetic_beep",
                severity=sev,
                confidence=finite_or_none(min(0.99, conf), 2),
                freq_band_hz=(round(max(20.0, f_hz - 25.0), 1), round(f_hz + 25.0, 1)),
                peak_db=finite_or_none(c["peak_db"], 2),
                notes=(
                    f"Zero-drift synthetic beep at {f_hz:.1f} Hz "
                    f"(repeats={repeats}, drift_std={c['f_std']:.2f} Hz). "
                    "Remove with spectral_match_suppress(narrowband_mode=True) or drone_notch."
                ),
            )
        )
    return events


def _hf_patch_flatness(mono: np.ndarray, center_idx: int, sample_rate: int) -> float:
    half = min(256, max(64, int(0.006 * sample_rate)))
    lo = max(0, center_idx - half)
    hi = min(mono.size, center_idx + half)
    seg = mono[lo:hi]
    if seg.size < 32:
        return 0.0
    w = seg * np.hanning(seg.size)
    spec = np.abs(np.fft.rfft(w)) ** 2
    freqs = np.fft.rfftfreq(seg.size, d=1.0 / sample_rate)
    hf = spec[freqs >= min(1800.0, sample_rate * 0.3)]
    if hf.size < 4:
        return 0.0
    eps = 1e-24
    gm = float(np.exp(np.mean(np.log(hf + eps))))
    am = float(np.mean(hf)) + eps
    return gm / am


def detect_click_trains_and_crackles(
    audio: np.ndarray,
    sample_rate: int,
    time_offset_s: float = 0.0,
    sensitivity: str = "medium",
) -> list[DistractionEvent]:
    """Detect periodic HF broadband click trains (cable ties) and crackle bursts (bags)."""
    arr = ensure_2d_audio(audio)
    n_samples, _ = arr.shape
    if n_samples < 1024 or sample_rate < 6000:
        return []

    mono = np.mean(arr, axis=1)
    events: list[DistractionEvent] = []

    # 1. Click-train detector (cable-tie ratchet)
    cutoff = min(1800.0, sample_rate * 0.35)
    sos_hp = signal.butter(4, cutoff, btype="highpass", fs=sample_rate, output="sos")
    hp = signal.sosfiltfilt(sos_hp, arr, axis=0)
    env = np.max(np.abs(hp), axis=1)
    med = float(np.median(env))
    sigma = max(float(np.median(np.abs(env - med))) * 1.4826, 1e-7)
    z_req = {"low": 10.5, "medium": 8.0, "high": 6.0}.get(sensitivity, 8.0)
    z = (env - med) / sigma
    hits = np.flatnonzero((z > z_req) & (env > 2e-4))
    if hits.size >= 5:
        merge_gap = max(1, int(0.004 * sample_rate))
        micro_spikes: list[tuple[int, float]] = []
        s = e = int(hits[0])
        pk = float(np.max(np.abs(arr[s])))
        for h in hits[1:]:
            ih = int(h)
            if ih - e <= merge_gap:
                e = ih
                pk = max(pk, float(np.max(np.abs(arr[ih]))))
            else:
                if (e - s) <= int(0.008 * sample_rate):
                    micro_spikes.append(((s + e) // 2, pk))
                s = e = ih
                pk = float(np.max(np.abs(arr[ih])))
        if (e - s) <= int(0.008 * sample_rate):
            micro_spikes.append(((s + e) // 2, pk))

        if len(micro_spikes) >= 5:
            max_train_gap = int(0.11 * sample_rate)
            clusters: list[list[tuple[int, float]]] = [[micro_spikes[0]]]
            for item in micro_spikes[1:]:
                if item[0] - clusters[-1][-1][0] <= max_train_gap:
                    clusters[-1].append(item)
                else:
                    clusters.append([item])

            for cl in clusters:
                if len(cl) < 5:
                    continue
                s0_i, s1_i = cl[0][0], cl[-1][0]
                dur_s = (s1_i - s0_i) / float(sample_rate)
                if dur_s < 0.08 or dur_s > 4.5:
                    continue
                icis_s = np.diff([p[0] for p in cl]) / float(sample_rate)
                med_ici = float(np.median(icis_s))
                cv_ici = float(np.std(icis_s) / max(med_ici, 1e-5))
                if not (0.004 <= med_ici <= 0.10) or cv_ici > 0.72:
                    continue

                # Screen per-click spectral flatness: cable ties are broadband, insects narrowband
                sample_pts = cl[:: max(1, len(cl) // 8)]
                flats = [_hf_patch_flatness(mono, p[0], sample_rate) for p in sample_pts]
                mean_flat = float(np.mean(flats))
                if mean_flat < 0.20:
                    continue

                pk_db = db_from_linear(max(p[1] for p in cl), 2)
                rate_hz = 1.0 / max(med_ici, 1e-4)
                conf = min(
                    0.96, 0.55 + 0.6 * min(0.4, mean_flat - 0.18) + 0.015 * min(len(cl), 20)
                )
                s_ev = round(time_offset_s + max(0.0, s0_i / sample_rate - 0.02), 3)
                e_ev = round(
                    time_offset_s + min(n_samples / sample_rate, s1_i / sample_rate + 0.02), 3
                )
                events.append(
                    DistractionEvent(
                        id="pending",
                        start_s=s_ev,
                        end_s=e_ev,
                        kind="click_train",
                        severity="high" if len(cl) >= 8 else "medium",
                        confidence=finite_or_none(conf, 2),
                        freq_band_hz=(round(cutoff, 1), round(min(18000.0, sample_rate * 0.48), 1)),
                        peak_db=pk_db,
                        notes=(
                            f"Broadband HF click train ({len(cl)} clicks @ ~{rate_hz:.0f} Hz, "
                            f"flatness={mean_flat:.2f}, e.g. cable-tie ratchet). "
                            "Repair locally with declick_impulse(method='ar', spans=[...])."
                        ),
                    )
                )

    # 2. Aperiodic HF crackle-burst detector (plastic bag crumpling)
    freqs, times, z_list, _, _ = _stft_2d(arr, sample_rate, n_fft=1024)
    if times.size >= 8:
        power = np.mean([np.abs(z) ** 2 for z in z_list], axis=0)
        f_hf_lo = min(1800.0, sample_rate * 0.25)
        f_hf_hi = min(16000.0, sample_rate * 0.46)
        hf_m = (freqs >= f_hf_lo) & (freqs <= f_hf_hi)
        if np.sum(hf_m) >= 6:
            p_hf = power[hf_m]
            hf_db = 10.0 * np.log10(np.maximum(np.mean(p_hf, axis=0), 1e-24))
            gm_hf = np.exp(np.mean(np.log(p_hf + 1e-24), axis=0))
            am_hf = np.mean(p_hf, axis=0) + 1e-24
            flat_t = gm_hf / am_hf

            mag_hf = np.sqrt(np.maximum(p_hf, 1e-24))
            flux_t = np.zeros(times.size, dtype=np.float64)
            flux_t[1:] = np.mean(np.maximum(0.0, np.diff(mag_hf, axis=1)), axis=0)
            med_flux = max(float(np.median(flux_t)), 1e-9)

            dt = float(times[1] - times[0]) if times.size > 1 else 0.01
            k_sm = max(3, int(round(0.18 / max(dt, 1e-4))))
            hf_sm = ndimage.uniform_filter1d(hf_db, size=k_sm, mode="nearest")
            flat_sm = ndimage.uniform_filter1d(flat_t, size=k_sm, mode="nearest")
            flux_norm = ndimage.uniform_filter1d(flux_t / med_flux, size=k_sm, mode="nearest")

            med_hf = float(np.median(hf_sm))
            exc_req = {"low": 8.0, "medium": 6.0, "high": 4.2}.get(sensitivity, 6.0)
            mask = (
                (hf_sm >= med_hf + exc_req)
                & (hf_sm > -80.0)
                & (flat_sm >= 0.28)
                & (flux_norm >= 1.55)
            )
            spans = _merge_bool_spans(mask, times, dt, max_gap_s=0.35)
            for i0, i1, s_t, e_t in spans:
                dur_s = e_t - s_t
                if dur_s < 0.25 or dur_s > 18.0:
                    continue
                peak_exc = float(np.max(hf_sm[i0 : i1 + 1]) - med_hf)
                mean_fl = float(np.mean(flat_sm[i0 : i1 + 1]))
                mean_fx = float(np.mean(flux_norm[i0 : i1 + 1]))
                conf = min(0.92, 0.45 + 0.025 * peak_exc + 0.4 * max(0.0, mean_fl - 0.25))
                events.append(
                    DistractionEvent(
                        id="pending",
                        start_s=round(time_offset_s + s_t, 3),
                        end_s=round(time_offset_s + min(n_samples / sample_rate, e_t), 3),
                        kind="crackle_burst",
                        severity="high" if peak_exc >= 11.0 else "medium",
                        confidence=finite_or_none(conf, 2),
                        freq_band_hz=(1800.0, round(min(16000.0, sample_rate * 0.46), 1)),
                        peak_db=finite_or_none(float(np.max(hf_db[i0 : i1 + 1])), 2),
                        notes=(
                            f"Broadband HF crackle (+{peak_exc:.1f} dB, flatness={mean_fl:.2f}, "
                            f"flux={mean_fx:.1f}x; e.g. plastic bag). Repair via ambience_patch."
                        ),
                    )
                )
    return events


def _pitch_harmonicity(
    seg: np.ndarray, sample_rate: int, f0_min_hz: float, f0_max_hz: float
) -> tuple[float, float]:
    """Return (best_f0_hz, normalized_autocorr_harmonicity) for a 1D slice."""
    x = seg - np.mean(seg)
    energy = float(np.dot(x, x))
    if energy <= 1e-12 or x.size < 64:
        return 0.0, 0.0
    ac = np.correlate(x, x, mode="full")[x.size - 1 :] / energy
    lag_min = max(2, int(sample_rate / max(f0_max_hz, 40.0)))
    lag_max = min(x.size // 2, int(sample_rate / max(f0_min_hz, 20.0)))
    if lag_max <= lag_min + 2:
        return 0.0, 0.0
    sub = ac[lag_min:lag_max]
    best_rel = int(np.argmax(sub))
    best_lag = lag_min + best_rel
    return float(sample_rate / best_lag), max(0.0, min(1.0, float(sub[best_rel])))


def detect_voice_and_dog_barks(
    mono: np.ndarray,
    sample_rate: int,
    time_offset_s: float = 0.0,
    sensitivity: str = "medium",
) -> list[DistractionEvent]:
    """Detect human voice syllable modulation and percussive harmonic dog bark bursts."""
    x = np.asarray(mono, dtype=np.float64).ravel()
    if x.size < int(sample_rate * 0.5) or sample_rate < 4000:
        return []

    events: list[DistractionEvent] = []
    # Shared speech/bark body filter (220 - 3400 Hz)
    f_hi = min(3400.0, sample_rate * 0.44)
    if f_hi <= 260.0:
        return []
    sos_sb = signal.butter(4, [220.0, f_hi], btype="bandpass", fs=sample_rate, output="sos")
    sb = signal.sosfiltfilt(sos_sb, x)

    # HF check band (> 3600 Hz) to guard against high birdsong false positives
    hf_sb = np.zeros_like(x)
    if sample_rate >= 10000:
        sos_hf = signal.butter(4, 3600.0, btype="highpass", fs=sample_rate, output="sos")
        hf_sb = signal.sosfiltfilt(sos_hf, x)

    # 1. Dog bark detector (20 ms frames -> sharp percussive harmonic bursts)
    win = max(64, int(0.025 * sample_rate))
    hop = max(32, int(0.0125 * sample_rate))
    n_f = 1 + max(0, (sb.size - win) // hop)
    if n_f >= 12:
        idx_mat = np.arange(win)[np.newaxis, :] + np.arange(n_f)[:, np.newaxis] * hop
        rms = np.sqrt(np.mean(sb[idx_mat] ** 2, axis=1))
        hf_rms = np.sqrt(np.mean(hf_sb[idx_mat] ** 2, axis=1))
        db = 20.0 * np.log10(np.maximum(rms, 1e-12))
        hf_db = 20.0 * np.log10(np.maximum(hf_rms, 1e-12))
        t_f = np.arange(n_f) * (hop / float(sample_rate))
        dt = hop / float(sample_rate)

        med_db = float(np.median(db))
        mad_db = float(np.median(np.abs(db - med_db))) * 1.4826
        bark_exc = {"low": 11.5, "medium": 9.0, "high": 7.0}.get(sensitivity, 9.0)
        thr = max(med_db + bark_exc, med_db + 3.5 * max(mad_db, 1.2))
        raw_bursts = _merge_bool_spans((db > thr) & (db > -78.0), t_f, dt, max_gap_s=0.06)

        verified_barks: list[tuple[float, float, float, float]] = []
        for i0, i1, s_t, e_t in raw_bursts:
            dur = e_t - s_t
            if dur < 0.05 or dur > 0.55:
                continue
            pk_db = float(np.max(db[i0 : i1 + 1]))
            hf_peak = float(np.max(hf_db[i0 : i1 + 1]))
            # Dog barks carry dominant body below 3.4 kHz rather than >3.6 kHz bird peaks
            if hf_peak > pk_db - 1.5 and sample_rate >= 10000:
                continue
            s_i = max(0, int(round(s_t * sample_rate)))
            e_i = min(sb.size, int(round(e_t * sample_rate)))
            f0_b, harm = _pitch_harmonicity(sb[s_i:e_i], sample_rate, 160.0, 880.0)
            if harm < 0.36:
                continue
            verified_barks.append((s_t, e_t, pk_db - med_db, harm))

        if verified_barks:
            grouped: list[list[tuple[float, float, float, float]]] = [[verified_barks[0]]]
            for b in verified_barks[1:]:
                if b[0] - grouped[-1][-1][1] <= 2.2:
                    grouped[-1].append(b)
                else:
                    grouped.append([b])

            for grp in grouped:
                # Require either >=2 barks in cadence OR a single strong (+12 dB) bark
                max_ex = max(g[2] for g in grp)
                if len(grp) < 2 and max_ex < 12.0:
                    continue
                conf = min(
                    0.95,
                    0.48
                    + 0.12 * min(3, len(grp))
                    + 0.2 * float(np.mean([g[3] for g in grp]))
                    + 0.01 * max(0.0, max_ex - 9.0),
                )
                events.append(
                    DistractionEvent(
                        id="pending",
                        start_s=round(time_offset_s + grp[0][0], 3),
                        end_s=round(time_offset_s + grp[-1][1], 3),
                        kind="dog_bark",
                        severity="high" if (len(grp) >= 3 or max_ex >= 14.0) else "medium",
                        confidence=finite_or_none(conf, 2),
                        freq_band_hz=(180.0, round(f_hi, 1)),
                        peak_db=finite_or_none(med_db + max_ex, 2),
                        notes=(
                            f"Dog bark cadence ({len(grp)} burst(s), +{max_ex:.1f} dB over bed). "
                            "Push back with distance_soften(mode='distance', protect_birds=True)."
                        ),
                    )
                )

    # 2. Human voice detector (harmonic pitch 85-320 Hz + 3-8 Hz syllable envelope)
    win_v = max(128, int(0.040 * sample_rate))
    hop_v = max(64, int(0.025 * sample_rate))  # 40 Hz envelope sample rate
    n_v = 1 + max(0, (sb.size - win_v) // hop_v)
    if n_v >= 28:
        idx_v = np.arange(win_v)[np.newaxis, :] + np.arange(n_v)[:, np.newaxis] * hop_v
        rms_v = np.sqrt(np.mean(sb[idx_v] ** 2, axis=1))
        hf_rms_v = np.sqrt(np.mean(hf_sb[idx_v] ** 2, axis=1))
        db_v = 20.0 * np.log10(np.maximum(rms_v, 1e-12))
        t_v = np.arange(n_v) * (hop_v / float(sample_rate))
        dt_v = hop_v / float(sample_rate)

        k_sm = max(3, int(round(0.35 / dt_v)))
        db_sm = ndimage.uniform_filter1d(db_v, size=k_sm, mode="nearest")
        med_v = float(np.median(db_sm))
        exc_v = {"low": 7.5, "medium": 5.5, "high": 4.0}.get(sensitivity, 5.5)
        voice_spans = _merge_bool_spans(
            (db_sm > med_v + exc_v) & (db_sm > -80.0), t_v, dt_v, max_gap_s=0.45
        )
        for i0, i1, s_t, e_t in voice_spans:
            dur = e_t - s_t
            if dur < 0.8 or dur > 45.0:
                continue
            # Check body dominance vs HF >3.6 kHz
            e_sb = float(np.mean(rms_v[i0 : i1 + 1] ** 2)) + 1e-18
            e_hf = float(np.mean(hf_rms_v[i0 : i1 + 1] ** 2))
            if e_hf > e_sb * 0.85 and sample_rate >= 10000:
                continue

            # 3-8 Hz syllable modulation power fraction
            env_seg = rms_v[i0 : i1 + 1]
            env_zm = env_seg - np.mean(env_seg)
            if env_zm.size < 16 or float(np.sum(env_zm * env_zm)) < 1e-14:
                continue
            spec_env = np.abs(np.fft.rfft(env_zm * np.hanning(env_zm.size))) ** 2
            f_env = np.fft.rfftfreq(env_zm.size, d=dt_v)
            tot_mod = float(np.sum(spec_env[(f_env >= 0.8) & (f_env <= 18.0)]))
            syl_mod = float(np.sum(spec_env[(f_env >= 2.5) & (f_env <= 8.5)]))
            syl_ratio = syl_mod / max(tot_mod, 1e-18)
            if syl_ratio < 0.28:
                continue

            # Check vocal fundamental 85-320 Hz across sub-slices
            s_i = max(0, int(round(s_t * sample_rate)))
            e_i = min(sb.size, int(round(e_t * sample_rate)))
            sub_len = max(1024, int(0.08 * sample_rate))
            harms: list[float] = []
            for p in range(s_i, max(s_i + 1, e_i - sub_len), max(1, (e_i - s_i) // 6)):
                _, h_val = _pitch_harmonicity(sb[p : p + sub_len], sample_rate, 85.0, 320.0)
                harms.append(h_val)
            med_harm = float(np.median(harms)) if harms else 0.0
            if med_harm < 0.28:
                continue

            max_ex = float(np.max(db_v[i0 : i1 + 1]) - med_v)
            conf = min(0.92, 0.42 + 0.35 * syl_ratio + 0.35 * med_harm)
            events.append(
                DistractionEvent(
                    id="pending",
                    start_s=round(time_offset_s + s_t, 3),
                    end_s=round(time_offset_s + e_t, 3),
                    kind="human_voice",
                    severity="high" if max_ex >= 11.0 else "medium",
                    confidence=finite_or_none(conf, 2),
                    freq_band_hz=(220.0, round(f_hi, 1)),
                    peak_db=finite_or_none(med_v + max_ex, 2),
                    notes=(
                        f"Human voice ({dur:.1f}s, syllable_mod={syl_ratio:.2f}, "
                        f"pitch_harm={med_harm:.2f}). Muffle via distance_soften."
                    ),
                )
            )
    return events


def detect_variable_rpm_engines(
    mono: np.ndarray,
    sample_rate: int,
    time_offset_s: float = 0.0,
    sensitivity: str = "medium",
) -> list[DistractionEvent]:
    """Detect variable-RPM harmonic combs (leafblowers, shifting engines, trimmers)."""
    x = np.asarray(mono, dtype=np.float64).ravel()
    if x.size < int(sample_rate * 1.5) or sample_rate < 4000:
        return []

    nperseg = min(4096, x.size)
    noverlap = int(nperseg * 0.75)
    freqs, times, zxx = signal.stft(
        x, fs=sample_rate, window="hann", nperseg=nperseg, noverlap=noverlap, boundary="zeros"
    )
    if times.size < 6:
        return []

    mag = np.abs(zxx)
    f0_track, comb_conf = track_harmonic_comb(freqs, times, mag, f0_min_hz=55.0, f0_max_hz=300.0)
    thr_conf = {"low": 0.45, "medium": 0.36, "high": 0.28}.get(sensitivity, 0.36)
    dt = float(times[1] - times[0]) if times.size > 1 else 0.03
    spans = _merge_bool_spans(comb_conf >= thr_conf, times, dt, max_gap_s=0.6)

    band_m = (freqs >= 55.0) & (freqs <= min(3500.0, sample_rate * 0.45))
    band_db = 10.0 * np.log10(np.maximum(np.mean(mag[band_m] ** 2, axis=0), 1e-24))
    med_band = float(np.median(band_db))

    events: list[DistractionEvent] = []
    for i0, i1, s_t, e_t in spans:
        dur = e_t - s_t
        if dur < 1.2 or dur > 45.0:
            continue
        f0_seg = f0_track[i0 : i1 + 1]
        c_seg = comb_conf[i0 : i1 + 1]
        f0_ptp = float(np.ptp(f0_seg))
        ex_db = float(np.max(band_db[i0 : i1 + 1]) - med_band)
        # Require either genuine RPM sweep OR loud active localized excess
        if f0_ptp < 6.0 and ex_db < 6.0:
            continue
        mean_c = float(np.mean(c_seg))
        f0_lo = round(float(np.percentile(f0_seg, 10)), 1)
        f0_hi = round(float(np.percentile(f0_seg, 90)), 1)
        conf = min(0.95, 0.45 + 0.45 * mean_c + (0.12 if f0_ptp >= 12.0 else 0.0))
        events.append(
            DistractionEvent(
                id="pending",
                start_s=round(time_offset_s + s_t, 3),
                end_s=round(time_offset_s + min(x.size / sample_rate, e_t), 3),
                kind="variable_rpm_engine",
                severity="high" if ex_db >= 10.0 else "medium",
                confidence=finite_or_none(conf, 2),
                freq_band_hz=(
                    round(max(40.0, f0_lo * 0.8), 1),
                    round(min(4000.0, f0_hi * 10.0), 1),
                ),
                peak_db=finite_or_none(med_band + ex_db, 2),
                notes=(
                    f"Sweeping harmonic comb f0 ~ {f0_lo:.0f}-{f0_hi:.0f} Hz "
                    f"(+{ex_db:.1f} dB, conf={mean_c:.2f}; e.g. leafblower/engine). "
                    "Use harmonic_track_suppress or ambience_patch."
                ),
            )
        )
    return events


def detect_spatial_anomalies(
    audio: np.ndarray,
    sample_rate: int,
    time_offset_s: float = 0.0,
    sensitivity: str = "medium",
) -> list[DistractionEvent]:
    """Detect low-coherence wind turbulence and channel-biased (L or R) low/mid distractions."""
    arr = ensure_2d_audio(audio)
    n_samples, n_ch = arr.shape
    if n_ch < 2 or n_samples < int(sample_rate * 0.8) or sample_rate < 1200:
        return []

    freqs, times, z_list, _, _ = _stft_2d(arr[:, :2], sample_rate, n_fft=2048)
    if times.size < 6:
        return []
    z_l, z_r = z_list[0], z_list[1]
    p_l = np.abs(z_l) ** 2
    p_r = np.abs(z_r) ** 2
    dt = float(times[1] - times[0]) if times.size > 1 else 0.02
    events: list[DistractionEvent] = []

    # 1. Low-frequency uncorrelated wind turbulence (< 160 Hz)
    low_m = (freqs >= 15.0) & (freqs <= 160.0)
    if np.sum(low_m) >= 2:
        p_low_mean = 0.5 * (np.mean(p_l[low_m], axis=0) + np.mean(p_r[low_m], axis=0))
        low_db = 10.0 * np.log10(np.maximum(p_low_mean, 1e-24))
        k_c = max(3, int(round(0.35 / max(dt, 1e-4))))
        cross_low = np.mean(z_l[low_m] * np.conj(z_r[low_m]), axis=0)
        num_c = ndimage.uniform_filter1d(cross_low.real, k_c)
        num_ci = ndimage.uniform_filter1d(cross_low.imag, k_c)
        den_c = np.sqrt(
            np.maximum(ndimage.uniform_filter1d(np.mean(p_l[low_m], axis=0), k_c), 1e-24)
            * np.maximum(ndimage.uniform_filter1d(np.mean(p_r[low_m], axis=0), k_c), 1e-24)
        )
        coh_low = np.sqrt(num_c * num_c + num_ci * num_ci) / den_c

        low_sm = ndimage.uniform_filter1d(low_db, size=k_c, mode="nearest")
        med_low = float(np.median(low_sm))
        exc_w = {"low": 9.0, "medium": 7.0, "high": 5.0}.get(sensitivity, 7.0)
        wind_mask = (low_sm >= med_low + exc_w) & (low_sm > -82.0) & (coh_low < 0.48)
        for i0, i1, s_t, e_t in _merge_bool_spans(wind_mask, times, dt, max_gap_s=0.45):
            dur = e_t - s_t
            if dur < 0.35 or dur > 45.0:
                continue
            peak_ex = float(np.max(low_sm[i0 : i1 + 1]) - med_low)
            min_coh = float(np.min(coh_low[i0 : i1 + 1]))
            events.append(
                DistractionEvent(
                    id="pending",
                    start_s=round(time_offset_s + s_t, 3),
                    end_s=round(time_offset_s + min(n_samples / sample_rate, e_t), 3),
                    kind="wind_noise",
                    severity="high" if peak_ex >= 12.0 else "medium",
                    confidence=finite_or_none(min(0.95, 0.55 + 0.4 * (0.5 - min_coh)), 2),
                    freq_band_hz=(15.0, 160.0),
                    peak_db=finite_or_none(med_low + peak_ex, 2),
                    notes=(
                        f"Uncorrelated <160 Hz wind turbulence (+{peak_ex:.1f} dB, "
                        f"coherence={min_coh:.2f}). Suppress with spatial_mix(low_mono_hz=140)."
                    ),
                )
            )

    # 2. Channel-biased L/R noise (< 1500 Hz weighted to avoid stereo birds)
    bias_m = (freqs >= 40.0) & (freqs <= min(1500.0, sample_rate * 0.45))
    hi_m = (freqs > 1800.0) & (freqs <= min(8000.0, sample_rate * 0.45))
    if np.sum(bias_m) >= 3:
        k_s = max(3, int(round(0.45 / max(dt, 1e-4))))
        e_l = ndimage.uniform_filter1d(np.mean(p_l[bias_m], axis=0), k_s)
        e_r = ndimage.uniform_filter1d(np.mean(p_r[bias_m], axis=0), k_s)
        db_l = 10.0 * np.log10(np.maximum(e_l, 1e-24))
        db_r = 10.0 * np.log10(np.maximum(e_r, 1e-24))
        ild = db_l - db_r
        med_ild = float(np.median(ild))
        med_l = float(np.median(db_l))
        med_r = float(np.median(db_r))
        max_ex_t = np.maximum(db_l - med_l, db_r - med_r)

        if np.any(hi_m):
            e_hi = ndimage.uniform_filter1d(
                0.5 * (np.mean(p_l[hi_m], axis=0) + np.mean(p_r[hi_m], axis=0)), k_s
            )
            low_dom = (0.5 * (e_l + e_r)) / np.maximum(e_hi, 1e-24) >= 0.40
        else:
            low_dom = np.ones(times.size, dtype=bool)

        ild_thr = {"low": 6.0, "medium": 4.5, "high": 3.5}.get(sensitivity, 4.5)
        ex_thr = {"low": 6.5, "medium": 4.5, "high": 3.0}.get(sensitivity, 4.5)
        bias_mask = (np.abs(ild - med_ild) >= ild_thr) & (max_ex_t >= ex_thr) & low_dom
        for i0, i1, s_t, e_t in _merge_bool_spans(bias_mask, times, dt, max_gap_s=0.6):
            dur = e_t - s_t
            if dur < 1.0 or dur > 45.0:
                continue
            mean_shift = float(np.mean(ild[i0 : i1 + 1] - med_ild))
            side = "L-biased" if mean_shift > 0.0 else "R-biased"
            peak_ex = float(np.max(max_ex_t[i0 : i1 + 1]))
            conf = min(0.94, 0.50 + 0.04 * abs(mean_shift) + 0.02 * peak_ex)
            events.append(
                DistractionEvent(
                    id="pending",
                    start_s=round(time_offset_s + s_t, 3),
                    end_s=round(time_offset_s + min(n_samples / sample_rate, e_t), 3),
                    kind="channel_biased_noise",
                    severity="high" if (abs(mean_shift) >= 7.5 and peak_ex >= 8.0) else "medium",
                    confidence=finite_or_none(conf, 2),
                    freq_band_hz=(40.0, round(min(1500.0, sample_rate * 0.45), 1)),
                    peak_db=finite_or_none(float(np.max(np.maximum(db_l, db_r)[i0 : i1 + 1])), 2),
                    notes=(
                        f"{side} noise anomaly (ILD shift {mean_shift:+.1f} dB, +{peak_ex:.1f} dB "
                        "excess). Soften via spectral_match_suppress spatial_mode='channel_excess'."
                    ),
                )
            )
    return events


# ---------------------------------------------------------------------------
# 4. AR / LPC Transient Gap Repair Engine
# ---------------------------------------------------------------------------


def _solve_ar_coeffs(ctx: np.ndarray, order: int) -> np.ndarray | None:
    """Estimate Yule-Walker AR coefficients from a short clean context slice."""
    n = ctx.size
    if n < order + 4:
        return None
    x = ctx - np.mean(ctx)
    e = float(np.dot(x, x))
    if e < 1e-14:
        return None
    ac = np.correlate(x, x, mode="full")[n - 1 : n + order] / e
    r = ac[:order]
    r[0] += 1e-5  # Tikhonov stabilization against singular bursts
    rhs = -ac[1 : order + 1]
    try:
        a = linalg.solve_toeplitz((r, r), rhs, check_finite=False)
        if not np.all(np.isfinite(a)):
            return None
        return np.asarray(a, dtype=np.float64)
    except Exception:
        return None


def ar_interpolate_gap(x: np.ndarray, s_i: int, e_i: int, order: int = 16) -> np.ndarray:
    """Bidirectional Yule-Walker AR prediction across [s_i, e_i] inclusive."""
    n = x.size
    gap_len = e_i - s_i + 1
    if gap_len <= 0:
        return np.zeros(0, dtype=np.float64)
    p = min(order, max(4, gap_len * 2), s_i, max(0, n - 1 - e_i))
    ctx_len = min(max(p * 6, 96), s_i, max(0, n - 1 - e_i))
    if p < 4 or ctx_len < p + 4:
        left_x = np.arange(max(0, s_i - 4), s_i, dtype=np.float64)
        right_x = np.arange(e_i + 1, min(n, e_i + 5), dtype=np.float64)
        if left_x.size == 0 or right_x.size == 0:
            return np.zeros(gap_len, dtype=np.float64)
        x_ctx = np.concatenate([left_x, right_x])
        y_ctx = x[x_ctx.astype(np.int64)]
        return np.interp(np.arange(s_i, e_i + 1, dtype=np.float64), x_ctx, y_ctx)

    l_ctx = x[s_i - ctx_len : s_i]
    r_ctx = x[e_i + 1 : e_i + 1 + ctx_len]
    a_fwd = _solve_ar_coeffs(l_ctx, p)
    a_bwd = _solve_ar_coeffs(r_ctx[::-1], p)
    if a_fwd is None or a_bwd is None:
        left_x = np.arange(max(0, s_i - 4), s_i, dtype=np.float64)
        right_x = np.arange(e_i + 1, min(n, e_i + 5), dtype=np.float64)
        x_ctx = np.concatenate([left_x, right_x])
        x_q = np.arange(s_i, e_i + 1, dtype=np.float64)
        return np.interp(x_q, x_ctx, x[x_ctx.astype(np.int64)])

    fwd_buf = np.zeros(p + gap_len, dtype=np.float64)
    fwd_buf[:p] = l_ctx[-p:] - float(np.mean(l_ctx))
    mu_l = float(np.mean(l_ctx))
    for k in range(gap_len):
        prev = fwd_buf[p + k - 1 :: -1][:p]
        val = float(-np.dot(a_fwd, prev))
        fwd_buf[p + k] = max(-1.5, min(1.5, val))
    fwd_pred = fwd_buf[p:] + mu_l

    bwd_buf = np.zeros(p + gap_len, dtype=np.float64)
    mu_r = float(np.mean(r_ctx))
    bwd_buf[:p] = r_ctx[:p][::-1] - mu_r
    for k in range(gap_len):
        prev = bwd_buf[p + k - 1 :: -1][:p]
        val = float(-np.dot(a_bwd, prev))
        bwd_buf[p + k] = max(-1.5, min(1.5, val))
    bwd_pred = bwd_buf[p:][::-1] + mu_r

    w = np.linspace(0.0, 1.0, gap_len, endpoint=True)
    return (1.0 - w) * fwd_pred + w * bwd_pred


# ---------------------------------------------------------------------------
# 5. Unified Excess-Capped, Bird-Protected Mask Engine
# ---------------------------------------------------------------------------


def apply_masked_stft_transform(
    audio: np.ndarray,
    sample_rate: int,
    spans: list[dict[str, Any]],
    ref_overrides: dict[int, np.ndarray] | None = None,
    mode: str = "spectral_match",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Shared excess-capped, context-gapped, bird-protected STFT transform engine.

    Supports:
      - mode="spectral_match" (linked or channel_excess spatial suppression, optional narrowband)
      - mode="harmonic_comb" (sweeping f0(t) comb mask + optional ILD bias focus)
      - mode="distance_soften" (distance HF air rolloff + attack smear, or voice formant blur)
    """
    arr = ensure_2d_audio(audio)
    n_samples, n_ch = arr.shape
    if n_samples < 64 or not spans:
        return arr.copy(), []

    dur_s = n_samples / float(sample_rate)
    n_fft_val = max(int(sp.get("n_fft", 2048)) for sp in spans)
    freqs, times, z_list, nperseg, noverlap = _stft_2d(arr, sample_rate, n_fft_val)
    dt_frame = float(times[1] - times[0]) if times.size > 1 else 0.02

    mag_chs = [np.sqrt(np.maximum(np.abs(z) ** 2, 1e-24)) for z in z_list]
    mag_mean = np.sqrt(np.maximum(np.mean([np.abs(z) ** 2 for z in z_list], axis=0), 1e-24))
    mag_s_mean = ndimage.uniform_filter(mag_mean, size=(3, 5), mode="nearest")
    mag_s_chs = [ndimage.uniform_filter(mc, size=(3, 5), mode="nearest") for mc in mag_chs]

    protect_mask_all: np.ndarray | None = None
    if any(bool(sp.get("protect_birds", False)) for sp in spans):
        protect_mask_all = compute_bird_protect_mask(freqs, times, mag_mean)

    z_accum = [z.copy() for z in z_list]
    total_gain_db = [np.zeros_like(mag_mean) for _ in range(n_ch)]
    active_ranges: list[tuple[int, int]] = []
    logs: list[dict[str, Any]] = []
    all_intervals = [(float(o["start_s"]), float(o["end_s"])) for o in spans]

    for sp_idx, sp in enumerate(spans):
        s_t = max(0.0, float(sp["start_s"]))
        e_t = min(dur_s, float(sp["end_s"]))
        if e_t <= s_t:
            continue

        fade_s = float(sp.get("fade_ms", 1000.0)) * 0.001
        ctx_s = float(sp.get("context_s", 6.0))
        ctx_gap_s = max(0.0, float(sp.get("context_gap_s", 1.5)))
        f_lo = float(sp.get("low_hz", 25.0))
        f_hi = min(sample_rate * 0.5, float(sp.get("high_hz", 4000.0)))
        if f_hi <= f_lo:
            continue

        # Build clean reference magnitude separating context by context_gap_s
        ext_ref = ref_overrides.get(sp_idx) if ref_overrides else None
        donor_src = np.zeros((0, n_ch), dtype=np.float64)
        if ext_ref is not None and ext_ref.shape[0] >= 64:
            donor_src = ensure_2d_audio(ext_ref)
        elif sp.get("ref_start_s") is not None and sp.get("ref_end_s") is not None:
            rs0 = max(0, int(round(float(sp["ref_start_s"]) * sample_rate)))
            rs1 = min(n_samples, int(round(float(sp["ref_end_s"]) * sample_rate)))
            if rs1 - rs0 >= 64:
                donor_src = arr[rs0:rs1]

        if donor_src.shape[0] >= 64:
            if donor_src.shape[0] < nperseg:
                xf_pad = min(256, max(8, donor_src.shape[0] // 4))
                donor_src = _tiled_array(donor_src, nperseg, xf_pad)
            _, _, zr_list, _, _ = _stft_2d(donor_src, sample_rate, nperseg)
            mag_ref = np.sqrt(
                np.maximum(np.mean([np.abs(z) ** 2 for z in zr_list], axis=0), 1e-24)
            )
            bed_med = np.median(mag_ref, axis=1, keepdims=True)
            bed_p80 = np.percentile(mag_ref, 80.0, axis=1, keepdims=True)
            bed_chs_med = [
                np.median(np.sqrt(np.maximum(np.abs(zr) ** 2, 1e-24)), axis=1, keepdims=True)
                for zr in zr_list
            ]
        else:
            in_any_span = np.zeros(times.size, dtype=bool)
            for os_t, oe_t in all_intervals:
                in_any_span |= (times >= max(0.0, os_t - ctx_gap_s * 0.5)) & (
                    times <= min(dur_s, oe_t + ctx_gap_s * 0.5)
                )
            gap_eff = ctx_gap_s
            m_l = np.zeros(times.size, dtype=bool)
            m_r = np.zeros(times.size, dtype=bool)
            l_s0 = l_s1 = r_s0 = r_s1 = 0.0
            for _scale in (1.0, 0.35, 0.0):
                g_cur = gap_eff * _scale
                l_s0 = max(0.0, s_t - fade_s - g_cur - ctx_s)
                l_s1 = max(l_s0, s_t - fade_s * 0.25 - g_cur)
                r_s0 = min(dur_s, e_t + fade_s * 0.25 + g_cur)
                r_s1 = min(dur_s, e_t + fade_s + g_cur + ctx_s)
                m_l = (times >= l_s0) & (times < l_s1) & (~in_any_span)
                m_r = (times > r_s0) & (times <= r_s1) & (~in_any_span)
                if np.sum(m_l) >= 3 or np.sum(m_r) >= 3:
                    break
            if np.sum(m_l) < 3 and np.sum(m_r) < 3:
                l_s0 = max(0.0, s_t - fade_s - ctx_s)
                l_s1 = max(l_s0, s_t - fade_s * 0.25)
                r_s0 = min(dur_s, e_t + fade_s * 0.25)
                r_s1 = min(dur_s, e_t + fade_s + ctx_s)
                m_l = (times >= l_s0) & (times < l_s1)
                m_r = (times > r_s0) & (times <= r_s1)

            if np.sum(m_l) >= 3 and np.sum(m_r) >= 3:
                med_l = np.median(mag_mean[:, m_l], axis=1, keepdims=True)
                med_r = np.median(mag_mean[:, m_r], axis=1, keepdims=True)
                p80_l = np.percentile(mag_mean[:, m_l], 80.0, axis=1, keepdims=True)
                p80_r = np.percentile(mag_mean[:, m_r], 80.0, axis=1, keepdims=True)
                bed_med = np.minimum(med_l, med_r)
                bed_p80 = np.minimum(p80_l, p80_r)
                pick_l = float(np.sum(med_l)) <= float(np.sum(med_r))
                ds0, ds1 = (l_s0, l_s1) if pick_l else (r_s0, r_s1)
                m_ch_pick = m_l if pick_l else m_r
                bed_chs_med = [np.median(mc[:, m_ch_pick], axis=1, keepdims=True) for mc in mag_chs]
            elif np.sum(m_l) >= 3 or np.sum(m_r) >= 3:
                m_use = m_l if np.sum(m_l) >= np.sum(m_r) else m_r
                bed_med = np.median(mag_mean[:, m_use], axis=1, keepdims=True)
                bed_p80 = np.percentile(mag_mean[:, m_use], 80.0, axis=1, keepdims=True)
                ds0, ds1 = (l_s0, l_s1) if np.sum(m_l) >= np.sum(m_r) else (r_s0, r_s1)
                bed_chs_med = [np.median(mc[:, m_use], axis=1, keepdims=True) for mc in mag_chs]
            else:
                out_m = ~in_any_span
                if np.sum(out_m) < 3:
                    out_m = (times < s_t) | (times > e_t)
                m_use = out_m if np.sum(out_m) >= 3 else np.ones(times.size, dtype=bool)
                bed_med = np.median(mag_mean[:, m_use], axis=1, keepdims=True)
                bed_p80 = np.percentile(mag_mean[:, m_use], 80.0, axis=1, keepdims=True)
                ds0, ds1 = (0.0, max(0.0, s_t)) if s_t >= (dur_s - e_t) else (e_t, dur_s)
                bed_chs_med = [np.median(mc[:, m_use], axis=1, keepdims=True) for mc in mag_chs]

            di0 = max(0, min(n_samples, int(round(ds0 * sample_rate))))
            di1 = max(di0, min(n_samples, int(round(ds1 * sample_rate))))
            if di1 > di0:
                donor_src = arr[di0:di1]

        w_t = cosine_edge_mask(times, s_t, e_t, fade_s)
        narrow = bool(sp.get("narrowband_mode", False))
        f_fade = max(4.0, 0.04 * (f_hi - f_lo)) if narrow else max(15.0, 0.08 * (f_hi - f_lo))
        w_f = cosine_edge_mask(freqs, f_lo, f_hi, f_fade)
        mask_2d = np.outer(w_f, w_t)

        # Build mode-specific TF weight mod_mask in [0, 1] and mode-specific max cut ceiling
        mode_logs: dict[str, Any] = {}
        max_atten_db = float(sp.get("max_attenuation_db", sp.get("attenuation_db", -28.0)))
        strength = float(sp.get("strength", 1.0))

        if mode == "harmonic_comb":
            f0_track, comb_conf = track_harmonic_comb(
                freqs,
                times,
                mag_mean,
                f0_min_hz=float(sp.get("f0_min_hz", 55.0)),
                f0_max_hz=float(sp.get("f0_max_hz", 300.0)),
                max_harmonics=int(sp.get("max_harmonics", 12)),
            )
            comb_mask = build_harmonic_comb_mask(
                freqs,
                f0_track,
                comb_conf,
                max_harmonics=int(sp.get("max_harmonics", 12)),
                bandwidth_hz=float(sp.get("bandwidth_hz", 18.0)),
            )
            broad_assist = max(0.0, min(1.0, float(sp.get("broadband_assist", 0.35))))
            active_comb = comb_conf >= 0.30
            if np.any(active_comb):
                mode_logs["f0_min_tracked_hz"] = round(float(np.min(f0_track[active_comb])), 1)
                mode_logs["f0_max_tracked_hz"] = round(float(np.max(f0_track[active_comb])), 1)
            else:
                mode_logs["f0_min_tracked_hz"] = None
                mode_logs["f0_max_tracked_hz"] = None
            w_broad = broad_assist * (comb_conf[np.newaxis, :] >= 0.25)
            mask_2d = mask_2d * np.maximum(comb_mask, w_broad)
        elif mode == "distance_soften":
            dist_mode = str(sp.get("mode", "distance")).strip().lower()
            if dist_mode != "voice_muffle":
                air = max(0.0, min(1.0, float(sp.get("air_absorption", 0.65))))
                hf_weight = 1.0 + air * np.clip((freqs - 1200.0) / 3500.0, 0.0, 1.5)
                mask_2d = np.clip(mask_2d * hf_weight[:, np.newaxis], 0.0, 1.35)
                smear_ms = max(0.0, float(sp.get("attack_smear_ms", 12.0)))
                if smear_ms > 0.0:
                    k_smear = max(1, int(round((smear_ms * 0.001) / max(dt_frame, 1e-4))))
                    if k_smear > 1:
                        mask_2d = ndimage.uniform_filter1d(
                            mask_2d, size=k_smear, axis=1, mode="nearest"
                        )
            mode_logs["mode"] = dist_mode

        spatial_mode = str(sp.get("spatial_mode", "linked")).strip().lower()
        if mode == "harmonic_comb" and bool(sp.get("spatial_bias", True)) and n_ch >= 2:
            spatial_mode = "channel_excess"

        # Compute excess-capped per-channel cut (never cuts deeper than actual bin excess!)
        excess_db_chs: list[np.ndarray] = []
        if spatial_mode == "channel_excess" and n_ch >= 2:
            raw_db_chs: list[np.ndarray] = []
            other_ch = [1, 0]
            # Learn baseline ILD profile from clean reference frames
            base_ild_01 = bed_chs_med[0] / (bed_chs_med[1] + 1e-12)
            base_ild_10 = bed_chs_med[1] / (bed_chs_med[0] + 1e-12)
            cap_ratios = [base_ild_01, base_ild_10]
            for c_idx in range(2):
                mc_s = mag_s_chs[c_idx]
                mc_other = mag_s_chs[other_ch[c_idx]]
                # Local clean ceiling: max(own clean bed, quiet_channel * baseline_ILD)
                own_bed = bed_chs_med[c_idx]
                cross_cap = np.maximum(own_bed, mc_other * cap_ratios[c_idx]) * 1.15
                eff_target = np.minimum(own_bed, cross_cap)
                ex_lin = np.maximum(1.0, mc_s / (own_bed + 1e-12))
                ex_db_c = 20.0 * np.log10(ex_lin)
                excess_db_chs.append(ex_db_c)
                over_c = mc_s / (bed_p80 + 1e-12)
                g_c = np.where(over_c > 1.0, (eff_target / (mc_s + 1e-12)) ** strength, 1.0)
                g_db_c = 20.0 * np.log10(np.clip(g_c, 10.0 ** (max_atten_db / 20.0), 1.0))
                raw_db_chs.append(np.maximum(g_db_c, -ex_db_c))

            # Preserve extra surround/rear channels beyond stereo pair without raising IndexError
            for c_idx in range(2, n_ch):
                ex_extra = 20.0 * np.log10(
                    np.maximum(1.0, mag_s_chs[c_idx] / (bed_chs_med[c_idx] + 1e-12))
                )
                excess_db_chs.append(ex_extra)
                raw_db_chs.append(np.zeros_like(mag_mean))

            # Partial channel linking (each channel inherits at least 50% of the other's cut)
            # to avoid stereo image wobble on broadband L/R passes.
            linked_0 = np.minimum(raw_db_chs[0], 0.5 * raw_db_chs[1])
            linked_1 = np.minimum(raw_db_chs[1], 0.5 * raw_db_chs[0])
            ch_raw_db = [linked_0, linked_1, *raw_db_chs[2:]]
            over_ratio_log = np.maximum(
                mag_s_chs[0] / (bed_chs_med[0] + 1e-12),
                mag_s_chs[1] / (bed_chs_med[1] + 1e-12),
            )
            excess_db_global = np.maximum(excess_db_chs[0], excess_db_chs[1])
        else:
            mag_eff = np.maximum(mag_mean, mag_s_mean) if narrow else mag_s_mean
            excess_lin = np.maximum(1.0, mag_eff / (bed_med + 1e-12))
            excess_db_global = 20.0 * np.log10(excess_lin)
            over_ratio = mag_eff / (bed_p80 + 1e-12)
            over_ratio_log = excess_lin
            raw_gain = np.where(over_ratio > 1.0, (bed_med / (mag_eff + 1e-12)) ** strength, 1.0)
            min_gain = 10.0 ** (max_atten_db / 20.0)
            raw_gain = np.clip(raw_gain, min_gain, 1.0)
            raw_db = 20.0 * np.log10(np.maximum(raw_gain, 1e-6))
            # Hard invariant: never cut deeper than measured bin excess over bed_med
            raw_db = np.maximum(raw_db, -excess_db_global)
            ch_raw_db = [raw_db for _ in range(n_ch)]

        # Smooth + mask + excess cap + bird protection veto
        gauss_sig = (0.35, 1.5) if narrow else (1.0, 2.2)
        applied_db_chs: list[np.ndarray] = []
        for c_idx in range(n_ch):
            g_sm = ndimage.gaussian_filter(ch_raw_db[c_idx], sigma=gauss_sig, mode="nearest")
            use_ch_ex = spatial_mode == "channel_excess" and c_idx < len(excess_db_chs)
            ex_c = excess_db_chs[c_idx] if use_ch_ex else excess_db_global
            if narrow:
                # For pinpoint Garmin beeps, don't let neighbor Gaussian blur erode bin peak
                g_sm = np.minimum(g_sm, ch_raw_db[c_idx])
            app_c = g_sm * np.minimum(1.0, mask_2d)
            app_c = np.maximum(max_atten_db, np.maximum(app_c, -ex_c))
            if protect_mask_all is not None and bool(sp.get("protect_birds", False)):
                app_c = app_c * (1.0 - 0.88 * protect_mask_all)
            step_c = np.minimum(
                0.0, np.maximum(max_atten_db - total_gain_db[c_idx], app_c)
            )
            total_gain_db[c_idx] = total_gain_db[c_idx] + step_c
            applied_db_chs.append(step_c)
            gain_2d = 10.0 ** (step_c / 20.0)
            z_accum[c_idx] = z_accum[c_idx] * gain_2d

        # Distance/Voice extras on suppressed excess components
        if mode == "distance_soften":
            dist_mode = str(sp.get("mode", "distance")).strip().lower()
            mean_app = np.mean(applied_db_chs, axis=0)
            soft_weight = np.clip(-mean_app / 12.0, 0.0, 1.0) * np.minimum(1.0, mask_2d)
            if protect_mask_all is not None and bool(sp.get("protect_birds", False)):
                soft_weight = soft_weight * (1.0 - 0.88 * protect_mask_all)
            if dist_mode == "voice_muffle":
                fblur = max(0.0, min(1.0, float(sp.get("formant_blur", 0.5))))
                if fblur > 0.0:
                    for c_idx in range(n_ch):
                        mag_c = np.abs(z_accum[c_idx])
                        phase_c = z_accum[c_idx] / np.maximum(mag_c, 1e-12)
                        blur_m = ndimage.gaussian_filter1d(
                            mag_c, sigma=2.5, axis=0, mode="nearest"
                        )
                        w_b = fblur * soft_weight
                        z_accum[c_idx] = ((1.0 - w_b) * mag_c + w_b * blur_m) * phase_c
            narrow_st = max(0.0, min(1.0, float(sp.get("narrow_stereo", 0.35))))
            if n_ch >= 2 and narrow_st > 0.0:
                mid_z = 0.5 * (z_accum[0] + z_accum[1])
                side_z = 0.5 * (z_accum[0] - z_accum[1]) * (1.0 - narrow_st * soft_weight)
                z_accum[0] = mid_z + side_z
                z_accum[1] = mid_z - side_z

        # Optional clean-context spectral texture fill for deep cuts
        tex_fill = float(sp.get("texture_fill", 0.65 if mode != "distance_soften" else 0.0))
        if tex_fill > 0.0 and donor_src.shape[0] > 64:
            xf = min(int(0.5 * sample_rate), max(16, donor_src.shape[0] // 4))
            tiled = _tiled_array(donor_src, n_samples, xf)
            _, _, z_clean, _, _ = _stft_2d(tiled, sample_rate, nperseg)
            for c_idx in range(n_ch):
                cut_depth = np.maximum(0.0, -applied_db_chs[c_idx])
                depth_w = np.clip((cut_depth - 4.0) / 12.0, 0.0, 1.0)
                blend = tex_fill * depth_w * np.minimum(1.0, mask_2d)
                if protect_mask_all is not None and bool(sp.get("protect_birds", False)):
                    blend = blend * (1.0 - 0.88 * protect_mask_all)
                w_orig = np.sqrt(np.maximum(0.0, 1.0 - blend))
                w_fill = np.sqrt(np.maximum(0.0, blend))
                z_accum[c_idx] = z_accum[c_idx] * w_orig + z_clean[c_idx] * w_fill

        active_tf = np.outer(w_f > 0.2, w_t > 0.1)
        if np.any(active_tf):
            over_pk = float(np.max(over_ratio_log[active_tf]))
            peak_ex = finite_or_none(20.0 * math.log10(max(over_pk, 1.0)), digits=2) or 0.0
            cuts = np.max([np.maximum(0.0, -a[active_tf]) for a in applied_db_chs], axis=0)
            max_cut = finite_or_none(float(np.max(cuts)), digits=2) or 0.0
            mean_cut = finite_or_none(float(np.mean(cuts)), digits=2) or 0.0
            bird_pct = (
                finite_or_none(100.0 * float(np.mean(protect_mask_all[active_tf] >= 0.3)), digits=1)
                or 0.0
                if protect_mask_all is not None and bool(sp.get("protect_birds", False))
                else 0.0
            )
        else:
            peak_ex = max_cut = mean_cut = bird_pct = 0.0

        log_entry: dict[str, Any] = {
            "start_s": finite_or_none(s_t, digits=3) or 0.0,
            "end_s": finite_or_none(e_t, digits=3) or 0.0,
            "low_hz": finite_or_none(f_lo, digits=1) or 0.0,
            "high_hz": finite_or_none(f_hi, digits=1) or 0.0,
            "peak_excess_db": peak_ex,
            "max_cut_db": max_cut,
            "mean_cut_db": mean_cut,
            "texture_fill": finite_or_none(tex_fill, digits=2) or 0.0,
            "bird_protected_pct": bird_pct,
        }
        log_entry.update(mode_logs)
        logs.append(log_entry)

        pad_s = fade_s + (nperseg / float(sample_rate))
        active_ranges.append(
            (
                max(0, int((s_t - pad_s) * sample_rate)),
                min(n_samples, int((e_t + pad_s) * sample_rate)),
            )
        )

    if not active_ranges:
        return arr.copy(), []

    processed = _istft_2d(z_accum, sample_rate, nperseg, noverlap, n_samples)
    out = arr.copy()
    for i0, i1 in active_ranges:
        out[i0:i1] = processed[i0:i1]
    return out, logs


# ---------------------------------------------------------------------------
# 6. Psychoacoustic Pleasantness, Band Metrics & Guardrail Engine
# ---------------------------------------------------------------------------

# Zwicker critical-band upper edges (Hz) and Bark sharpness weighting g(z)
_BARK_BANDS_HZ: tuple[tuple[float, float, float], ...] = (
    (20.0, 100.0, 1.0),
    (100.0, 200.0, 1.0),
    (200.0, 300.0, 1.0),
    (300.0, 400.0, 1.0),
    (400.0, 510.0, 1.0),
    (510.0, 630.0, 1.0),
    (630.0, 770.0, 1.0),
    (770.0, 920.0, 1.0),
    (920.0, 1080.0, 1.0),
    (1080.0, 1270.0, 1.0),
    (1270.0, 1480.0, 1.0),
    (1480.0, 1720.0, 1.0),
    (1720.0, 2000.0, 1.0),
    (2000.0, 2320.0, 1.0),
    (2320.0, 2700.0, 1.05),
    (2700.0, 3150.0, 1.25),
    (3150.0, 3700.0, 1.55),
    (3700.0, 4400.0, 1.95),
    (4400.0, 5300.0, 2.45),
    (5300.0, 6400.0, 3.05),
    (6400.0, 7700.0, 3.70),
    (7700.0, 9500.0, 4.35),
    (9500.0, 12000.0, 4.95),
    (12000.0, 15500.0, 5.50),
)

_STANDARD_BANDS: tuple[tuple[str, float, float], ...] = (
    ("sub_20_80", 20.0, 80.0),
    ("low_80_250", 80.0, 250.0),
    ("mid_250_1000", 250.0, 1000.0),
    ("presence_1000_4000", 1000.0, 4000.0),
    ("air_4000_16000", 4000.0, 16000.0),
)


def compute_band_metrics_from_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    custom_range: tuple[float, float] | None = None,
    seg_psds: list[np.ndarray] | None = None,
    seg_freqs: np.ndarray | None = None,
) -> dict[str, dict[str, float | None]]:
    """Compute per-band integrated RMS dBFS, P10 bed dBFS, and peak excess dB from Welch PSD."""
    if freqs.size < 4 or psd.size != freqs.size or float(np.sum(psd)) <= 0.0:
        return {}

    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
    nyq = float(freqs[-1])
    bands: list[tuple[str, float, float]] = list(_STANDARD_BANDS)
    if custom_range is not None:
        bands.append(("custom_freq_range", float(custom_range[0]), float(custom_range[1])))

    seg_mat = np.asarray(seg_psds, dtype=np.float64) if seg_psds else None
    f_seg = seg_freqs if seg_freqs is not None else freqs

    out: dict[str, dict[str, float | None]] = {}
    for name, f_lo, f_hi in bands:
        hi_c = min(f_hi, nyq)
        m = (freqs >= f_lo) & (freqs <= hi_c)
        if not np.any(m) or hi_c <= f_lo:
            out[name] = {"rms_dbfs": None, "bed_p10_dbfs": None, "peak_excess_db": None}
            continue
        p_band = max(float(np.sum(psd[m]) * df), 1e-24)
        rms_db = finite_or_none(10.0 * math.log10(p_band), digits=2)

        bed_p10_db: float | None = None
        peak_exc_db: float | None = None
        if seg_mat is not None and seg_mat.shape[0] >= 2 and f_seg.size == seg_mat.shape[1]:
            df_s = float(f_seg[1] - f_seg[0]) if f_seg.size > 1 else df
            m_s = (f_seg >= f_lo) & (f_seg <= hi_c)
            if np.any(m_s):
                s_e = np.maximum(np.sum(seg_mat[:, m_s], axis=1) * df_s, 1e-24)
                s_db = 10.0 * np.log10(s_e)
                p10 = float(np.percentile(s_db, 10.0))
                pmax = float(np.max(s_db))
                bed_p10_db = finite_or_none(p10, digits=2)
                peak_exc_db = finite_or_none(max(0.0, pmax - p10), digits=2)
        if bed_p10_db is None:
            bed_p10_db = rms_db
            peak_exc_db = 0.0

        out[name] = {
            "rms_dbfs": rms_db,
            "bed_p10_dbfs": bed_p10_db,
            "peak_excess_db": peak_exc_db,
        }
    return out


def compute_pleasantness_metrics(
    freqs: np.ndarray,
    psd: np.ndarray,
    roughness_values: list[float] | np.ndarray | None = None,
    rumble_ratio_db: float | None = None,
    num_drones: int = 0,
) -> dict[str, float | None]:
    """Compute Zwicker-inspired sharpness_acum, roughness_index, NDSI, slope, and score."""
    if freqs.size < 8 or psd.size != freqs.size or float(np.sum(psd)) <= 1e-20:
        return {
            "sharpness_acum": None,
            "roughness_index": None,
            "ndsi": None,
            "spectral_slope_db_per_oct": None,
            "pleasantness_score": None,
        }

    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
    # 1. Sharpness (acum): specific loudness S^0.23 proxy across Bark bands weighted by g(z)
    num_s = 0.0
    den_s = 0.0
    for idx_z, (f0, f1, gz) in enumerate(_BARK_BANDS_HZ, start=1):
        m = (freqs >= f0) & (freqs < f1)
        if not np.any(m):
            continue
        band_pow = float(np.sum(psd[m]) * df)
        n_prime = max(0.0, band_pow ** 0.23)
        z_center = float(idx_z) - 0.5
        num_s += n_prime * gz * z_center
        den_s += n_prime
    sharpness = finite_or_none(0.11 * num_s / den_s, digits=3) if den_s > 1e-12 else None

    # 2. NDSI: (biophony 2-11 kHz - anthrophony 0.5-2 kHz) / sum (standard ecoacoustics form)
    bio_m = (freqs >= 2000.0) & (freqs <= min(11000.0, float(freqs[-1])))
    anth_m = (freqs >= 500.0) & (freqs < 2000.0)
    e_bio = float(np.sum(psd[bio_m])) if np.any(bio_m) else 0.0
    e_anth = float(np.sum(psd[anth_m])) if np.any(anth_m) else 0.0
    ndsi = (
        finite_or_none((e_bio - e_anth) / (e_bio + e_anth), digits=4)
        if (e_bio + e_anth) > 1e-24
        else None
    )

    # 3. Log-frequency spectral slope (dB/octave across 100 - 8000 Hz)
    sl_m = (freqs >= 100.0) & (freqs <= min(8000.0, float(freqs[-1])))
    slope_oct: float | None = None
    if np.sum(sl_m) >= 6:
        log_f = np.log2(freqs[sl_m])
        p_db = 10.0 * np.log10(np.maximum(psd[sl_m], 1e-24))
        x_zm = log_f - np.mean(log_f)
        denom_x = float(np.dot(x_zm, x_zm))
        if denom_x > 1e-12:
            raw_sl = float(np.dot(x_zm, p_db - np.mean(p_db)) / denom_x)
            slope_oct = finite_or_none(raw_sl, digits=2)

    # 4. Roughness index from 20-70 Hz amplitude modulation blocks
    roughness: float | None = None
    if roughness_values is not None and len(roughness_values) > 0:
        r_arr = np.asarray(roughness_values, dtype=np.float64)
        r_arr = r_arr[np.isfinite(r_arr)]
        if r_arr.size > 0:
            roughness = finite_or_none(float(np.median(r_arr)), digits=4)

    # 5. Composite 0-100 pleasantness heuristic (penalizes shrillness, rumble, drones, roughness)
    score = 82.0
    if sharpness is not None:
        # Optimal warm ambient sharpness is ~0.9 - 1.6 acum
        score -= max(0.0, sharpness - 1.6) * 14.0
    if rumble_ratio_db is not None:
        score -= max(0.0, rumble_ratio_db + 8.0) * 1.4
    if roughness is not None:
        score -= max(0.0, roughness - 0.12) * 45.0
    score -= min(24.0, float(num_drones) * 6.0)
    if slope_oct is not None and slope_oct > 0.0:
        score -= min(18.0, slope_oct * 3.0)
    pleasantness = finite_or_none(max(0.0, min(100.0, score)), digits=1)

    return {
        "sharpness_acum": sharpness,
        "roughness_index": roughness,
        "ndsi": ndsi,
        "spectral_slope_db_per_oct": slope_oct,
        "pleasantness_score": pleasantness,
    }


def block_roughness_sample(mono: np.ndarray, sample_rate: int) -> float | None:
    """Estimate 20-70 Hz modulation roughness index for an audio block."""
    x = np.asarray(mono, dtype=np.float64).ravel()
    if x.size < max(512, int(0.4 * sample_rate)) or sample_rate < 2000:
        return None
    f_hi = min(6000.0, sample_rate * 0.45)
    if f_hi <= 600.0:
        return None
    sos_bp = signal.butter(4, [500.0, f_hi], btype="bandpass", fs=sample_rate, output="sos")
    bp = signal.sosfiltfilt(sos_bp, x)
    win = max(8, int(round(0.004 * sample_rate)))  # ~250 Hz envelope sample rate
    n_b = bp.size // win
    if n_b < 32:
        return None
    env = np.sqrt(np.mean((bp[: n_b * win].reshape(n_b, win)) ** 2, axis=1))
    env_zm = env - np.mean(env)
    tot = float(np.dot(env_zm, env_zm))
    if tot <= 1e-18:
        return 0.0
    fs_e = sample_rate / float(win)
    hi_mod = min(70.0, fs_e * 0.45)
    if hi_mod <= 22.0:
        return 0.0
    sos_mod = signal.butter(2, [20.0, hi_mod], btype="bandpass", fs=fs_e, output="sos")
    m_sig = signal.sosfilt(sos_mod, env_zm)
    mod_e = float(np.dot(m_sig, m_sig))
    return min(1.0, math.sqrt(mod_e / tot))


def compute_comparison_guardrails(
    ref_audio: np.ndarray,
    proc_audio: np.ndarray,
    sample_rate: int,
) -> dict[str, Any]:
    """Compute bird_retention_db, musical_noise_index, and automated guardrail warnings."""
    ref = ensure_2d_audio(ref_audio)
    proc = ensure_2d_audio(proc_audio)
    n = min(ref.shape[0], proc.shape[0])
    if n < 256 or sample_rate <= 0:
        return {
            "bird_retention_db": None,
            "bird_retention_raw_db": None,
            "gain_offset_db": None,
            "musical_noise_index": None,
            "guardrail_warnings": [],
        }

    # Cap comparison slice at 30 s to keep memory and runtime bounded
    cap = min(n, int(sample_rate * 30))
    ref_m = np.mean(ref[:cap], axis=1)
    proc_m = np.mean(proc[:cap], axis=1)
    rms_r = float(np.sqrt(np.mean(ref_m**2)))
    rms_p = float(np.sqrt(np.mean(proc_m**2)))
    gain_ratio = (rms_p / rms_r) if (rms_r > 1e-12 and rms_p > 1e-12) else 1.0
    gain_offset_db = finite_or_none(20.0 * math.log10(max(gain_ratio, 1e-12)), digits=2)

    nperseg = min(2048, cap)
    freqs, times, z_r = signal.stft(ref_m, fs=sample_rate, nperseg=nperseg, noverlap=nperseg // 2)
    _, _, z_p = signal.stft(proc_m, fs=sample_rate, nperseg=nperseg, noverlap=nperseg // 2)
    mag_r = np.abs(z_r)
    mag_p = np.abs(z_p)
    mag_p_norm = mag_p / max(gain_ratio, 1e-12)

    protect = compute_bird_protect_mask(freqs, times, mag_r)
    bird_mask = protect >= 0.45
    bird_retention_db: float | None = None
    bird_retention_raw_db: float | None = None
    if int(np.sum(bird_mask)) >= 8:
        e_ref = float(np.mean(mag_r[bird_mask] ** 2))
        e_proc_raw = float(np.mean(mag_p[bird_mask] ** 2))
        e_proc_norm = float(np.mean(mag_p_norm[bird_mask] ** 2))
        if e_ref > 1e-18:
            bird_retention_raw_db = finite_or_none(
                10.0 * math.log10(max(e_proc_raw, 1e-18) / e_ref), digits=2
            )
            bird_retention_db = finite_or_none(
                10.0 * math.log10(max(e_proc_norm, 1e-18) / e_ref), digits=2
            )

    # Musical-noise index: spectral flux / kurtosis ratio in non-bird stationary frames
    f_band = (freqs >= 300.0) & (freqs <= min(8000.0, sample_rate * 0.45))
    musical_noise_index: float | None = None
    if np.sum(f_band) >= 8 and mag_p.shape[1] >= 4:
        sub_r = 10.0 * np.log10(np.maximum(mag_r[f_band] ** 2, 1e-24))
        sub_p = 10.0 * np.log10(np.maximum(mag_p[f_band] ** 2, 1e-24))
        var_t_r = float(np.median(np.std(np.diff(sub_r, axis=1), axis=1)))
        var_t_p = float(np.median(np.std(np.diff(sub_p, axis=1), axis=1)))
        if var_t_r > 1e-6:
            mn_val = max(0.0, math.log(max(var_t_p / var_t_r, 0.25)))
            musical_noise_index = finite_or_none(mn_val, digits=3)

    warnings: list[str] = []
    if bird_retention_db is not None and bird_retention_db < -1.5:
        warnings.append(
            f"Bird/tonal syllable energy dropped by {bird_retention_db:.2f} dB; "
            "tighten protect_birds or lower strength."
        )
    if musical_noise_index is not None and musical_noise_index > 0.35:
        warnings.append(
            f"Elevated musical-noise index ({musical_noise_index:.2f}); "
            "raise texture_fill or soften spectral_denoise."
        )
    return {
        "bird_retention_db": bird_retention_db,
        "bird_retention_raw_db": bird_retention_raw_db,
        "gain_offset_db": gain_offset_db,
        "musical_noise_index": musical_noise_index,
        "guardrail_warnings": warnings,
    }


def assess_do_no_harm(metrics: Any, events: list[DistractionEvent]) -> dict[str, Any]:
    """Advise whether cleanup DSP is warranted or plain mastering alone preserves quality."""
    localized_active = [
        ev
        for ev in events
        if ev.kind != "impulsive_click"
        and ev.severity in {"medium", "high"}
    ]
    rumble = getattr(metrics, "rumble_ratio_db", None)
    bal = getattr(metrics, "channel_balance_db", None)
    hums = getattr(metrics, "hum_peaks_hz", []) or []
    drones = getattr(metrics, "tonal_peaks_hz", []) or []
    clipped = getattr(metrics, "clipped_samples", 0) or 0

    reasons: list[str] = []
    if localized_active:
        reasons.append(f"{len(localized_active)} medium/high distraction event(s)")
    if hums or drones:
        reasons.append("steady hum/tonal peak(s)")
    if clipped > 0:
        reasons.append(f"{clipped} clipped sample(s)")
    if rumble is not None and rumble > -6.0:
        reasons.append(f"elevated rumble_ratio_db ({rumble:.1f} dB)")
    if bal is not None and abs(bal) > 3.5:
        reasons.append(f"channel imbalance ({bal:+.1f} dB)")

    if not reasons:
        return {
            "recommend_cleanup": False,
            "do_no_harm_reason": (
                "Clean baseline (no medium/high distractions, hum, drones, clipping, or rumble). "
                "Prefer Master-only loudness/bed_shape rather than aggressive cleanup."
            ),
        }
    return {
        "recommend_cleanup": True,
        "do_no_harm_reason": "Cleanup warranted due to: " + "; ".join(reasons) + ".",
    }
