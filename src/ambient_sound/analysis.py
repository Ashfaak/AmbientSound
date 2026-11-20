"""Psychoacoustic analysis and ambient distraction event detection."""

from __future__ import annotations

import math

import numpy as np
import pyloudnorm as pyln
from scipy import signal

from ambient_sound.io_audio import ensure_2d_audio
from ambient_sound.models import AudioMetrics, DistractionEvent


def _finite_or_none(value: float | np.floating | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    val = float(value)
    if not math.isfinite(val):
        return None
    return round(val, digits)


def _db_from_linear(value: float) -> float | None:
    if value <= 0.0 or not math.isfinite(value):
        return None
    return _finite_or_none(20.0 * math.log10(value))


def _merge_intervals(
    mask: np.ndarray,
    frame_times: np.ndarray,
    frame_dur_s: float,
    max_gap_s: float,
) -> list[tuple[float, float]]:
    """Merge boolean frame detections separated by <= max_gap_s into (start_s, end_s)."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []

    spans: list[tuple[float, float]] = []
    cur_start = float(frame_times[indices[0]])
    cur_end = cur_start + frame_dur_s

    for idx in indices[1:]:
        t_start = float(frame_times[idx])
        t_end = t_start + frame_dur_s
        if t_start - cur_end <= max_gap_s:
            cur_end = max(cur_end, t_end)
        else:
            spans.append((cur_start, cur_end))
            cur_start, cur_end = t_start, t_end
    spans.append((cur_start, cur_end))
    return spans


def _detect_hum_peaks(
    freqs: np.ndarray,
    psd: np.ndarray,
    sample_rate: int,
) -> list[float]:
    """Return 50 Hz or 60 Hz harmonic frequencies exhibiting > 10 dB narrowband prominence."""
    nyquist = sample_rate * 0.5
    if freqs.size < 16:
        return []

    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-24))
    detected: list[float] = []

    for base in (50.0, 60.0):
        series_peaks: list[tuple[float, float]] = []
        for harmonic in range(1, 5):
            f_target = base * harmonic
            if f_target >= nyquist * 0.9:
                break
            peak_mask = (freqs >= f_target - 3.0) & (freqs <= f_target + 3.0)
            nbr_mask = ((freqs >= max(10.0, f_target - 20.0)) & (freqs <= f_target - 6.0)) | (
                (freqs >= f_target + 6.0) & (freqs <= min(nyquist, f_target + 20.0))
            )
            if not np.any(peak_mask) or not np.any(nbr_mask):
                continue
            peak_val = float(np.max(psd_db[peak_mask]))
            nbr_med = float(np.median(psd_db[nbr_mask]))
            prominence = peak_val - nbr_med
            if prominence > 10.0 and peak_val > -110.0:
                series_peaks.append((f_target, prominence))
        if series_peaks and (
            any(f == base and p > 11.0 for f, p in series_peaks) or len(series_peaks) >= 2
        ):
            # Keep if fundamental is strong or multiple harmonics appear
            detected.extend(f for f, _ in series_peaks)

    return sorted({round(f, 1) for f in detected})


def _detect_rumble_events(
    mono: np.ndarray,
    sample_rate: int,
    duration_s: float,
) -> list[DistractionEvent]:
    """Detect localized low-frequency wind/handling rumble gusts (< 80 Hz)."""
    if sample_rate <= 220 or mono.size < int(sample_rate * 0.5):
        return []

    sos = signal.butter(
        4, min(80.0, sample_rate * 0.35), btype="lowpass", fs=sample_rate, output="sos"
    )
    low = signal.sosfiltfilt(sos, mono)

    win = max(64, int(0.25 * sample_rate))
    hop = max(32, int(0.125 * sample_rate))
    if low.size < win:
        return []

    n_frames = 1 + (low.size - win) // hop
    times = np.arange(n_frames) * (hop / sample_rate)
    rms_db = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = low[i * hop : i * hop + win]
        rms_db[i] = 20.0 * math.log10(max(float(np.sqrt(np.mean(seg * seg))), 1e-12))

    med = float(np.median(rms_db))
    mask = (rms_db > med + 8.0) & (rms_db > -55.0)
    spans = _merge_intervals(mask, times, win / sample_rate, max_gap_s=0.25)

    events: list[DistractionEvent] = []
    for s, e in spans:
        e_clamped = min(e, duration_s)
        if e_clamped - s < 0.2:
            continue
        seg_mask = (times >= s) & (times <= e_clamped)
        peak_db = float(np.max(rms_db[seg_mask])) if np.any(seg_mask) else med + 8.0
        excess = peak_db - med
        sev = "high" if excess > 16.0 else ("medium" if excess > 11.0 else "low")
        events.append(
            DistractionEvent(
                id="pending",
                start_s=round(s, 3),
                end_s=round(e_clamped, 3),
                kind="low_freq_rumble",
                severity=sev,
                freq_band_hz=(20.0, 80.0),
                peak_db=_finite_or_none(peak_db),
                notes=f"Low-frequency rumble +{excess:.1f} dB above floor (<80 Hz).",
            )
        )
    return events


def _detect_click_events(
    audio: np.ndarray,
    sample_rate: int,
    duration_s: float,
) -> list[DistractionEvent]:
    """Detect impulsive clicks/pops via high-frequency derivative MAD outlier score."""
    if audio.shape[0] < 64:
        return []

    cutoff = min(1500.0, sample_rate * 0.35)
    sos = signal.butter(4, cutoff, btype="highpass", fs=sample_rate, output="sos")
    hp = signal.sosfiltfilt(sos, audio, axis=0)
    env = np.max(np.abs(hp), axis=1)

    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med))) * 1.4826
    sigma = max(mad, 1e-7)
    z = (env - med) / sigma
    hits = np.flatnonzero((z > 10.0) & (env > 5e-3))
    if hits.size == 0:
        return []

    merge_samples = int(0.05 * sample_rate)
    clusters: list[tuple[int, int]] = []
    start_idx = int(hits[0])
    end_idx = int(hits[0])
    for idx in hits[1:]:
        i = int(idx)
        if i - end_idx <= merge_samples:
            end_idx = i
        else:
            clusters.append((start_idx, end_idx))
            start_idx = end_idx = i
    clusters.append((start_idx, end_idx))

    events: list[DistractionEvent] = []
    for s_idx, e_idx in clusters:
        # Only treat short bursts (< 120 ms) as impulsive clicks; longer ones are transients
        if (e_idx - s_idx) / sample_rate > 0.12:
            continue
        pad = int(0.005 * sample_rate)
        s_s = max(0.0, (s_idx - pad) / sample_rate)
        e_s = min(duration_s, (e_idx + pad + 1) / sample_rate)
        peak_lin = float(np.max(np.abs(audio[max(0, s_idx - 2) : min(audio.shape[0], e_idx + 3)])))
        peak_z = float(np.max(z[s_idx : e_idx + 1]))
        sev = "high" if peak_z > 30.0 or peak_lin > 0.5 else ("medium" if peak_z > 16.0 else "low")
        events.append(
            DistractionEvent(
                id="pending",
                start_s=round(s_s, 4),
                end_s=round(e_s, 4),
                kind="impulsive_click",
                severity=sev,
                freq_band_hz=(round(cutoff, 1), round(sample_rate * 0.48, 1)),
                peak_db=_db_from_linear(peak_lin),
                notes=f"Impulsive spike ({peak_z:.1f} sigma above HF baseline).",
            )
        )
    return events


def _detect_transient_distractions(
    mono: np.ndarray,
    sample_rate: int,
    duration_s: float,
) -> list[DistractionEvent]:
    """Detect localized broadband/tonal distractions exceeding ambient RMS bed."""
    win = max(128, int(0.10 * sample_rate))
    hop = max(64, int(0.05 * sample_rate))
    if mono.size < win * 4:
        return []

    n_frames = 1 + (mono.size - win) // hop
    times = np.arange(n_frames) * (hop / sample_rate)
    rms_db = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = mono[i * hop : i * hop + win]
        rms_db[i] = 20.0 * math.log10(max(float(np.sqrt(np.mean(seg * seg))), 1e-12))

    med = float(np.median(rms_db))
    mad = float(np.median(np.abs(rms_db - med))) * 1.4826
    threshold = max(med + 8.5, med + 4.5 * max(mad, 1.0))
    mask = (rms_db > threshold) & (rms_db > -55.0)
    spans = _merge_intervals(mask, times, win / sample_rate, max_gap_s=0.15)

    # Ambient median spectrum for identifying which band stands out
    n_fft = min(2048, mono.size)
    freqs_ref, psd_ref = signal.welch(mono, fs=sample_rate, nperseg=n_fft)

    events: list[DistractionEvent] = []
    for s, e in spans:
        e_clamped = min(e, duration_s)
        span_dur = e_clamped - s
        if span_dur < 0.12 or span_dur > 20.0:
            continue
        i0 = max(0, int(s * sample_rate))
        i1 = min(mono.size, int(e_clamped * sample_rate))
        seg = mono[i0:i1]
        if seg.size < 64:
            continue

        seg_peak = float(np.max(np.abs(seg)))
        seg_rms_db = 20.0 * math.log10(max(float(np.sqrt(np.mean(seg * seg))), 1e-12))
        excess = seg_rms_db - med

        # Compare event spectrum against global ambient spectrum to localize freq_band_hz
        f_ev, p_ev = signal.welch(seg, fs=sample_rate, nperseg=min(n_fft, seg.size))
        p_ref_interp = np.interp(f_ev, freqs_ref, psd_ref)
        diff_psd = np.maximum(p_ev - p_ref_interp, 0.0)
        total_diff = float(np.sum(diff_psd))
        if total_diff > 0.0:
            cum = np.cumsum(diff_psd) / total_diff
            low_idx = int(np.searchsorted(cum, 0.10))
            high_idx = int(np.searchsorted(cum, 0.90))
            low_hz = round(float(f_ev[min(low_idx, f_ev.size - 1)]), 1)
            high_hz = round(float(f_ev[min(max(high_idx, low_idx + 1), f_ev.size - 1)]), 1)
        else:
            low_hz, high_hz = 80.0, round(sample_rate * 0.45, 1)

        sev = "high" if excess > 15.0 else ("medium" if excess > 10.5 else "low")
        events.append(
            DistractionEvent(
                id="pending",
                start_s=round(s, 3),
                end_s=round(e_clamped, 3),
                kind="transient_distraction",
                severity=sev,
                freq_band_hz=(low_hz, max(high_hz, low_hz + 20.0)),
                peak_db=_db_from_linear(seg_peak),
                notes=f"Localized foreground event +{excess:.1f} dB over ambient bed.",
            )
        )
    return events


def _detect_clipping_events(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[int, list[DistractionEvent]]:
    clipped_mask = np.any(np.abs(audio) >= 0.9999, axis=1)
    clipped_count = int(np.sum(np.abs(audio) >= 0.9999))
    if clipped_count == 0:
        return 0, []

    indices = np.flatnonzero(clipped_mask)
    merge_gap = int(0.1 * sample_rate)
    spans: list[tuple[int, int]] = []
    s = e = int(indices[0])
    for idx in indices[1:]:
        i = int(idx)
        if i - e <= merge_gap:
            e = i
        else:
            spans.append((s, e))
            s = e = i
    spans.append((s, e))

    events = [
        DistractionEvent(
            id="pending",
            start_s=round(s_i / sample_rate, 4),
            end_s=round((e_i + 1) / sample_rate, 4),
            kind="clipping",
            severity="high",
            freq_band_hz=None,
            peak_db=0.0,
            notes="Digital full-scale saturation (|sample| >= 0.9999).",
        )
        for s_i, e_i in spans
    ]
    return clipped_count, events


def analyze_audio(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[AudioMetrics, list[DistractionEvent]]:
    """Compute finite psychoacoustic metrics and detect localized ambient distractions."""
    arr = ensure_2d_audio(audio)
    num_samples, num_channels = arr.shape
    duration_s = num_samples / float(sample_rate) if sample_rate > 0 else 0.0

    if num_samples == 0 or sample_rate <= 0:
        metrics = AudioMetrics(
            duration_s=0.0,
            sample_rate=int(sample_rate),
            channels=int(num_channels),
        )
        return metrics, []

    mono = np.mean(arr, axis=1)
    rms_lin = float(np.sqrt(np.mean(arr * arr)))
    peak_lin = float(np.max(np.abs(arr)))

    if num_samples >= 4:
        upsampled = signal.resample_poly(arr, 4, 1, axis=0)
        true_peak_lin = float(np.max(np.abs(upsampled)))
    else:
        true_peak_lin = peak_lin

    rms_dbfs = _db_from_linear(rms_lin)
    peak_dbfs = _db_from_linear(peak_lin)
    true_peak_dbtp = _db_from_linear(true_peak_lin)
    crest_factor_db = (
        _finite_or_none(peak_dbfs - rms_dbfs)
        if peak_dbfs is not None and rms_dbfs is not None
        else None
    )
    dc_offset = _finite_or_none(float(np.mean(arr)), digits=6)

    # Integrated LUFS (pyloudnorm requires >= 0.4 s default block)
    integrated_lufs: float | None = None
    if duration_s >= 0.45 and rms_lin > 1e-10:
        try:
            meter = pyln.Meter(sample_rate)
            lufs_val = float(meter.integrated_loudness(arr))
            integrated_lufs = _finite_or_none(lufs_val)
        except Exception:
            integrated_lufs = None

    # Stereo coherence & Mid/Side ratio
    stereo_correlation: float | None = None
    mid_side_ratio_db: float | None = None
    if num_channels >= 2:
        l_ch = arr[:, 0]
        r_ch = arr[:, 1]
        l_zm = l_ch - np.mean(l_ch)
        r_zm = r_ch - np.mean(r_ch)
        denom = float(np.linalg.norm(l_zm) * np.linalg.norm(r_zm))
        if denom > 1e-12:
            corr = max(-1.0, min(1.0, float(np.dot(l_zm, r_zm) / denom)))
            stereo_correlation = _finite_or_none(corr)

        mid = 0.5 * (l_ch + r_ch)
        side = 0.5 * (l_ch - r_ch)
        mid_rms = float(np.sqrt(np.mean(mid * mid)))
        side_rms = float(np.sqrt(np.mean(side * side)))
        if mid_rms > 1e-12:
            ratio_db = 20.0 * math.log10(mid_rms / max(side_rms, 1e-12))
            mid_side_ratio_db = _finite_or_none(ratio_db)

    # Spectral metrics via Welch PSD
    centroid_hz: float | None = None
    rolloff_hz: float | None = None
    flatness: float | None = None
    rumble_ratio_db: float | None = None
    hum_peaks_hz: list[float] = []

    if num_samples >= 64 and float(np.max(np.abs(mono))) > 1e-12:
        nperseg = min(num_samples, 16384)
        freqs, psd = signal.welch(mono, fs=sample_rate, nperseg=nperseg)
        psd_sum = float(np.sum(psd))
        if psd_sum > 0.0:
            centroid_hz = _finite_or_none(float(np.sum(freqs * psd) / psd_sum), digits=2)
            cum_energy = np.cumsum(psd) / psd_sum
            rolloff_idx = int(np.searchsorted(cum_energy, 0.85))
            rolloff_hz = _finite_or_none(
                float(freqs[min(rolloff_idx, freqs.size - 1)]), digits=2
            )
            pos_psd = psd[freqs >= 20.0]
            if pos_psd.size > 0:
                eps = 1e-24
                geom_mean = float(np.exp(np.mean(np.log(pos_psd + eps))))
                arith_mean = float(np.mean(pos_psd)) + eps
                flatness = _finite_or_none(geom_mean / arith_mean, digits=5)

            low_energy = float(np.sum(psd[freqs < 80.0]))
            body_energy = float(np.sum(psd[(freqs >= 80.0) & (freqs <= 20000.0)]))
            if body_energy > 1e-24 and low_energy > 0.0:
                rumble_ratio_db = _finite_or_none(
                    10.0 * math.log10(low_energy / body_energy)
                )

            hum_peaks_hz = _detect_hum_peaks(freqs, psd, sample_rate)

    clipped_samples, clip_events = _detect_clipping_events(arr, sample_rate)

    events: list[DistractionEvent] = []
    if hum_peaks_hz:
        is_50_series = any(abs(f - 50.0) < 1.0 or abs(f - 100.0) < 1.0 for f in hum_peaks_hz)
        base_guess = 50.0 if is_50_series else 60.0
        events.append(
            DistractionEvent(
                id="pending",
                start_s=0.0,
                end_s=round(duration_s, 3),
                kind="mains_hum",
                severity="high" if len(hum_peaks_hz) >= 2 else "medium",
                freq_band_hz=(round(min(hum_peaks_hz) - 5.0, 1), round(max(hum_peaks_hz) + 5.0, 1)),
                peak_db=None,
                notes=f"Detected {base_guess:.0f} Hz harmonic hum peaks at {hum_peaks_hz} Hz.",
            )
        )

    events.extend(_detect_rumble_events(mono, sample_rate, duration_s))
    events.extend(_detect_click_events(arr, sample_rate, duration_s))
    events.extend(_detect_transient_distractions(mono, sample_rate, duration_s))
    events.extend(clip_events)

    events.sort(key=lambda ev: (ev.start_s, ev.end_s))
    for idx, ev in enumerate(events, start=1):
        ev.id = f"evt_{idx:03d}"

    metrics = AudioMetrics(
        duration_s=round(duration_s, 4),
        sample_rate=int(sample_rate),
        channels=int(num_channels),
        integrated_lufs=integrated_lufs,
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        true_peak_dbtp=true_peak_dbtp,
        crest_factor_db=crest_factor_db,
        dc_offset=dc_offset,
        clipped_samples=clipped_samples,
        stereo_correlation=stereo_correlation,
        mid_side_ratio_db=mid_side_ratio_db,
        spectral_centroid_hz=centroid_hz,
        spectral_rolloff_hz=rolloff_hz,
        spectral_flatness=flatness,
        rumble_ratio_db=rumble_ratio_db,
        hum_peaks_hz=hum_peaks_hz,
    )
    return metrics, events
