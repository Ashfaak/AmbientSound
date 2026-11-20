"""Low-RAM streaming psychoacoustic analysis and configurable distraction detection."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
import pyloudnorm as pyln
from scipy import signal

from ambient_sound.io_audio import (
    ensure_2d_audio,
    iter_audio_blocks,
    load_audio_slice,
    probe_audio,
)
from ambient_sound.models import AudioMetrics, DistractionEvent

SensitivityLevel = Literal["low", "medium", "high"]

_SENSITIVITY_PARAMS: dict[str, tuple[float, float]] = {
    "low": (11.0, 5.5),
    "medium": (8.0, 4.2),
    "high": (5.5, 3.2),
}


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


def chunked_true_peak_linear(audio: np.ndarray, chunk_samples: int = 96_000) -> float:
    """Compute 4x oversampled true peak linear amplitude in bounded RAM."""
    arr = ensure_2d_audio(audio)
    n = arr.shape[0]
    if n == 0:
        return 0.0
    if n < 4:
        return float(np.max(np.abs(arr)))

    pad = 64
    peak = float(np.max(np.abs(arr)))
    for pos in range(0, n, chunk_samples):
        s = max(0, pos - pad)
        e = min(n, pos + chunk_samples + pad)
        up = signal.resample_poly(arr[s:e], 4, 1, axis=0)
        trim_l = (pos - s) * 4
        trim_r = trim_l + min(chunk_samples, n - pos) * 4
        seg = up[trim_l:trim_r]
        if seg.size:
            peak = max(peak, float(np.max(np.abs(seg))))
    return peak


class StreamingTruePeak:
    """Stateful chunked 4x oversampled true-peak tracker."""

    def __init__(self, channels: int) -> None:
        self.channels = channels
        self._pad = 64
        self._tail = np.zeros((0, channels), dtype=np.float64)
        self.peak_lin = 0.0

    def update(self, block: np.ndarray) -> None:
        arr = ensure_2d_audio(block)
        if arr.shape[0] == 0:
            return
        self.peak_lin = max(self.peak_lin, float(np.max(np.abs(arr))))
        combined = np.vstack([self._tail, arr]) if self._tail.size else arr
        if combined.shape[0] >= 4:
            up = signal.resample_poly(combined, 4, 1, axis=0)
            skip = self._tail.shape[0] * 4
            valid = up[skip:]
            if valid.size:
                self.peak_lin = max(self.peak_lin, float(np.max(np.abs(valid))))
        keep = min(self._pad, combined.shape[0])
        self._tail = combined[-keep:].copy()


class StreamingLUFS:
    """Bounded-RAM ITU-R BS.1770-4 integrated loudness accumulator."""

    def __init__(self, sample_rate: int, channels: int) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.sub_len = max(1, int(round(0.1 * self.sample_rate)))
        self._total_samples = 0
        self._sub_energies: list[np.ndarray] = []
        self._tail = np.zeros((0, self.channels), dtype=np.float64)

        meter = pyln.Meter(self.sample_rate)
        self._stages: list[tuple[np.ndarray, np.ndarray, float, np.ndarray]] = []
        for stage in meter._filters.values():
            b = np.asarray(stage.b, dtype=np.float64)
            a = np.asarray(stage.a, dtype=np.float64)
            gain = float(stage.passband_gain)
            order = max(len(b), len(a)) - 1
            zi = np.zeros((order, self.channels), dtype=np.float64)
            self._stages.append((b, a, gain, zi))

    def update(self, block: np.ndarray) -> None:
        arr = ensure_2d_audio(block)
        if arr.shape[0] == 0:
            return
        self._total_samples += arr.shape[0]

        filtered = arr
        new_stages = []
        for b, a, gain, zi in self._stages:
            y, zi_next = signal.lfilter(b, a, filtered, axis=0, zi=zi)
            filtered = gain * y
            new_stages.append((b, a, gain, zi_next))
        self._stages = new_stages

        if self._tail.size:
            filtered = np.vstack([self._tail, filtered])

        n_full = filtered.shape[0] // self.sub_len
        if n_full > 0:
            usable = filtered[: n_full * self.sub_len]
            reshaped = usable.reshape(n_full, self.sub_len, self.channels)
            energies = np.mean(reshaped * reshaped, axis=1)
            self._sub_energies.extend(energies)
            self._tail = filtered[n_full * self.sub_len :].copy()
        else:
            self._tail = filtered.copy()

    def finalize(self) -> float | None:
        duration_s = self._total_samples / float(max(self.sample_rate, 1))
        if duration_s < 0.45 or len(self._sub_energies) < 4:
            return None

        subs = np.asarray(self._sub_energies, dtype=np.float64)  # (N_sub, C)
        # 400 ms gating blocks = 4 consecutive 100 ms sub-blocks (75% overlap)
        blocks = (subs[:-3] + subs[1:-2] + subs[2:-1] + subs[3:]) * 0.25
        weights = np.array([1.0, 1.0, 1.0, 1.41, 1.41], dtype=np.float64)
        if self.channels <= 5:
            g = weights[: self.channels]
        else:
            g = np.concatenate([weights, np.ones(self.channels - 5, dtype=np.float64)])

        weighted_sum = blocks @ g
        with np.errstate(divide="ignore", invalid="ignore"):
            l_blocks = -0.691 + 10.0 * np.log10(np.maximum(weighted_sum, 1e-24))

        abs_mask = l_blocks >= -70.0
        if not np.any(abs_mask):
            return None

        z_avg_rel = np.mean(blocks[abs_mask], axis=0)
        rel_sum = float(np.dot(g, z_avg_rel))
        if rel_sum <= 0.0:
            return None
        gamma_r = -0.691 + 10.0 * math.log10(rel_sum) - 10.0

        final_mask = (l_blocks > gamma_r) & (l_blocks > -70.0)
        if not np.any(final_mask):
            return None

        z_avg_final = np.mean(blocks[final_mask], axis=0)
        final_sum = float(np.dot(g, z_avg_final))
        if final_sum <= 0.0:
            return None
        return _finite_or_none(-0.691 + 10.0 * math.log10(final_sum))


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
            detected.extend(f for f, _ in series_peaks)

    return sorted({round(f, 1) for f in detected})


def _detect_tonal_peaks(
    freqs: np.ndarray,
    psd: np.ndarray,
    sample_rate: int,
    exclude_freqs: list[float],
) -> list[float]:
    """Detect arbitrary narrowband machine/drone tonal peaks between 30 Hz and 1000 Hz."""
    nyquist = sample_rate * 0.5
    max_f = min(1000.0, nyquist * 0.85)
    if freqs.size < 32 or max_f <= 40.0:
        return []

    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-24))
    band_indices = np.flatnonzero((freqs >= 30.0) & (freqs <= max_f))
    candidates: list[tuple[float, float]] = []

    for idx in band_indices:
        if idx <= 0 or idx >= freqs.size - 1:
            continue
        if psd_db[idx] < psd_db[idx - 1] or psd_db[idx] < psd_db[idx + 1]:
            continue
        f_cand = float(freqs[idx])
        if any(abs(f_cand - ex) <= 4.5 for ex in exclude_freqs):
            continue
        nbr_mask = ((freqs >= max(12.0, f_cand - 22.0)) & (freqs <= f_cand - 6.0)) | (
            (freqs >= f_cand + 6.0) & (freqs <= min(nyquist, f_cand + 22.0))
        )
        if not np.any(nbr_mask):
            continue
        nbr_med = float(np.median(psd_db[nbr_mask]))
        prom = float(psd_db[idx]) - nbr_med
        if prom > 12.0 and float(psd_db[idx]) > -105.0:
            candidates.append((f_cand, prom))

    candidates.sort(key=lambda pair: pair[1], reverse=True)
    selected: list[float] = []
    for f_cand, _ in candidates:
        if any(abs(f_cand - s) < 6.0 for s in selected):
            continue
        selected.append(round(f_cand, 1))
        if len(selected) >= 12:
            break
    return sorted(selected)


def _rumble_events_from_curve(
    times: np.ndarray,
    rms_db: np.ndarray,
    win_dur_s: float,
    duration_s: float,
    time_offset_s: float,
) -> list[DistractionEvent]:
    if rms_db.size < 4:
        return []
    med = float(np.median(rms_db))
    mask = (rms_db > med + 8.0) & (rms_db > -55.0)
    spans = _merge_intervals(mask, times, win_dur_s, max_gap_s=0.25)

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
                start_s=round(time_offset_s + s, 3),
                end_s=round(time_offset_s + e_clamped, 3),
                kind="low_freq_rumble",
                severity=sev,
                freq_band_hz=(20.0, 80.0),
                peak_db=_finite_or_none(peak_db),
                notes=f"Low-frequency rumble +{excess:.1f} dB above floor (<80 Hz).",
            )
        )
    return events


def _click_events_from_hits(
    hits: list[tuple[int, float, float]],
    sample_rate: int,
    duration_s: float,
    cutoff: float,
    time_offset_s: float,
) -> list[DistractionEvent]:
    if not hits:
        return []
    hits.sort(key=lambda x: x[0])
    merge_samples = int(0.05 * sample_rate)
    clusters: list[tuple[int, int, float, float]] = []

    s_idx, peak_lin, peak_z = hits[0]
    e_idx = s_idx
    for idx, p_lin, p_z in hits[1:]:
        if idx - e_idx <= merge_samples:
            e_idx = idx
            peak_lin = max(peak_lin, p_lin)
            peak_z = max(peak_z, p_z)
        else:
            clusters.append((s_idx, e_idx, peak_lin, peak_z))
            s_idx, e_idx, peak_lin, peak_z = idx, idx, p_lin, p_z
    clusters.append((s_idx, e_idx, peak_lin, peak_z))

    events: list[DistractionEvent] = []
    for c_s, c_e, p_lin, p_z in clusters:
        if (c_e - c_s) / sample_rate > 0.12:
            continue
        pad = int(0.005 * sample_rate)
        s_s = max(0.0, (c_s - pad) / sample_rate)
        e_s = min(duration_s, (c_e + pad + 1) / sample_rate)
        sev = "high" if p_z > 30.0 or p_lin > 0.5 else ("medium" if p_z > 16.0 else "low")
        events.append(
            DistractionEvent(
                id="pending",
                start_s=round(time_offset_s + s_s, 4),
                end_s=round(time_offset_s + e_s, 4),
                kind="impulsive_click",
                severity=sev,
                freq_band_hz=(round(cutoff, 1), round(sample_rate * 0.48, 1)),
                peak_db=_db_from_linear(p_lin),
                notes=f"Impulsive spike ({p_z:.1f} sigma above HF baseline).",
            )
        )
    return events


def _estimate_event_band(
    seg: np.ndarray,
    sample_rate: int,
    freqs_ref: np.ndarray,
    psd_ref: np.ndarray,
) -> tuple[float, float]:
    if seg.size < 64 or freqs_ref.size < 4:
        return 80.0, round(sample_rate * 0.45, 1)
    n_fft = min(2048, seg.size)
    f_ev, p_ev = signal.welch(seg, fs=sample_rate, nperseg=n_fft)
    p_ref_interp = np.interp(f_ev, freqs_ref, psd_ref)
    diff_psd = np.maximum(p_ev - p_ref_interp, 0.0)
    total_diff = float(np.sum(diff_psd))
    if total_diff <= 0.0:
        return 80.0, round(sample_rate * 0.45, 1)
    cum = np.cumsum(diff_psd) / total_diff
    low_idx = int(np.searchsorted(cum, 0.10))
    high_idx = int(np.searchsorted(cum, 0.90))
    low_hz = round(float(f_ev[min(low_idx, f_ev.size - 1)]), 1)
    high_hz = round(float(f_ev[min(max(high_idx, low_idx + 1), f_ev.size - 1)]), 1)
    return low_hz, max(high_hz, low_hz + 20.0)


def _foreground_events_from_curve(
    times: np.ndarray,
    rms_db: np.ndarray,
    win_dur_s: float,
    duration_s: float,
    sample_rate: int,
    freqs_ref: np.ndarray,
    psd_ref: np.ndarray,
    sensitivity: SensitivityLevel,
    min_event_s: float,
    max_event_s: float,
    freq_range_hz: tuple[float, float] | None,
    time_offset_s: float,
    slice_loader: tuple[Path, float] | np.ndarray,
) -> list[DistractionEvent]:
    if rms_db.size < 4:
        return []
    excess_req, mad_mult = _SENSITIVITY_PARAMS.get(sensitivity, _SENSITIVITY_PARAMS["medium"])
    med = float(np.median(rms_db))
    mad = float(np.median(np.abs(rms_db - med))) * 1.4826
    threshold = max(med + excess_req, med + mad_mult * max(mad, 1.0))
    mask = (rms_db > threshold) & (rms_db > -60.0)
    spans = _merge_intervals(mask, times, win_dur_s, max_gap_s=0.20)

    events: list[DistractionEvent] = []
    for s, e in spans:
        e_clamped = min(e, duration_s)
        span_dur = e_clamped - s
        if span_dur < min_event_s or span_dur > max_event_s:
            continue

        seg_mask = (times >= s) & (times <= e_clamped)
        seg_rms_db = float(np.max(rms_db[seg_mask])) if np.any(seg_mask) else med + excess_req
        excess = seg_rms_db - med

        abs_s = time_offset_s + s
        abs_e = time_offset_s + e_clamped
        if isinstance(slice_loader, np.ndarray):
            i0 = max(0, int(s * sample_rate))
            i1 = min(slice_loader.size, int(e_clamped * sample_rate))
            seg = slice_loader[i0:i1]
        else:
            src_path, _ = slice_loader
            probe_end = min(abs_e, abs_s + 8.0)
            seg_2d, _ = load_audio_slice(src_path, start_s=abs_s, end_s=probe_end, dtype="float64")
            seg = np.mean(seg_2d, axis=1) if seg_2d.size else np.zeros(0, dtype=np.float64)

        seg_peak = float(np.max(np.abs(seg))) if seg.size else 0.0
        band = freq_range_hz or _estimate_event_band(seg, sample_rate, freqs_ref, psd_ref)
        kind: Literal["slow_pass", "transient_distraction"] = (
            "slow_pass" if span_dur >= 3.0 else "transient_distraction"
        )
        desc = (
            "Sustained passing distraction"
            if kind == "slow_pass"
            else "Localized foreground event"
        )
        sev = "high" if excess > 15.0 else ("medium" if excess > 10.0 else "low")
        events.append(
            DistractionEvent(
                id="pending",
                start_s=round(abs_s, 3),
                end_s=round(abs_e, 3),
                kind=kind,
                severity=sev,
                freq_band_hz=(round(float(band[0]), 1), round(float(band[1]), 1)),
                peak_db=_db_from_linear(seg_peak),
                notes=f"{desc} ({span_dur:.2f}s, +{excess:.1f} dB over bed).",
            )
        )
    return events


def _build_global_events(
    duration_s: float,
    time_offset_s: float,
    hum_peaks_hz: list[float],
    tonal_peaks_hz: list[float],
) -> list[DistractionEvent]:
    events: list[DistractionEvent] = []
    s_abs = round(time_offset_s, 3)
    e_abs = round(time_offset_s + duration_s, 3)
    if hum_peaks_hz:
        is_50 = any(abs(f - 50.0) < 1.0 or abs(f - 100.0) < 1.0 for f in hum_peaks_hz)
        base_guess = 50.0 if is_50 else 60.0
        events.append(
            DistractionEvent(
                id="pending",
                start_s=s_abs,
                end_s=e_abs,
                kind="mains_hum",
                severity="high" if len(hum_peaks_hz) >= 2 else "medium",
                freq_band_hz=(
                    round(min(hum_peaks_hz) - 5.0, 1),
                    round(max(hum_peaks_hz) + 5.0, 1),
                ),
                peak_db=None,
                notes=f"Detected {base_guess:.0f} Hz harmonic hum peaks at {hum_peaks_hz} Hz.",
            )
        )
    if tonal_peaks_hz:
        events.append(
            DistractionEvent(
                id="pending",
                start_s=s_abs,
                end_s=e_abs,
                kind="machine_drone",
                severity="medium" if len(tonal_peaks_hz) == 1 else "high",
                freq_band_hz=(
                    round(min(tonal_peaks_hz) - 5.0, 1),
                    round(max(tonal_peaks_hz) + 5.0, 1),
                ),
                peak_db=None,
                notes=f"Detected persistent tonal drone peaks at {tonal_peaks_hz} Hz.",
            )
        )
    return events


def _finalize_spectral_summary(
    freqs: np.ndarray,
    psd: np.ndarray,
    sample_rate: int,
) -> tuple[float | None, float | None, float | None, float | None, list[float], list[float]]:
    psd_sum = float(np.sum(psd))
    if psd_sum <= 0.0:
        return None, None, None, None, [], []

    centroid_hz = _finite_or_none(float(np.sum(freqs * psd) / psd_sum), digits=2)
    cum_energy = np.cumsum(psd) / psd_sum
    rolloff_idx = int(np.searchsorted(cum_energy, 0.85))
    rolloff_hz = _finite_or_none(float(freqs[min(rolloff_idx, freqs.size - 1)]), digits=2)

    pos_psd = psd[freqs >= 20.0]
    flatness: float | None = None
    if pos_psd.size > 0:
        eps = 1e-24
        geom_mean = float(np.exp(np.mean(np.log(pos_psd + eps))))
        arith_mean = float(np.mean(pos_psd)) + eps
        flatness = _finite_or_none(geom_mean / arith_mean, digits=5)

    low_energy = float(np.sum(psd[freqs < 80.0]))
    body_energy = float(np.sum(psd[(freqs >= 80.0) & (freqs <= 20000.0)]))
    rumble_ratio_db: float | None = None
    if body_energy > 1e-24 and low_energy > 0.0:
        rumble_ratio_db = _finite_or_none(10.0 * math.log10(low_energy / body_energy))

    hum_peaks = _detect_hum_peaks(freqs, psd, sample_rate)
    tonal_peaks = _detect_tonal_peaks(freqs, psd, sample_rate, hum_peaks)
    return centroid_hz, rolloff_hz, flatness, rumble_ratio_db, hum_peaks, tonal_peaks


def analyze_audio(
    audio: np.ndarray,
    sample_rate: int,
    sensitivity: SensitivityLevel = "medium",
    min_event_s: float = 0.12,
    max_event_s: float = 45.0,
    freq_range_hz: tuple[float, float] | None = None,
    time_offset_s: float = 0.0,
) -> tuple[AudioMetrics, list[DistractionEvent]]:
    """Compute finite psychoacoustic metrics and detect distractions from an in-memory array."""
    arr = ensure_2d_audio(audio)
    num_samples, num_channels = arr.shape
    duration_s = num_samples / float(sample_rate) if sample_rate > 0 else 0.0

    if num_samples == 0 or sample_rate <= 0:
        return (
            AudioMetrics(duration_s=0.0, sample_rate=int(sample_rate), channels=int(num_channels)),
            [],
        )

    mono = np.mean(arr, axis=1)
    rms_lin = float(np.sqrt(np.mean(arr * arr)))
    peak_lin = float(np.max(np.abs(arr)))
    true_peak_lin = chunked_true_peak_linear(arr)

    rms_dbfs = _db_from_linear(rms_lin)
    peak_dbfs = _db_from_linear(peak_lin)
    true_peak_dbtp = _db_from_linear(true_peak_lin)
    crest_factor_db = (
        _finite_or_none(peak_dbfs - rms_dbfs)
        if peak_dbfs is not None and rms_dbfs is not None
        else None
    )
    dc_offset = _finite_or_none(float(np.mean(arr)), digits=6)

    integrated_lufs: float | None = None
    if duration_s >= 0.45 and rms_lin > 1e-10:
        lufs_acc = StreamingLUFS(sample_rate, num_channels)
        lufs_acc.update(arr)
        integrated_lufs = lufs_acc.finalize()

    stereo_correlation: float | None = None
    mid_side_ratio_db: float | None = None
    if num_channels >= 2:
        l_ch, r_ch = arr[:, 0], arr[:, 1]
        l_zm, r_zm = l_ch - np.mean(l_ch), r_ch - np.mean(r_ch)
        denom = float(np.linalg.norm(l_zm) * np.linalg.norm(r_zm))
        if denom > 1e-12:
            corr = max(-1.0, min(1.0, float(np.dot(l_zm, r_zm) / denom)))
            stereo_correlation = _finite_or_none(corr)
        mid = 0.5 * (l_ch + r_ch)
        side = 0.5 * (l_ch - r_ch)
        mid_rms = float(np.sqrt(np.mean(mid * mid)))
        side_rms = float(np.sqrt(np.mean(side * side)))
        if mid_rms > 1e-12:
            mid_side_ratio_db = _finite_or_none(20.0 * math.log10(mid_rms / max(side_rms, 1e-12)))

    centroid_hz = rolloff_hz = flatness = rumble_ratio_db = None
    hum_peaks_hz: list[float] = []
    tonal_peaks_hz: list[float] = []
    freqs_ref = np.zeros(0, dtype=np.float64)
    psd_ref = np.zeros(0, dtype=np.float64)

    if num_samples >= 64 and float(np.max(np.abs(mono))) > 1e-12:
        nperseg = min(num_samples, 16384)
        freqs_ref, psd_ref = signal.welch(mono, fs=sample_rate, nperseg=nperseg)
        (
            centroid_hz,
            rolloff_hz,
            flatness,
            rumble_ratio_db,
            hum_peaks_hz,
            tonal_peaks_hz,
        ) = _finalize_spectral_summary(freqs_ref, psd_ref, sample_rate)

    events: list[DistractionEvent] = _build_global_events(
        duration_s, time_offset_s, hum_peaks_hz, tonal_peaks_hz
    )

    # Rumble detector
    if sample_rate > 220 and mono.size >= int(sample_rate * 0.5):
        sos_lp = signal.butter(
            4, min(80.0, sample_rate * 0.35), btype="lowpass", fs=sample_rate, output="sos"
        )
        low = signal.sosfiltfilt(sos_lp, mono)
        win_r = max(64, int(0.25 * sample_rate))
        hop_r = max(32, int(0.125 * sample_rate))
        if low.size >= win_r:
            n_f = 1 + (low.size - win_r) // hop_r
            t_r = np.arange(n_f) * (hop_r / sample_rate)
            db_r = np.array(
                [
                    20.0
                    * math.log10(
                        max(float(np.sqrt(np.mean(low[i * hop_r : i * hop_r + win_r] ** 2))), 1e-12)
                    )
                    for i in range(n_f)
                ]
            )
            events.extend(
                _rumble_events_from_curve(t_r, db_r, win_r / sample_rate, duration_s, time_offset_s)
            )

    # Click detector
    if num_samples >= 64:
        cutoff = min(1500.0, sample_rate * 0.35)
        sos_hp = signal.butter(4, cutoff, btype="highpass", fs=sample_rate, output="sos")
        hp = signal.sosfiltfilt(sos_hp, arr, axis=0)
        env = np.max(np.abs(hp), axis=1)
        med = float(np.median(env))
        sigma = max(float(np.median(np.abs(env - med))) * 1.4826, 1e-7)
        z = (env - med) / sigma
        hit_idx = np.flatnonzero((z > 10.0) & (env > 5e-3))
        hits = [
            (int(i), float(np.max(np.abs(arr[i]))), float(z[i]))
            for i in hit_idx
        ]
        events.extend(_click_events_from_hits(hits, sample_rate, duration_s, cutoff, time_offset_s))

    # Foreground transient & slow_pass detector
    fg_signal = mono
    if freq_range_hz is not None and mono.size >= 64:
        f_lo = max(5.0, min(float(freq_range_hz[0]), sample_rate * 0.44))
        f_hi = max(f_lo + 5.0, min(float(freq_range_hz[1]), sample_rate * 0.48))
        sos_bp = signal.butter(4, [f_lo, f_hi], btype="bandpass", fs=sample_rate, output="sos")
        fg_signal = signal.sosfiltfilt(sos_bp, mono)

    win_fg = max(128, int(0.10 * sample_rate))
    hop_fg = max(64, int(0.05 * sample_rate))
    if fg_signal.size >= win_fg * 4:
        n_fg = 1 + (fg_signal.size - win_fg) // hop_fg
        t_fg = np.arange(n_fg) * (hop_fg / sample_rate)
        db_fg = np.array(
            [
                20.0
                * math.log10(
                    max(
                        float(
                            np.sqrt(np.mean(fg_signal[i * hop_fg : i * hop_fg + win_fg] ** 2))
                        ),
                        1e-12,
                    )
                )
                for i in range(n_fg)
            ]
        )
        events.extend(
            _foreground_events_from_curve(
                t_fg,
                db_fg,
                win_fg / sample_rate,
                duration_s,
                sample_rate,
                freqs_ref,
                psd_ref,
                sensitivity,
                min_event_s,
                max_event_s,
                freq_range_hz,
                time_offset_s,
                mono,
            )
        )

    # Clipping detector
    clipped_mask = np.any(np.abs(arr) >= 0.9999, axis=1)
    clipped_samples = int(np.sum(np.abs(arr) >= 0.9999))
    if clipped_samples > 0:
        idx_c = np.flatnonzero(clipped_mask)
        gap = int(0.1 * sample_rate)
        s_i = e_i = int(idx_c[0])
        for idx in idx_c[1:]:
            i = int(idx)
            if i - e_i <= gap:
                e_i = i
            else:
                events.append(
                    DistractionEvent(
                        id="pending",
                        start_s=round(time_offset_s + s_i / sample_rate, 4),
                        end_s=round(time_offset_s + (e_i + 1) / sample_rate, 4),
                        kind="clipping",
                        severity="high",
                        peak_db=0.0,
                        notes="Digital full-scale saturation (|sample| >= 0.9999).",
                    )
                )
                s_i = e_i = i
        events.append(
            DistractionEvent(
                id="pending",
                start_s=round(time_offset_s + s_i / sample_rate, 4),
                end_s=round(time_offset_s + (e_i + 1) / sample_rate, 4),
                kind="clipping",
                severity="high",
                peak_db=0.0,
                notes="Digital full-scale saturation (|sample| >= 0.9999).",
            )
        )

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
        tonal_peaks_hz=tonal_peaks_hz,
    )
    return metrics, events


def analyze_file(
    path: str | Path,
    start_s: float | None = None,
    end_s: float | None = None,
    sensitivity: SensitivityLevel = "medium",
    min_event_s: float = 0.12,
    max_event_s: float = 45.0,
    freq_range_hz: tuple[float, float] | None = None,
) -> tuple[AudioMetrics, list[DistractionEvent]]:
    """Stream-analyze an audio file (or sub-region) in bounded RAM (< 100 MB)."""
    src = Path(path)
    info = probe_audio(src)
    s0 = max(0.0, float(start_s)) if start_s is not None else 0.0
    s1 = min(info.duration_s, float(end_s)) if end_s is not None else info.duration_s

    if s1 <= s0 or info.frames == 0 or info.sample_rate <= 0:
        return (
            AudioMetrics(duration_s=0.0, sample_rate=info.sample_rate, channels=info.channels),
            [],
        )

    sr = info.sample_rate
    ch = info.channels

    # For short regions (< 90 s) or non-libsndfile containers, direct slice + analyze_audio is fast
    if (s1 - s0) <= 90.0 or info.format_name.upper() not in {"WAV", "FLAC", "OGG", "AIFF"}:
        try:
            slice_arr, slice_sr = load_audio_slice(src, start_s=s0, end_s=s1, dtype="float64")
            return analyze_audio(
                slice_arr,
                slice_sr,
                sensitivity=sensitivity,
                min_event_s=min_event_s,
                max_event_s=max_event_s,
                freq_range_hz=freq_range_hz,
                time_offset_s=s0,
            )
        except Exception:
            pass

    lufs_acc = StreamingLUFS(sr, ch)
    tp_acc = StreamingTruePeak(ch)

    total_frames = 0
    sum_all = 0.0
    sum_sq_all = 0.0
    peak_lin = 0.0
    clipped_samples = 0
    clip_indices: list[int] = []

    sum_l = sum_r = sum_l2 = sum_r2 = sum_lr = sum_mid2 = sum_side2 = 0.0

    nperseg = 16384
    freqs_ref = np.fft.rfftfreq(nperseg, d=1.0 / sr)
    psd_accum = np.zeros_like(freqs_ref, dtype=np.float64)
    psd_weight = 0

    # Stateful filters for rumble, click, and foreground curves
    sos_lp = signal.butter(4, min(80.0, sr * 0.35), btype="lowpass", fs=sr, output="sos")
    zi_lp = np.zeros((sos_lp.shape[0], 2), dtype=np.float64)

    cutoff_hp = min(1500.0, sr * 0.35)
    sos_hp = signal.butter(4, cutoff_hp, btype="highpass", fs=sr, output="sos")
    zi_hp = np.zeros((sos_hp.shape[0], 2, ch), dtype=np.float64)

    sos_bp = None
    zi_bp = None
    if freq_range_hz is not None:
        f_lo = max(5.0, min(float(freq_range_hz[0]), sr * 0.44))
        f_hi = max(f_lo + 5.0, min(float(freq_range_hz[1]), sr * 0.48))
        sos_bp = signal.butter(4, [f_lo, f_hi], btype="bandpass", fs=sr, output="sos")
        zi_bp = np.zeros((sos_bp.shape[0], 2), dtype=np.float64)

    hop_r = max(32, int(0.125 * sr))
    rumble_tail = np.zeros(0, dtype=np.float64)
    rumble_db_list: list[float] = []

    hop_fg = max(64, int(0.05 * sr))
    fg_tail = np.zeros(0, dtype=np.float64)
    fg_db_list: list[float] = []

    click_hits: list[tuple[int, float, float]] = []

    for abs_frame, block32 in iter_audio_blocks(
        src, block_seconds=30.0, start_s=s0, end_s=s1, dtype="float32"
    ):
        block = block32.astype(np.float64, copy=False)
        n_b = block.shape[0]
        rel_start_frame = total_frames
        total_frames += n_b

        sum_all += float(np.sum(block))
        sum_sq_all += float(np.sum(block * block))
        peak_lin = max(peak_lin, float(np.max(np.abs(block))))

        clip_mask_b = np.abs(block) >= 0.9999
        c_cnt = int(np.sum(clip_mask_b))
        if c_cnt > 0:
            clipped_samples += c_cnt
            rows = np.flatnonzero(np.any(clip_mask_b, axis=1))
            clip_indices.extend((rel_start_frame + int(r)) for r in rows[:5000])

        if ch >= 2:
            l_b, r_b = block[:, 0], block[:, 1]
            sum_l += float(np.sum(l_b))
            sum_r += float(np.sum(r_b))
            sum_l2 += float(np.sum(l_b * l_b))
            sum_r2 += float(np.sum(r_b * r_b))
            sum_lr += float(np.dot(l_b, r_b))
            mid_b = 0.5 * (l_b + r_b)
            side_b = 0.5 * (l_b - r_b)
            sum_mid2 += float(np.sum(mid_b * mid_b))
            sum_side2 += float(np.sum(side_b * side_b))

        lufs_acc.update(block)
        tp_acc.update(block)

        mono = np.mean(block, axis=1)
        if mono.size >= nperseg:
            _, p_b = signal.welch(mono, fs=sr, nperseg=nperseg)
            psd_accum += p_b * n_b
            psd_weight += n_b

        # Rumble curve (125 ms frames)
        low_b, zi_lp = signal.sosfilt(sos_lp, mono, zi=zi_lp)
        low_cat = np.concatenate([rumble_tail, low_b]) if rumble_tail.size else low_b
        n_r_frames = low_cat.size // hop_r
        if n_r_frames > 0:
            r_mat = low_cat[: n_r_frames * hop_r].reshape(n_r_frames, hop_r)
            r_rms = np.sqrt(np.mean(r_mat * r_mat, axis=1))
            rumble_db_list.extend(20.0 * np.log10(np.maximum(r_rms, 1e-12)))
            rumble_tail = low_cat[n_r_frames * hop_r :]
        else:
            rumble_tail = low_cat

        # Click spikes per block
        hp_b, zi_hp = signal.sosfilt(sos_hp, block, axis=0, zi=zi_hp)
        env_b = np.max(np.abs(hp_b), axis=1)
        med_b = float(np.median(env_b))
        sig_b = max(float(np.median(np.abs(env_b - med_b))) * 1.4826, 1e-7)
        z_b = (env_b - med_b) / sig_b
        hits_b = np.flatnonzero((z_b > 10.0) & (env_b > 5e-3))
        if hits_b.size > 0:
            for idx_h in hits_b[:2000]:
                i_h = int(idx_h)
                click_hits.append(
                    (
                        rel_start_frame + i_h,
                        float(np.max(np.abs(block[i_h]))),
                        float(z_b[i_h]),
                    )
                )

        # Foreground curve (50 ms frames)
        if sos_bp is not None and zi_bp is not None:
            fg_b, zi_bp = signal.sosfilt(sos_bp, mono, zi=zi_bp)
        else:
            fg_b = mono
        fg_cat = np.concatenate([fg_tail, fg_b]) if fg_tail.size else fg_b
        n_fg_frames = fg_cat.size // hop_fg
        if n_fg_frames > 0:
            fg_mat = fg_cat[: n_fg_frames * hop_fg].reshape(n_fg_frames, hop_fg)
            fg_rms = np.sqrt(np.mean(fg_mat * fg_mat, axis=1))
            fg_db_list.extend(20.0 * np.log10(np.maximum(fg_rms, 1e-12)))
            fg_tail = fg_cat[n_fg_frames * hop_fg :]
        else:
            fg_tail = fg_cat

        del abs_frame  # unused

    if total_frames == 0:
        return (
            AudioMetrics(duration_s=0.0, sample_rate=sr, channels=ch),
            [],
        )

    duration_s = total_frames / float(sr)
    rms_lin = math.sqrt(sum_sq_all / (total_frames * ch))
    rms_dbfs = _db_from_linear(rms_lin)
    peak_dbfs = _db_from_linear(peak_lin)
    true_peak_dbtp = _db_from_linear(tp_acc.peak_lin)
    crest_factor_db = (
        _finite_or_none(peak_dbfs - rms_dbfs)
        if peak_dbfs is not None and rms_dbfs is not None
        else None
    )
    dc_offset = _finite_or_none(sum_all / (total_frames * ch), digits=6)
    integrated_lufs = lufs_acc.finalize() if rms_lin > 1e-10 else None

    stereo_correlation: float | None = None
    mid_side_ratio_db: float | None = None
    if ch >= 2:
        n_f = float(total_frames)
        var_l = max(0.0, sum_l2 - (sum_l * sum_l) / n_f)
        var_r = max(0.0, sum_r2 - (sum_r * sum_r) / n_f)
        denom = math.sqrt(var_l * var_r)
        if denom > 1e-12:
            cov = sum_lr - (sum_l * sum_r) / n_f
            stereo_correlation = _finite_or_none(max(-1.0, min(1.0, cov / denom)))
        mid_rms = math.sqrt(sum_mid2 / n_f)
        side_rms = math.sqrt(sum_side2 / n_f)
        if mid_rms > 1e-12:
            mid_side_ratio_db = _finite_or_none(20.0 * math.log10(mid_rms / max(side_rms, 1e-12)))

    centroid_hz = rolloff_hz = flatness = rumble_ratio_db = None
    hum_peaks_hz: list[float] = []
    tonal_peaks_hz: list[float] = []
    psd_mean = np.zeros_like(freqs_ref)
    if psd_weight > 0:
        psd_mean = psd_accum / psd_weight
        (
            centroid_hz,
            rolloff_hz,
            flatness,
            rumble_ratio_db,
            hum_peaks_hz,
            tonal_peaks_hz,
        ) = _finalize_spectral_summary(freqs_ref, psd_mean, sr)

    events: list[DistractionEvent] = _build_global_events(
        duration_s, s0, hum_peaks_hz, tonal_peaks_hz
    )

    if rumble_db_list:
        r_arr = np.asarray(rumble_db_list, dtype=np.float64)
        t_r = np.arange(r_arr.size) * (hop_r / sr)
        events.extend(_rumble_events_from_curve(t_r, r_arr, hop_r / sr, duration_s, s0))

    events.extend(_click_events_from_hits(click_hits, sr, duration_s, cutoff_hp, s0))

    if fg_db_list:
        fg_arr = np.asarray(fg_db_list, dtype=np.float64)
        t_fg = np.arange(fg_arr.size) * (hop_fg / sr)
        events.extend(
            _foreground_events_from_curve(
                t_fg,
                fg_arr,
                hop_fg / sr,
                duration_s,
                sr,
                freqs_ref,
                psd_mean,
                sensitivity,
                min_event_s,
                max_event_s,
                freq_range_hz,
                s0,
                (src, s0),
            )
        )

    if clip_indices:
        gap = int(0.1 * sr)
        s_i = e_i = clip_indices[0]
        for i in clip_indices[1:]:
            if i - e_i <= gap:
                e_i = i
            else:
                events.append(
                    DistractionEvent(
                        id="pending",
                        start_s=round(s0 + s_i / sr, 4),
                        end_s=round(s0 + (e_i + 1) / sr, 4),
                        kind="clipping",
                        severity="high",
                        peak_db=0.0,
                        notes="Digital full-scale saturation (|sample| >= 0.9999).",
                    )
                )
                s_i = e_i = i
        events.append(
            DistractionEvent(
                id="pending",
                start_s=round(s0 + s_i / sr, 4),
                end_s=round(s0 + (e_i + 1) / sr, 4),
                kind="clipping",
                severity="high",
                peak_db=0.0,
                notes="Digital full-scale saturation (|sample| >= 0.9999).",
            )
        )

    events.sort(key=lambda ev: (ev.start_s, ev.end_s))
    for idx, ev in enumerate(events, start=1):
        ev.id = f"evt_{idx:03d}"

    metrics = AudioMetrics(
        duration_s=round(duration_s, 4),
        sample_rate=sr,
        channels=ch,
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
        tonal_peaks_hz=tonal_peaks_hz,
    )
    return metrics, events
