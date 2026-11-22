"""Low-RAM streaming psychoacoustic analysis and configurable distraction detection."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyloudnorm as pyln
from scipy import ndimage, signal

from ambient_sound.engines import (
    block_roughness_sample,
    compute_band_metrics_from_psd,
    compute_pleasantness_metrics,
    detect_channel_layout,
    detect_channel_layout_from_stats,
    detect_click_trains_and_crackles,
    detect_spatial_anomalies,
    detect_synthetic_beeps,
    detect_variable_rpm_engines,
    detect_voice_and_dog_barks,
)
from ambient_sound.io_audio import (
    ensure_2d_audio,
    iter_audio_blocks,
    load_audio_slice,
    probe_audio,
)
from ambient_sound.models import AudioMetrics, DistractionEvent, ProcessingRecipe

SensitivityLevel = Literal["low", "medium", "high"]

_SENSITIVITY_PARAMS: dict[str, tuple[float, float]] = {
    "low": (10.0, 5.0),
    "medium": (7.0, 3.8),
    "high": (4.5, 2.6),
}

_SLOW_SENSITIVITY_PARAMS: dict[str, tuple[float, float]] = {
    "low": (7.0, 5.5),
    "medium": (5.0, 4.5),
    "high": (3.5, 3.2),
}

_RUMBLE_SENSITIVITY_PARAMS: dict[str, tuple[float, float]] = {
    "low": (10.5, 4.8),
    "medium": (7.5, 3.8),
    "high": (5.0, 2.8),
}


def event_kind_counts(events: list[DistractionEvent]) -> dict[str, int]:
    """Count distraction events grouped by event kind."""
    counts: dict[str, int] = {}
    for ev in events:
        counts[ev.kind] = counts.get(ev.kind, 0) + 1
    return dict(sorted(counts.items()))


def summarize_events(
    events: list[DistractionEvent],
    event_kinds: list[str] | None = None,
    max_events: int = 50,
    include_clicks: bool | None = None,
) -> tuple[list[DistractionEvent], dict[str, int], int, bool]:
    """Filter and bound events for compact MCP payloads, protecting the macro budget."""
    counts = event_kind_counts(events)
    if event_kinds is not None:
        allowed = {str(k).strip() for k in event_kinds if str(k).strip()}
        filtered = [ev for ev in events if ev.kind in allowed]
    else:
        filtered = list(events)

    total_filtered = len(filtered)
    if max_events <= 0:
        return [], counts, total_filtered, total_filtered > 0

    non_clicks = [ev for ev in filtered if ev.kind != "impulsive_click"]
    clicks = [ev for ev in filtered if ev.kind == "impulsive_click"]
    hide_clicks = (
        event_kinds is None
        and (include_clicks is False or (include_clicks is None and len(clicks) > 25))
    )

    if hide_clicks:
        pool = non_clicks
        if len(pool) <= max_events:
            return pool, counts, total_filtered, len(clicks) > 0
    elif total_filtered <= max_events:
        return filtered, counts, total_filtered, False

    sev_rank = {"high": 0, "medium": 1, "low": 2}
    if len(non_clicks) > max_events:
        ranked = sorted(
            non_clicks, key=lambda ev: (sev_rank.get(ev.severity, 9), ev.start_s, ev.end_s)
        )
        selected = ranked[:max_events]
    else:
        selected = list(non_clicks)
        remaining = max_events - len(selected)
        if remaining > 0 and clicks and not hide_clicks:
            selected.extend(clicks[:remaining])
    selected.sort(key=lambda ev: (ev.start_s, ev.end_s))
    return selected, counts, total_filtered, len(selected) < total_filtered


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

    def short_term_metrics(
        self,
        time_offset_s: float = 0.0,
        max_timeline_points: int = 120,
    ) -> tuple[float | None, dict[str, float] | None, list[dict[str, float]]]:
        """Compute LRA (LU), short-term LUFS summary stats, and downsampled timeline."""
        if len(self._sub_energies) < 30:
            return None, None, []

        subs = np.asarray(self._sub_energies, dtype=np.float64)
        weights = np.array([1.0, 1.0, 1.0, 1.41, 1.41], dtype=np.float64)
        if self.channels <= 5:
            g = weights[: self.channels]
        else:
            g = np.concatenate([weights, np.ones(self.channels - 5, dtype=np.float64)])

        weighted = subs @ g
        csum = np.concatenate([[0.0], np.cumsum(weighted)])
        st_weighted = np.maximum((csum[30:] - csum[:-30]) / 30.0, 1e-24)
        with np.errstate(divide="ignore", invalid="ignore"):
            st_lufs = -0.691 + 10.0 * np.log10(st_weighted)

        valid_mask = np.isfinite(st_lufs) & (st_lufs > -120.0)
        if not np.any(valid_mask):
            return None, None, []

        valid_vals = st_lufs[valid_mask]
        stats = {
            "min_lufs": round(float(np.min(valid_vals)), 4),
            "p10_lufs": round(float(np.percentile(valid_vals, 10)), 4),
            "median_lufs": round(float(np.median(valid_vals)), 4),
            "p95_lufs": round(float(np.percentile(valid_vals, 95)), 4),
            "max_lufs": round(float(np.max(valid_vals)), 4),
        }

        lra: float | None = None
        abs_mask = st_lufs >= -70.0
        if np.any(abs_mask):
            avg_power = float(np.mean(st_weighted[abs_mask]))
            if avg_power > 0.0:
                rel_thresh = -0.691 + 10.0 * math.log10(avg_power) - 20.0
                gated = st_lufs[(st_lufs >= rel_thresh) & abs_mask]
                if gated.size >= 2:
                    lra = _finite_or_none(
                        max(0.0, float(np.percentile(gated, 95) - np.percentile(gated, 10))),
                        digits=2,
                    )

        n_pts = st_lufs.size
        step = max(1, math.ceil(n_pts / max(1, max_timeline_points)))
        timeline: list[dict[str, float]] = []
        for idx in range(0, n_pts, step):
            val = float(st_lufs[idx])
            if math.isfinite(val) and val > -120.0:
                timeline.append(
                    {
                        "time_s": round(time_offset_s + idx * 0.1, 2),
                        "lufs": round(val, 2),
                    }
                )
        return lra, stats, timeline


def _bilateral_prominence_db(
    freqs: np.ndarray,
    psd_db: np.ndarray,
    f_center: float,
    peak_val: float,
    min_left_hz: float,
    max_right_hz: float,
    inner_gap_hz: float = 6.0,
    outer_span_hz: float = 22.0,
) -> float | None:
    """Measure how much a spectral peak exceeds BOTH lower and upper sidebands."""
    left_mask = (freqs >= max(min_left_hz, f_center - outer_span_hz)) & (
        freqs <= f_center - inner_gap_hz
    )
    right_mask = (freqs >= f_center + inner_gap_hz) & (
        freqs <= min(max_right_hz, f_center + outer_span_hz)
    )
    if not np.any(left_mask) or not np.any(right_mask):
        return None
    left_med = float(np.median(psd_db[left_mask]))
    right_med = float(np.median(psd_db[right_mask]))
    return min(peak_val - left_med, peak_val - right_med)


def _detect_hum_peaks(
    freqs: np.ndarray,
    psd: np.ndarray,
    sample_rate: int,
    freqs_band: np.ndarray | None = None,
    segment_psds_band: list[np.ndarray] | None = None,
) -> list[float]:
    """Return 50 Hz or 60 Hz harmonic frequencies exhibiting bilateral narrowband prominence."""
    nyquist = sample_rate * 0.5
    use_median = (
        segment_psds_band is not None
        and len(segment_psds_band) >= 3
        and freqs_band is not None
        and freqs_band.size >= 16
    )
    if use_median:
        assert freqs_band is not None and segment_psds_band is not None
        work_freqs = freqs_band
        work_psd = np.median(np.asarray(segment_psds_band, dtype=np.float64), axis=0)
    else:
        work_freqs = freqs
        work_psd = psd

    if work_freqs.size < 16:
        return []

    psd_db = 10.0 * np.log10(np.maximum(work_psd, 1e-24))
    detected: list[float] = []

    for base in (50.0, 60.0):
        series_peaks: list[tuple[float, float]] = []
        for harmonic in range(1, 5):
            f_target = base * harmonic
            if f_target >= nyquist * 0.9:
                break
            peak_mask = (work_freqs >= f_target - 2.5) & (work_freqs <= f_target + 2.5)
            if not np.any(peak_mask):
                continue
            cand_idx = np.flatnonzero(peak_mask)
            best_rel = int(np.argmax(psd_db[cand_idx]))
            peak_idx = int(cand_idx[best_rel])
            peak_val = float(psd_db[peak_idx])
            f_center = float(work_freqs[peak_idx])
            prominence = _bilateral_prominence_db(
                work_freqs,
                psd_db,
                f_center,
                peak_val,
                10.0,
                nyquist,
                inner_gap_hz=5.5,
                outer_span_hz=18.0,
            )
            if prominence is not None and prominence > 7.5 and peak_val > -112.0:
                series_peaks.append((f_target, prominence))
        if series_peaks and (
            any(f == base and p > 8.5 for f, p in series_peaks) or len(series_peaks) >= 2
        ):
            detected.extend(f for f, _ in series_peaks)

    return sorted({round(f, 1) for f in detected})


def _detect_tonal_peaks(
    freqs: np.ndarray,
    psd: np.ndarray,
    sample_rate: int,
    exclude_freqs: list[float],
    freqs_band: np.ndarray | None = None,
    segment_psds_band: list[np.ndarray] | None = None,
) -> list[float]:
    """Detect persistent narrowband tonal peaks (30-8000 Hz) via median-PSD & constant-Q sides."""
    nyquist = sample_rate * 0.5
    max_f = min(8000.0, nyquist * 0.85)
    if freqs.size < 32 or max_f <= 40.0:
        return []

    check_persistence = (
        segment_psds_band is not None
        and len(segment_psds_band) >= 3
        and freqs_band is not None
        and freqs_band.size >= 32
    )
    if check_persistence:
        assert freqs_band is not None and segment_psds_band is not None
        seg_mat = np.asarray(segment_psds_band, dtype=np.float64)
        work_freqs = freqs_band
        work_psd = np.median(seg_mat, axis=0)
        seg_dbs = 10.0 * np.log10(np.maximum(seg_mat, 1e-24))
    else:
        work_freqs = freqs
        work_psd = psd
        seg_dbs = None

    psd_db = 10.0 * np.log10(np.maximum(work_psd, 1e-24))
    band_indices = np.flatnonzero((work_freqs >= 30.0) & (work_freqs <= max_f))
    candidates: list[tuple[float, float]] = []

    for idx in band_indices:
        if idx <= 0 or idx >= work_freqs.size - 1:
            continue
        if psd_db[idx] < psd_db[idx - 1] or psd_db[idx] < psd_db[idx + 1]:
            continue
        f_cand = float(work_freqs[idx])
        if any(abs(f_cand - ex) <= 4.5 for ex in exclude_freqs):
            continue
        inner = max(6.0, 0.03 * f_cand)
        outer = max(22.0, 0.12 * f_cand)
        peak_val = float(psd_db[idx])
        prom = _bilateral_prominence_db(
            work_freqs,
            psd_db,
            f_cand,
            peak_val,
            12.0,
            nyquist,
            inner_gap_hz=inner,
            outer_span_hz=outer,
        )
        if prom is None or prom <= 7.0 or peak_val <= -112.0:
            continue

        if check_persistence and seg_dbs is not None and freqs_band is not None:
            half_p = max(3.5, 0.01 * f_cand)
            p_mask = (freqs_band >= f_cand - half_p) & (freqs_band <= f_cand + half_p)
            l_mask = (freqs_band >= max(12.0, f_cand - outer)) & (freqs_band <= f_cand - inner)
            r_mask = (freqs_band >= f_cand + inner) & (freqs_band <= min(nyquist, f_cand + outer))
            if np.any(p_mask) and np.any(l_mask) and np.any(r_mask):
                pks = np.max(seg_dbs[:, p_mask], axis=1)
                l_meds = np.median(seg_dbs[:, l_mask], axis=1)
                r_meds = np.median(seg_dbs[:, r_mask], axis=1)
                seg_proms = np.minimum(pks - l_meds, pks - r_meds)
                if float(np.mean(seg_proms >= 6.0)) < 0.60:
                    continue

        candidates.append((f_cand, prom))

    candidates.sort(key=lambda pair: pair[1], reverse=True)
    selected: list[float] = []
    for f_cand, _ in candidates:
        min_spacing = max(6.0, 0.04 * f_cand)
        if any(abs(f_cand - s) < min_spacing for s in selected):
            continue
        selected.append(round(f_cand, 1))
        if len(selected) >= 12:
            break
    return sorted(selected)


def _detect_segment_local_tones(
    freqs_band: np.ndarray,
    segment_psds_band: list[np.ndarray],
    segment_bounds_s: list[tuple[float, float]],
    sample_rate: int,
    global_exclude: list[float],
    freq_range_hz: tuple[float, float] | None = None,
) -> list[DistractionEvent]:
    """Detect temporary localized narrowband tones active in subsets of a long recording."""
    if (
        len(segment_psds_band) < 3
        or len(segment_bounds_s) != len(segment_psds_band)
        or freqs_band.size < 32
    ):
        return []

    nyquist = sample_rate * 0.5
    f_min, f_max = 40.0, min(4000.0, nyquist * 0.85)
    if freq_range_hz is not None:
        f_min = max(f_min, float(freq_range_hz[0]) - 10.0)
        f_max = min(f_max, float(freq_range_hz[1]) + 10.0)
    if f_max <= f_min:
        return []

    seg_mat = np.asarray(segment_psds_band, dtype=np.float64)
    med_db = 10.0 * np.log10(np.maximum(np.median(seg_mat, axis=0), 1e-24))
    band_indices = np.flatnonzero((freqs_band >= f_min) & (freqs_band <= f_max))
    raw_hits: list[tuple[int, float, float]] = []

    for seg_i in range(len(segment_psds_band)):
        seg_db = 10.0 * np.log10(np.maximum(seg_mat[seg_i], 1e-24))
        cands: list[tuple[float, float]] = []
        for idx in band_indices:
            if idx <= 0 or idx >= freqs_band.size - 1:
                continue
            if seg_db[idx] < seg_db[idx - 1] or seg_db[idx] < seg_db[idx + 1]:
                continue
            if float(seg_db[idx] - med_db[idx]) < 12.0:
                continue
            f_cand = float(freqs_band[idx])
            if any(abs(f_cand - ex) <= 6.0 for ex in global_exclude):
                continue
            inner = max(6.0, 0.03 * f_cand)
            outer = max(22.0, 0.12 * f_cand)
            prom = _bilateral_prominence_db(
                freqs_band,
                seg_db,
                f_cand,
                float(seg_db[idx]),
                12.0,
                nyquist,
                inner_gap_hz=inner,
                outer_span_hz=outer,
            )
            if prom is not None and prom >= 12.0 and float(seg_db[idx]) > -108.0:
                cands.append((f_cand, prom))
        cands.sort(key=lambda pair: pair[1], reverse=True)
        picked: list[tuple[float, float]] = []
        for f_cand, prom in cands:
            min_sp = max(6.0, 0.04 * f_cand)
            if any(abs(f_cand - pf) < min_sp for pf, _ in picked):
                continue
            picked.append((f_cand, prom))
            raw_hits.append((seg_i, round(f_cand, 1), prom))
            if len(picked) >= 2:
                break

    if not raw_hits:
        return []

    raw_hits.sort(key=lambda item: (item[0], item[1]))
    grouped: list[list[tuple[int, float, float]]] = []
    for seg_i, f_hz, prom in raw_hits:
        matched = False
        for grp in grouped:
            last_seg, last_f, _ = grp[-1]
            if seg_i - last_seg <= 1 and abs(f_hz - last_f) <= max(6.0, 0.03 * last_f):
                grp.append((seg_i, f_hz, prom))
                matched = True
                break
        if not matched:
            grouped.append([(seg_i, f_hz, prom)])

    total_span = max(1e-6, segment_bounds_s[-1][1] - segment_bounds_s[0][0])
    events: list[DistractionEvent] = []
    for grp in grouped:
        s_t = segment_bounds_s[grp[0][0]][0]
        e_t = segment_bounds_s[grp[-1][0]][1]
        if (e_t - s_t) >= 0.70 * total_span:
            continue
        f_med = round(float(np.median([g[1] for g in grp])), 1)
        max_prom = max(g[2] for g in grp)
        events.append(
            DistractionEvent(
                id="pending",
                start_s=round(s_t, 3),
                end_s=round(e_t, 3),
                kind="machine_drone",
                severity="high" if max_prom >= 18.0 else "medium",
                freq_band_hz=(round(max(10.0, f_med - 5.0), 1), round(f_med + 5.0, 1)),
                peak_db=None,
                notes=(
                    f"Temporary localized tonal drone at {f_med:.1f} Hz "
                    f"(+{max_prom:.1f} dB in block)."
                ),
            )
        )
    return events


def _rumble_events_from_curve(
    times: np.ndarray,
    rms_db: np.ndarray,
    win_dur_s: float,
    duration_s: float,
    time_offset_s: float,
    sensitivity: SensitivityLevel = "medium",
    min_event_s: float = 0.12,
    max_event_s: float = 45.0,
    ambient_bed_dbfs: float | None = None,
) -> list[DistractionEvent]:
    if rms_db.size < 4:
        return []
    excess_req, mad_mult = _RUMBLE_SENSITIVITY_PARAMS.get(
        sensitivity, _RUMBLE_SENSITIVITY_PARAMS["medium"]
    )
    med = float(np.median(rms_db))
    mad = float(np.median(np.abs(rms_db - med))) * 1.4826
    threshold = max(med + excess_req, med + mad_mult * max(mad, 1.0))
    floor_db = max(-92.0, float(ambient_bed_dbfs) - 22.0) if ambient_bed_dbfs is not None else -92.0
    mask = (rms_db > threshold) & (rms_db > floor_db)
    raw_spans = _merge_intervals(mask, times, win_dur_s, max_gap_s=0.60)

    # Extend core spans outward onto smoothed curve until rumble drops within +1.5 dB of floor
    hop_s = float(times[1] - times[0]) if times.size > 1 else win_dur_s
    k_sm = max(1, int(round(0.8 / max(hop_s, 1e-3))))
    sm = (
        ndimage.uniform_filter1d(rms_db, size=k_sm, mode="nearest")
        if rms_db.size >= max(4, k_sm) and k_sm > 1
        else rms_db
    )
    med_sm = float(np.median(sm))
    ext_thr = med_sm + 1.5
    extended: list[tuple[float, float]] = []
    for s0, e0 in raw_spans:
        i0 = max(0, min(sm.size - 1, int(np.searchsorted(times, s0, side="left"))))
        i1 = max(i0, min(sm.size - 1, int(np.searchsorted(times, e0 - win_dur_s, side="left"))))
        while (
            i0 > 0
            and sm[i0 - 1] > ext_thr
            and (float(times[i1]) + win_dur_s - float(times[i0 - 1])) <= max_event_s
        ):
            i0 -= 1
        while (
            i1 < sm.size - 1
            and sm[i1 + 1] > ext_thr
            and (float(times[i1 + 1]) + win_dur_s - float(times[i0])) <= max_event_s
        ):
            i1 += 1
        ns = float(times[i0])
        ne = float(times[i1]) + win_dur_s
        if extended and ns <= extended[-1][1] + 0.40:
            extended[-1] = (extended[-1][0], max(extended[-1][1], ne))
        else:
            extended.append((ns, ne))

    min_dur = max(0.2, float(min_event_s))
    events: list[DistractionEvent] = []
    for s, e in extended:
        e_clamped = min(e, duration_s)
        dur = e_clamped - s
        if dur < min_dur or dur > max_event_s:
            continue
        seg_mask = (times >= s) & (times <= e_clamped)
        peak_db = float(np.max(rms_db[seg_mask])) if np.any(seg_mask) else threshold
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
    min_event_s: float = 0.12,
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
        if min_event_s > 0.15 and (e_s - s_s) < min_event_s:
            continue
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


def _compute_texture_tick_rate(
    click_events: list[DistractionEvent],
    duration_s: float,
    ref_bed_dbfs: float | None,
) -> float | None:
    """Detect high-rate organic texture ticks (e.g. insect/hive clicks >= 30/min)."""
    if duration_s < 10.0 or not click_events:
        return None
    rate_per_min = len(click_events) / (duration_s / 60.0)
    if rate_per_min < 30.0:
        return None
    peaks = [ev.peak_db for ev in click_events if ev.peak_db is not None]
    if peaks and ref_bed_dbfs is not None and float(np.median(peaks)) >= ref_bed_dbfs + 14.0:
        return None
    return _finite_or_none(rate_per_min, digits=2)


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
    fast_mask = (rms_db > threshold) & (rms_db > -92.0)
    fast_spans = _merge_intervals(fast_mask, times, win_dur_s, max_gap_s=0.50)

    slow_spans: list[tuple[float, float]] = []
    hop_s = float(times[1] - times[0]) if times.size > 1 else win_dur_s
    k_1s = max(1, int(round(1.0 / max(hop_s, 1e-3))))
    if rms_db.size >= max(8, k_1s):
        sm = ndimage.uniform_filter1d(rms_db, size=k_1s, mode="nearest")
        med_sm = float(np.median(sm))
        mad_sm = float(np.median(np.abs(sm - med_sm))) * 1.4826
        s_ex, s_mad = _SLOW_SENSITIVITY_PARAMS.get(
            sensitivity, _SLOW_SENSITIVITY_PARAMS["medium"]
        )
        thr_sm = max(med_sm + s_ex, med_sm + s_mad * max(mad_sm, 0.5))
        # Extend detected passes outward to within +1.5 dB of local floor (catches approach/tail)
        ext_thr = med_sm + 1.5
        raw_slow = _merge_intervals(
            (sm > thr_sm) & (sm > -92.0), times, win_dur_s, max_gap_s=1.5
        )
        for s0, e0 in raw_slow:
            i0 = max(0, min(sm.size - 1, int(np.searchsorted(times, s0, side="left"))))
            i1 = max(i0, min(sm.size - 1, int(np.searchsorted(times, e0 - win_dur_s, side="left"))))
            while (
                i0 > 0
                and sm[i0 - 1] > ext_thr
                and (float(times[i1]) + win_dur_s - float(times[i0 - 1])) <= max_event_s
            ):
                i0 -= 1
            while (
                i1 < sm.size - 1
                and sm[i1 + 1] > ext_thr
                and (float(times[i1 + 1]) + win_dur_s - float(times[i0])) <= max_event_s
            ):
                i1 += 1
            ns = float(times[i0])
            ne = float(times[i1]) + win_dur_s
            if (ne - ns) >= max(2.5, min_event_s):
                slow_spans.append((ns, ne))

    combined = sorted([*fast_spans, *slow_spans], key=lambda pair: (pair[0], pair[1]))
    spans: list[tuple[float, float]] = []
    for cs, ce in combined:
        if spans and cs - spans[-1][1] <= 0.50:
            spans[-1] = (spans[-1][0], max(spans[-1][1], ce))
        else:
            spans.append((cs, ce))

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
    freq_range_hz: tuple[float, float] | None = None,
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
    for freq_hz in tonal_peaks_hz:
        if freq_range_hz is not None and (
            freq_hz < float(freq_range_hz[0]) - 10.0 or freq_hz > float(freq_range_hz[1]) + 10.0
        ):
            continue
        events.append(
            DistractionEvent(
                id="pending",
                start_s=s_abs,
                end_s=e_abs,
                kind="machine_drone",
                severity="medium",
                freq_band_hz=(round(max(10.0, freq_hz - 5.0), 1), round(freq_hz + 5.0, 1)),
                peak_db=None,
                notes=f"Detected persistent tonal drone peak at {freq_hz} Hz.",
            )
        )
    return events


def _finalize_spectral_summary(
    freqs: np.ndarray,
    psd: np.ndarray,
    sample_rate: int,
    freqs_band: np.ndarray | None = None,
    segment_psds_band: list[np.ndarray] | None = None,
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

    hum_peaks = _detect_hum_peaks(
        freqs,
        psd,
        sample_rate,
        freqs_band=freqs_band,
        segment_psds_band=segment_psds_band,
    )
    tonal_peaks = _detect_tonal_peaks(
        freqs,
        psd,
        sample_rate,
        hum_peaks,
        freqs_band=freqs_band,
        segment_psds_band=segment_psds_band,
    )
    return centroid_hz, rolloff_hz, flatness, rumble_ratio_db, hum_peaks, tonal_peaks


def _floor_stats_from_chunk_rms(
    chunk_rms_db: list[float] | np.ndarray,
) -> tuple[float | None, float | None]:
    arr = np.asarray(chunk_rms_db, dtype=np.float64)
    if arr.size == 0:
        return None, None
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return None, None
    return (
        _finite_or_none(float(np.percentile(valid, 10)), digits=2),
        _finite_or_none(float(np.percentile(valid, 50)), digits=2),
    )


def analyze_audio(
    audio: np.ndarray,
    sample_rate: int,
    sensitivity: SensitivityLevel = "medium",
    min_event_s: float = 0.12,
    max_event_s: float = 45.0,
    freq_range_hz: tuple[float, float] | None = None,
    time_offset_s: float = 0.0,
    event_kinds: list[str] | None = None,
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

    allowed_kinds = (
        {str(k).strip() for k in event_kinds if str(k).strip()} if event_kinds is not None else None
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

    floor_win = max(1, int(0.1 * sample_rate))
    n_floor_bins = num_samples // floor_win
    if n_floor_bins >= 1:
        sq_bins = np.mean(
            (arr[: n_floor_bins * floor_win] ** 2).reshape(n_floor_bins, floor_win, num_channels),
            axis=(1, 2),
        )
        floor_db_arr = 10.0 * np.log10(np.maximum(sq_bins, 1e-24))
        noise_floor_dbfs, ambient_bed_dbfs = _floor_stats_from_chunk_rms(floor_db_arr)
    else:
        noise_floor_dbfs = rms_dbfs
        ambient_bed_dbfs = rms_dbfs

    high_freq_floor_dbfs: float | None = None
    if sample_rate > 14000 and n_floor_bins >= 1 and num_samples >= 64:
        sos_hf_f = signal.butter(
            4, min(6000.0, sample_rate * 0.35), btype="highpass", fs=sample_rate, output="sos"
        )
        hf_mono = signal.sosfiltfilt(sos_hf_f, mono)
        hf_sq = np.mean(
            (hf_mono[: n_floor_bins * floor_win] ** 2).reshape(n_floor_bins, floor_win),
            axis=1,
        )
        high_freq_floor_dbfs, _ = _floor_stats_from_chunk_rms(
            10.0 * np.log10(np.maximum(hf_sq, 1e-24))
        )

    integrated_lufs: float | None = None
    loudness_range_lu: float | None = None
    short_term_lufs_stats: dict[str, float] | None = None
    short_term_lufs_timeline: list[dict[str, float]] = []
    if duration_s >= 0.45 and rms_lin > 1e-10:
        lufs_acc = StreamingLUFS(sample_rate, num_channels)
        lufs_acc.update(arr)
        integrated_lufs = lufs_acc.finalize()
        loudness_range_lu, short_term_lufs_stats, short_term_lufs_timeline = (
            lufs_acc.short_term_metrics(time_offset_s=time_offset_s)
        )

    stereo_correlation: float | None = None
    mid_side_ratio_db: float | None = None
    channel_balance_db: float | None = None
    if num_channels >= 2:
        l_ch, r_ch = arr[:, 0], arr[:, 1]
        l_zm, r_zm = l_ch - np.mean(l_ch), r_ch - np.mean(r_ch)
        denom = float(np.linalg.norm(l_zm) * np.linalg.norm(r_zm))
        if denom > 1e-12:
            corr = max(-1.0, min(1.0, float(np.dot(l_zm, r_zm) / denom)))
            stereo_correlation = _finite_or_none(corr)
        rms_l = float(np.sqrt(np.mean(l_ch * l_ch)))
        rms_r = float(np.sqrt(np.mean(r_ch * r_ch)))
        if rms_l > 1e-12 or rms_r > 1e-12:
            channel_balance_db = _finite_or_none(
                20.0 * math.log10(max(rms_l, 1e-12) / max(rms_r, 1e-12)), digits=3
            )
        mid = 0.5 * (l_ch + r_ch)
        side = 0.5 * (l_ch - r_ch)
        mid_rms = float(np.sqrt(np.mean(mid * mid)))
        side_rms = float(np.sqrt(np.mean(side * side)))
        if mid_rms > 1e-12:
            mid_side_ratio_db = _finite_or_none(20.0 * math.log10(mid_rms / max(side_rms, 1e-12)))

    layout_info = detect_channel_layout(arr, sample_rate)

    centroid_hz = rolloff_hz = flatness = rumble_ratio_db = None
    hum_peaks_hz: list[float] = []
    tonal_peaks_hz: list[float] = []
    freqs_ref = np.zeros(0, dtype=np.float64)
    psd_ref = np.zeros(0, dtype=np.float64)
    segment_psds_full: list[np.ndarray] = []
    seg_freqs_full: np.ndarray | None = None

    if num_samples >= 64 and float(np.max(np.abs(mono))) > 1e-12:
        nperseg = min(num_samples, 16384)
        freqs_ref, psd_ref = signal.welch(mono, fs=sample_rate, nperseg=nperseg)
        freqs_band = freqs_ref[freqs_ref <= 8500.0]
        segment_psds_band: list[np.ndarray] = []
        seg_freqs_band: np.ndarray | None = None
        if duration_s >= 2.0:
            n_segs = min(12, max(4, int(duration_s // 5.0)))
            seg_len = num_samples // n_segs
            seg_nperseg = min(nperseg, seg_len)
            if seg_nperseg >= 1024:
                for seg_idx in range(n_segs):
                    chunk = mono[seg_idx * seg_len : (seg_idx + 1) * seg_len]
                    sf_i, p_seg = signal.welch(chunk, fs=sample_rate, nperseg=seg_nperseg)
                    seg_b_mask = sf_i <= 8500.0
                    seg_freqs_full = sf_i
                    seg_freqs_band = sf_i[seg_b_mask]
                    segment_psds_band.append(p_seg[seg_b_mask])
                    segment_psds_full.append(p_seg)
        (
            centroid_hz,
            rolloff_hz,
            flatness,
            rumble_ratio_db,
            hum_peaks_hz,
            tonal_peaks_hz,
        ) = _finalize_spectral_summary(
            freqs_ref,
            psd_ref,
            sample_rate,
            freqs_band=seg_freqs_band if seg_freqs_band is not None else freqs_band,
            segment_psds_band=segment_psds_band,
        )

    rough_val = block_roughness_sample(mono, sample_rate)
    pleasant = compute_pleasantness_metrics(
        freqs_ref,
        psd_ref,
        roughness_values=[rough_val] if rough_val is not None else [],
        rumble_ratio_db=rumble_ratio_db,
        num_drones=len(hum_peaks_hz) + len(tonal_peaks_hz),
    )
    band_metrics = compute_band_metrics_from_psd(
        freqs_ref,
        psd_ref,
        custom_range=freq_range_hz,
        seg_psds=segment_psds_full or None,
        seg_freqs=seg_freqs_full if seg_freqs_full is not None else freqs_ref,
    )

    events: list[DistractionEvent] = _build_global_events(
        duration_s,
        time_offset_s,
        hum_peaks_hz,
        tonal_peaks_hz,
        freq_range_hz=freq_range_hz,
    )

    skip_rumble = freq_range_hz is not None and float(freq_range_hz[0]) >= 80.0
    skip_clicks = freq_range_hz is not None and float(freq_range_hz[1]) <= 1500.0

    # Rumble detector
    if (
        not skip_rumble
        and (allowed_kinds is None or "low_freq_rumble" in allowed_kinds)
        and sample_rate > 220
        and mono.size >= int(sample_rate * 0.5)
    ):
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
                _rumble_events_from_curve(
                    t_r,
                    db_r,
                    win_r / sample_rate,
                    duration_s,
                    time_offset_s,
                    sensitivity=sensitivity,
                    min_event_s=min_event_s,
                    max_event_s=max_event_s,
                    ambient_bed_dbfs=ambient_bed_dbfs,
                )
            )

    # Click detector
    texture_tick_rate_per_min: float | None = None
    if (
        not skip_clicks
        and (allowed_kinds is None or "impulsive_click" in allowed_kinds)
        and num_samples >= 64
    ):
        cutoff = min(1500.0, sample_rate * 0.35)
        sos_hp = signal.butter(4, cutoff, btype="highpass", fs=sample_rate, output="sos")
        hp = signal.sosfiltfilt(sos_hp, arr, axis=0)
        env = np.max(np.abs(hp), axis=1)
        med = float(np.median(env))
        sigma = max(float(np.median(np.abs(env - med))) * 1.4826, 1e-7)
        z = (env - med) / sigma
        hit_idx = np.flatnonzero((z > 10.0) & (env > 1e-4))
        hits = [
            (int(i), float(np.max(np.abs(arr[i]))), float(z[i]))
            for i in hit_idx
        ]
        click_events = _click_events_from_hits(
            hits,
            sample_rate,
            duration_s,
            cutoff,
            time_offset_s,
            min_event_s=min_event_s,
        )
        events.extend(click_events)
        texture_tick_rate_per_min = _compute_texture_tick_rate(
            click_events,
            duration_s,
            ambient_bed_dbfs if ambient_bed_dbfs is not None else rms_dbfs,
        )

    # Foreground transient & slow_pass detector
    if allowed_kinds is None or bool({"slow_pass", "transient_distraction"} & allowed_kinds):
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

    # Outdoor nature / distraction classifiers (gated cleanly by allowed_kinds & freq_range_hz)
    f_lo_req = float(freq_range_hz[0]) if freq_range_hz is not None else 0.0
    f_hi_req = float(freq_range_hz[1]) if freq_range_hz is not None else float(sample_rate * 0.5)
    if (allowed_kinds is None or "synthetic_beep" in allowed_kinds) and f_hi_req >= 1400.0:
        events.extend(detect_synthetic_beeps(mono, sample_rate, time_offset_s, sensitivity))
    if (
        allowed_kinds is None or bool({"click_train", "crackle_burst"} & allowed_kinds)
    ) and f_hi_req >= 1800.0:
        events.extend(
            detect_click_trains_and_crackles(arr, sample_rate, time_offset_s, sensitivity)
        )
    if (
        allowed_kinds is None or bool({"dog_bark", "human_voice"} & allowed_kinds)
    ) and f_hi_req >= 220.0 and f_lo_req <= 3400.0:
        events.extend(detect_voice_and_dog_barks(mono, sample_rate, time_offset_s, sensitivity))
    if (allowed_kinds is None or "variable_rpm_engine" in allowed_kinds) and f_lo_req <= 1200.0:
        events.extend(detect_variable_rpm_engines(mono, sample_rate, time_offset_s, sensitivity))
    if (
        num_channels >= 2
        and (allowed_kinds is None or bool({"wind_noise", "channel_biased_noise"} & allowed_kinds))
        and f_lo_req <= 1500.0
    ):
        events.extend(detect_spatial_anomalies(arr, sample_rate, time_offset_s, sensitivity))

    # Clipping detector
    clipped_mask = np.any(np.abs(arr) >= 0.9999, axis=1)
    clipped_samples = int(np.sum(np.abs(arr) >= 0.9999))
    if clipped_samples > 0 and (allowed_kinds is None or "clipping" in allowed_kinds):
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

    if allowed_kinds is not None:
        events = [ev for ev in events if ev.kind in allowed_kinds]

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
        loudness_range_lu=loudness_range_lu,
        noise_floor_dbfs=noise_floor_dbfs,
        ambient_bed_dbfs=ambient_bed_dbfs,
        high_freq_floor_dbfs=high_freq_floor_dbfs,
        texture_tick_rate_per_min=texture_tick_rate_per_min,
        short_term_lufs_stats=short_term_lufs_stats,
        short_term_lufs_timeline=short_term_lufs_timeline,
        sharpness_acum=pleasant["sharpness_acum"],
        roughness_index=pleasant["roughness_index"],
        ndsi=pleasant["ndsi"],
        spectral_slope_db_per_oct=pleasant["spectral_slope_db_per_oct"],
        channel_balance_db=channel_balance_db,
        stereo_layout_hint=layout_info["layout"],
        stereo_layout_confidence=layout_info["confidence"],
        pleasantness_score=pleasant["pleasantness_score"],
        band_metrics=band_metrics,
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
    event_kinds: list[str] | None = None,
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

    allowed_kinds = (
        {str(k).strip() for k in event_kinds if str(k).strip()} if event_kinds is not None else None
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
                event_kinds=event_kinds,
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
    max_l_lin = max_r_lin = 0.0
    e1_low = e1_mid = e2_low = e2_mid = 0.0
    csd_lf_real = csd_lf_imag = lf_p1 = lf_p2 = 0.0

    nperseg = 16384
    freqs_ref = np.fft.rfftfreq(nperseg, d=1.0 / sr)
    psd_accum = np.zeros_like(freqs_ref, dtype=np.float64)
    psd_weight = 0
    band_mask = freqs_ref <= 8500.0
    freqs_band = freqs_ref[band_mask]
    segment_psds_band: list[np.ndarray] = []
    segment_psds_full: list[np.ndarray] = []
    segment_bounds_s: list[tuple[float, float]] = []
    skip_rumble = freq_range_hz is not None and float(freq_range_hz[0]) >= 80.0
    skip_clicks = freq_range_hz is not None and float(freq_range_hz[1]) <= 1500.0

    f_lo_req = float(freq_range_hz[0]) if freq_range_hz is not None else 0.0
    f_hi_req = float(freq_range_hz[1]) if freq_range_hz is not None else float(sr * 0.5)
    run_beeps = (allowed_kinds is None or "synthetic_beep" in allowed_kinds) and f_hi_req >= 1400.0
    run_clicks_crackles = (
        allowed_kinds is None or bool({"click_train", "crackle_burst"} & allowed_kinds)
    ) and f_hi_req >= 1800.0
    run_voice_bark = (
        allowed_kinds is None or bool({"dog_bark", "human_voice"} & allowed_kinds)
    ) and f_hi_req >= 220.0 and f_lo_req <= 3400.0
    run_engines = (
        allowed_kinds is None or "variable_rpm_engine" in allowed_kinds
    ) and f_lo_req <= 1200.0
    run_spatial = (
        ch >= 2
        and (allowed_kinds is None or bool({"wind_noise", "channel_biased_noise"} & allowed_kinds))
        and f_lo_req <= 1500.0
    )
    roughness_samples: list[float] = []
    outdoor_events: list[DistractionEvent] = []

    floor_hop = max(1, int(0.1 * sr))
    floor_tail = np.zeros(0, dtype=np.float64)
    floor_db_list: list[float] = []

    sos_hf_floor = (
        signal.butter(4, min(6000.0, sr * 0.35), btype="highpass", fs=sr, output="sos")
        if sr > 14000
        else None
    )
    zi_hf_floor = (
        np.zeros((sos_hf_floor.shape[0], 2), dtype=np.float64)
        if sos_hf_floor is not None
        else None
    )
    hf_floor_tail = np.zeros(0, dtype=np.float64)
    hf_floor_db_list: list[float] = []

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

        sq_ch_mean = np.mean(block * block, axis=1)
        f_cat = np.concatenate([floor_tail, sq_ch_mean]) if floor_tail.size else sq_ch_mean
        n_floor = f_cat.size // floor_hop
        if n_floor > 0:
            f_mat = f_cat[: n_floor * floor_hop].reshape(n_floor, floor_hop)
            f_db = 10.0 * np.log10(np.maximum(np.mean(f_mat, axis=1), 1e-24))
            floor_db_list.extend(f_db)
            floor_tail = f_cat[n_floor * floor_hop :]
        else:
            floor_tail = f_cat

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
            max_l_lin = max(max_l_lin, float(np.max(np.abs(l_b))))
            max_r_lin = max(max_r_lin, float(np.max(np.abs(r_b))))
            mid_b = 0.5 * (l_b + r_b)
            side_b = 0.5 * (l_b - r_b)
            sum_mid2 += float(np.sum(mid_b * mid_b))
            sum_side2 += float(np.sum(side_b * side_b))
            if sr >= 1200 and n_b >= 512:
                nps_c = min(4096, n_b)
                f_c, p1_c = signal.welch(l_b, fs=sr, nperseg=nps_c)
                _, p2_c = signal.welch(r_b, fs=sr, nperseg=nps_c)
                _, p12_c = signal.csd(l_b, r_b, fs=sr, nperseg=nps_c)
                lm_c = (f_c >= 30.0) & (f_c <= 250.0)
                mm_c = (f_c >= 500.0) & (f_c <= min(4000.0, sr * 0.45))
                if np.any(lm_c) and np.any(mm_c):
                    e1_low += float(np.mean(p1_c[lm_c])) * n_b
                    e1_mid += float(np.mean(p1_c[mm_c])) * n_b
                    e2_low += float(np.mean(p2_c[lm_c])) * n_b
                    e2_mid += float(np.mean(p2_c[mm_c])) * n_b
                lf_c = (f_c >= 30.0) & (f_c <= 200.0)
                if np.any(lf_c):
                    csd_lf_real += float(np.sum(p12_c[lf_c].real)) * n_b
                    csd_lf_imag += float(np.sum(p12_c[lf_c].imag)) * n_b
                    lf_p1 += float(np.sum(p1_c[lf_c])) * n_b
                    lf_p2 += float(np.sum(p2_c[lf_c])) * n_b

        lufs_acc.update(block)
        tp_acc.update(block)

        mono = np.mean(block, axis=1)
        blk_offset_s = s0 + rel_start_frame / float(sr)
        r_b_sample = block_roughness_sample(mono, sr)
        if r_b_sample is not None:
            roughness_samples.append(r_b_sample)
        if run_beeps:
            outdoor_events.extend(detect_synthetic_beeps(mono, sr, blk_offset_s, sensitivity))
        if run_clicks_crackles:
            outdoor_events.extend(
                detect_click_trains_and_crackles(block, sr, blk_offset_s, sensitivity)
            )
        if run_voice_bark:
            outdoor_events.extend(detect_voice_and_dog_barks(mono, sr, blk_offset_s, sensitivity))
        if run_engines:
            outdoor_events.extend(detect_variable_rpm_engines(mono, sr, blk_offset_s, sensitivity))
        if run_spatial:
            outdoor_events.extend(detect_spatial_anomalies(block, sr, blk_offset_s, sensitivity))
        if sos_hf_floor is not None and zi_hf_floor is not None:
            hf_b, zi_hf_floor = signal.sosfilt(sos_hf_floor, mono, zi=zi_hf_floor)
            hf_sq = hf_b * hf_b
            hf_cat = np.concatenate([hf_floor_tail, hf_sq]) if hf_floor_tail.size else hf_sq
            n_hf = hf_cat.size // floor_hop
            if n_hf > 0:
                hf_mat = hf_cat[: n_hf * floor_hop].reshape(n_hf, floor_hop)
                hf_floor_db_list.extend(10.0 * np.log10(np.maximum(np.mean(hf_mat, axis=1), 1e-24)))
                hf_floor_tail = hf_cat[n_hf * floor_hop :]
            else:
                hf_floor_tail = hf_cat

        if mono.size >= nperseg:
            _, p_b = signal.welch(mono, fs=sr, nperseg=nperseg)
            psd_accum += p_b * n_b
            psd_weight += n_b
            segment_psds_band.append(p_b[band_mask].copy())
            if len(segment_psds_full) < 256:
                segment_psds_full.append(p_b.copy())
            segment_bounds_s.append(
                (s0 + rel_start_frame / float(sr), s0 + (rel_start_frame + n_b) / float(sr))
            )

        # Rumble curve (125 ms frames)
        if not skip_rumble and (allowed_kinds is None or "low_freq_rumble" in allowed_kinds):
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
        if (
            not skip_clicks
            and (allowed_kinds is None or "impulsive_click" in allowed_kinds)
            and min_event_s <= 0.15
        ):
            hp_b, zi_hp = signal.sosfilt(sos_hp, block, axis=0, zi=zi_hp)
            env_b = np.max(np.abs(hp_b), axis=1)
            med_b = float(np.median(env_b))
            sig_b = max(float(np.median(np.abs(env_b - med_b))) * 1.4826, 1e-7)
            z_b = (env_b - med_b) / sig_b
            hits_b = np.flatnonzero((z_b > 10.0) & (env_b > 1e-4))
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
        if allowed_kinds is None or bool({"slow_pass", "transient_distraction"} & allowed_kinds):
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
    if floor_db_list:
        noise_floor_dbfs, ambient_bed_dbfs = _floor_stats_from_chunk_rms(floor_db_list)
    else:
        noise_floor_dbfs = rms_dbfs
        ambient_bed_dbfs = rms_dbfs
    high_freq_floor_dbfs = (
        _floor_stats_from_chunk_rms(hf_floor_db_list)[0] if hf_floor_db_list else None
    )

    integrated_lufs: float | None = None
    loudness_range_lu: float | None = None
    short_term_lufs_stats: dict[str, float] | None = None
    short_term_lufs_timeline: list[dict[str, float]] = []
    if rms_lin > 1e-10:
        integrated_lufs = lufs_acc.finalize()
        loudness_range_lu, short_term_lufs_stats, short_term_lufs_timeline = (
            lufs_acc.short_term_metrics(time_offset_s=s0)
        )

    stereo_correlation: float | None = None
    mid_side_ratio_db: float | None = None
    channel_balance_db: float | None = None
    ch2_low_loss_db: float | None = None
    lf_coherence: float | None = None
    directivity_diff_db: float | None = None
    if ch >= 2:
        n_f = float(total_frames)
        var_l = max(0.0, sum_l2 - (sum_l * sum_l) / n_f)
        var_r = max(0.0, sum_r2 - (sum_r * sum_r) / n_f)
        denom = math.sqrt(var_l * var_r)
        if denom > 1e-12:
            cov = sum_lr - (sum_l * sum_r) / n_f
            stereo_correlation = _finite_or_none(max(-1.0, min(1.0, cov / denom)))
        rms_l_g = math.sqrt(sum_l2 / n_f)
        rms_r_g = math.sqrt(sum_r2 / n_f)
        if rms_l_g > 1e-12 or rms_r_g > 1e-12:
            channel_balance_db = _finite_or_none(
                20.0 * math.log10(max(rms_l_g, 1e-12) / max(rms_r_g, 1e-12)), digits=3
            )
            c1 = 20.0 * math.log10(max(max_l_lin, 1e-12) / max(rms_l_g, 1e-12))
            c2 = 20.0 * math.log10(max(max_r_lin, 1e-12) / max(rms_r_g, 1e-12))
            directivity_diff_db = (c1 - c2) + abs(channel_balance_db or 0.0) * 0.5
        if e1_mid > 0.0 and e2_mid > 0.0 and e1_low > 0.0 and e2_low > 0.0:
            c1_lm = 10.0 * math.log10(e1_low / e1_mid)
            c2_lm = 10.0 * math.log10(e2_low / e2_mid)
            ch2_low_loss_db = c1_lm - c2_lm
        if lf_p1 > 0.0 and lf_p2 > 0.0:
            lf_coherence = max(
                0.0, min(1.0, math.hypot(csd_lf_real, csd_lf_imag) / math.sqrt(lf_p1 * lf_p2))
            )
        mid_rms = math.sqrt(sum_mid2 / n_f)
        side_rms = math.sqrt(sum_side2 / n_f)
        if mid_rms > 1e-12:
            mid_side_ratio_db = _finite_or_none(20.0 * math.log10(mid_rms / max(side_rms, 1e-12)))

    layout_info = detect_channel_layout_from_stats(
        ch,
        stereo_correlation,
        mid_side_ratio_db,
        channel_balance_db,
        ch2_low_loss_db=ch2_low_loss_db,
        low_freq_coherence=lf_coherence,
        directivity_diff_db=directivity_diff_db,
    )

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
        ) = _finalize_spectral_summary(
            freqs_ref,
            psd_mean,
            sr,
            freqs_band=freqs_band,
            segment_psds_band=segment_psds_band,
        )

    pleasant = compute_pleasantness_metrics(
        freqs_ref,
        psd_mean,
        roughness_values=roughness_samples,
        rumble_ratio_db=rumble_ratio_db,
        num_drones=len(hum_peaks_hz) + len(tonal_peaks_hz),
    )
    band_metrics = compute_band_metrics_from_psd(
        freqs_ref,
        psd_mean,
        custom_range=freq_range_hz,
        seg_psds=segment_psds_full or None,
        seg_freqs=freqs_ref,
    )

    # Cross-block synthetic beep frequency fingerprinting
    beep_events = [ev for ev in outdoor_events if ev.kind == "synthetic_beep"]
    if len(beep_events) >= 2:
        centers = [
            0.5 * (b.freq_band_hz[0] + b.freq_band_hz[1]) if b.freq_band_hz else 0.0
            for b in beep_events
        ]
        for b, fc in zip(beep_events, centers, strict=True):
            reps = sum(1 for c in centers if abs(c - fc) <= 4.0)
            if reps >= 2:
                b.severity = "high"
                b.confidence = _finite_or_none(min(0.99, max(b.confidence or 0.7, 0.94)), 2)

    events: list[DistractionEvent] = _build_global_events(
        duration_s,
        s0,
        hum_peaks_hz,
        tonal_peaks_hz,
        freq_range_hz=freq_range_hz,
    )
    events.extend(outdoor_events)

    if len(segment_psds_band) >= 3 and (allowed_kinds is None or "machine_drone" in allowed_kinds):
        events.extend(
            _detect_segment_local_tones(
                freqs_band,
                segment_psds_band,
                segment_bounds_s,
                sr,
                [*hum_peaks_hz, *tonal_peaks_hz],
                freq_range_hz=freq_range_hz,
            )
        )

    if (
        not skip_rumble
        and rumble_db_list
        and (allowed_kinds is None or "low_freq_rumble" in allowed_kinds)
    ):
        r_arr = np.asarray(rumble_db_list, dtype=np.float64)
        t_r = np.arange(r_arr.size) * (hop_r / sr)
        events.extend(
            _rumble_events_from_curve(
                t_r,
                r_arr,
                hop_r / sr,
                duration_s,
                s0,
                sensitivity=sensitivity,
                min_event_s=min_event_s,
                max_event_s=max_event_s,
                ambient_bed_dbfs=ambient_bed_dbfs,
            )
        )

    texture_tick_rate_per_min: float | None = None
    if (
        not skip_clicks
        and click_hits
        and (allowed_kinds is None or "impulsive_click" in allowed_kinds)
    ):
        click_events = _click_events_from_hits(
            click_hits, sr, duration_s, cutoff_hp, s0, min_event_s=min_event_s
        )
        events.extend(click_events)
        texture_tick_rate_per_min = _compute_texture_tick_rate(
            click_events,
            duration_s,
            ambient_bed_dbfs if ambient_bed_dbfs is not None else rms_dbfs,
        )

    if fg_db_list and (
        allowed_kinds is None or bool({"slow_pass", "transient_distraction"} & allowed_kinds)
    ):
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

    if clip_indices and (allowed_kinds is None or "clipping" in allowed_kinds):
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

    if allowed_kinds is not None:
        events = [ev for ev in events if ev.kind in allowed_kinds]

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
        loudness_range_lu=loudness_range_lu,
        noise_floor_dbfs=noise_floor_dbfs,
        ambient_bed_dbfs=ambient_bed_dbfs,
        high_freq_floor_dbfs=high_freq_floor_dbfs,
        texture_tick_rate_per_min=texture_tick_rate_per_min,
        short_term_lufs_stats=short_term_lufs_stats,
        short_term_lufs_timeline=short_term_lufs_timeline,
        sharpness_acum=pleasant["sharpness_acum"],
        roughness_index=pleasant["roughness_index"],
        ndsi=pleasant["ndsi"],
        spectral_slope_db_per_oct=pleasant["spectral_slope_db_per_oct"],
        channel_balance_db=channel_balance_db,
        stereo_layout_hint=layout_info["layout"],
        stereo_layout_confidence=layout_info["confidence"],
        pleasantness_score=pleasant["pleasantness_score"],
        band_metrics=band_metrics,
    )
    return metrics, events


YOUTUBE_ASMR_DURATIONS_MIN: tuple[int, ...] = (
    10,
    15,
    20,
    30,
    45,
    60,
    90,
    120,
    180,
    240,
    300,
    360,
    480,
    600,
)

_LOCALIZED_DISTRACTION_KINDS = frozenset(
    {
        "low_freq_rumble",
        "transient_distraction",
        "slow_pass",
        "clipping",
        "synthetic_beep",
        "click_train",
        "crackle_burst",
        "human_voice",
        "dog_bark",
        "variable_rpm_engine",
        "channel_biased_noise",
        "wind_noise",
    }
)


def _format_hms(seconds: float) -> str:
    total_s = max(0, int(round(float(seconds))))
    hours, rem = divmod(total_s, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_round_duration(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes} min"
    if minutes % 60 == 0:
        return f"{minutes // 60} hr"
    if minutes % 30 == 0:
        return f"{minutes / 60.0:.1f} hr"
    return f"{minutes} min"


def _merge_span_list(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not spans:
        return []
    ordered = sorted((float(s), float(e)) for s, e in spans if float(e) > float(s))
    if not ordered:
        return []
    merged: list[tuple[float, float]] = [ordered[0]]
    for s, e in ordered[1:]:
        if s <= merged[-1][1] + 1e-6:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _apply_cut_transform(
    spans: list[tuple[float, float]], regions: list[dict[str, Any]]
) -> list[tuple[float, float]]:
    """Remap prior spans past cut_regions and insert brief splice markers at cut joins."""
    cuts = _merge_span_list(
        [
            (float(r["start_s"]), float(r["end_s"]))
            for r in regions
            if "start_s" in r and "end_s" in r
        ]
    )
    if not cuts:
        return spans

    def _removed_before(t: float) -> float:
        removed = 0.0
        for cs, ce in cuts:
            if t <= cs:
                break
            removed += min(t, ce) - cs
        return removed

    remapped: list[tuple[float, float]] = []
    for s, e in spans:
        cur_s = s
        for cs, ce in cuts:
            if ce <= cur_s:
                continue
            if cs >= e:
                break
            if cur_s < cs:
                off = _removed_before(cur_s)
                remapped.append((cur_s - off, cs - off))
            cur_s = max(cur_s, ce)
        if cur_s < e:
            off = _removed_before(cur_s)
            remapped.append((cur_s - off, e - off))

    # Mark each kept-segment splice boundary so crossfaded cuts split untouched spans.
    removed_so_far = 0.0
    for cs, ce in cuts:
        join_t = max(0.0, cs - removed_so_far)
        remapped.append((max(0.0, join_t - 0.25), join_t + 0.25))
        removed_so_far += ce - cs
    return remapped


def extract_localized_edit_spans(
    recipe: ProcessingRecipe | dict[str, Any] | None,
) -> list[tuple[float, float]]:
    """Extract finalized spans where localized edits ran."""
    if recipe is None:
        return []
    steps = recipe.get("steps") or [] if isinstance(recipe, dict) else list(recipe.steps)
    cur_spans: list[tuple[float, float]] = []

    for st in steps:
        fname = st.get("filter_name") if isinstance(st, dict) else st.filter_name
        params = st.get("params") or {} if isinstance(st, dict) else dict(st.params)
        is_local_declick = fname == "declick_impulse" and bool(
            params.get("spans") or params.get("start_s") is not None
        )
        if fname in {
            "band_attenuate",
            "spectral_match_suppress",
            "ambience_patch",
            "harmonic_track_suppress",
            "distance_soften",
        } or is_local_declick:
            if params.get("spans"):
                for sp in params["spans"]:
                    s = sp.get("start_s")
                    e = sp.get("end_s")
                    if s is not None and e is not None and float(e) > float(s):
                        cur_spans.append((max(0.0, float(s)), float(e)))
            else:
                s = params.get("start_s")
                e = params.get("end_s")
                if s is not None and e is not None and float(e) > float(s):
                    cur_spans.append((max(0.0, float(s)), float(e)))
        elif fname == "trim_range":
            t0 = max(0.0, float(params.get("start_s", 0.0)))
            t1_raw = params.get("end_s")
            t1 = float("inf") if t1_raw is None else float(t1_raw)
            shifted: list[tuple[float, float]] = []
            for s, e in cur_spans:
                cs = max(t0, s)
                ce = min(t1, e)
                if ce > cs:
                    shifted.append((cs - t0, ce - t0))
            cur_spans = shifted
        elif fname == "cut_regions":
            cur_spans = _apply_cut_transform(cur_spans, params.get("regions") or [])

    return _merge_span_list(cur_spans)


def extract_localized_distraction_spans(
    events: list[DistractionEvent],
    include_low_severity: bool = False,
) -> list[tuple[float, float]]:
    """Extract residual localized distraction spans (excluding clicks and steady global drones)."""
    raw = [
        (max(0.0, float(ev.start_s)), float(ev.end_s))
        for ev in events
        if ev.kind in _LOCALIZED_DISTRACTION_KINDS
        and float(ev.end_s) > float(ev.start_s)
        and (include_low_severity or ev.severity != "low")
    ]
    return _merge_span_list(raw)


def _count_and_overlap(
    intervals: list[tuple[float, float]],
    starts: list[float],
    ends: list[float],
    w0: float,
    w1: float,
    prefix_dur: list[float] | None = None,
) -> tuple[int, float]:
    """Count intervals intersecting (w0, w1) and optionally sum their overlap in O(log N)."""
    if not intervals:
        return 0, 0.0
    lo = bisect_right(ends, w0 + 1e-9)
    hi = bisect_left(starts, w1 - 1e-9)
    count = max(0, hi - lo)
    if prefix_dur is None or count == 0:
        return count, 0.0
    if count == 1:
        s, e = intervals[lo]
        return 1, max(0.0, min(e, w1) - max(s, w0))
    total = 0.0
    s_first, e_first = intervals[lo]
    total += max(0.0, min(e_first, w1) - max(s_first, w0))
    if hi - lo > 2:
        total += prefix_dur[hi - 1] - prefix_dur[lo + 1]
    s_last, e_last = intervals[hi - 1]
    total += max(0.0, min(e_last, w1) - max(s_last, w0))
    return count, total


def compute_clean_sections(
    segments: list[dict[str, Any]],
    max_edits_per_hour: float = 2.0,
    max_edited_pct: float = 1.0,
    round_to_file_local_minutes: bool = False,
    exclude_spans: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Find largest untouched and minimal-edit stretches sized for YouTube ASMR videos."""
    if not segments:
        return {
            "total_duration_s": 0.0,
            "total_duration_hms": "00:00:00",
            "segment_count": 0,
            "total_localized_edits": 0,
            "total_localized_distractions": 0,
            "largest_untouched_raw_min": 0,
            "largest_untouched_asmr_target": None,
            "largest_minimal_edit_asmr_target": None,
            "top_untouched_gaps": [],
            "asmr_targets": [],
        }

    seg_bounds: list[tuple[float, float, str]] = []
    all_edits: list[tuple[float, float]] = []
    all_distractions: list[tuple[float, float]] = []
    cum_t = 0.0

    for seg in segments:
        dur = max(0.0, float(seg.get("duration_s", 0.0)))
        label = str(seg.get("label") or f"segment_{len(seg_bounds) + 1}")
        seg_start = cum_t
        seg_end = cum_t + dur
        seg_bounds.append((seg_start, seg_end, label))

        for s, e in seg.get("edit_spans") or []:
            cs = max(0.0, min(dur, float(s)))
            ce = max(0.0, min(dur, float(e)))
            if ce > cs:
                all_edits.append((seg_start + cs, seg_start + ce))
        for s, e in seg.get("distraction_spans") or []:
            cs = max(0.0, min(dur, float(s)))
            ce = max(0.0, min(dur, float(e)))
            if ce > cs:
                all_distractions.append((seg_start + cs, seg_start + ce))
        cum_t = seg_end

    total_dur = cum_t
    if total_dur <= 0.0:
        return {
            "total_duration_s": 0.0,
            "total_duration_hms": "00:00:00",
            "segment_count": len(segments),
            "total_localized_edits": 0,
            "total_localized_distractions": 0,
            "largest_untouched_raw_min": 0,
            "largest_untouched_asmr_target": None,
            "largest_minimal_edit_asmr_target": None,
            "top_untouched_gaps": [],
            "asmr_targets": [],
        }

    seg_ends = [b[1] for b in seg_bounds]

    def _map_cum_time(t: float) -> tuple[str, float, str]:
        clamped = max(0.0, min(total_dur, float(t)))
        idx = min(len(seg_bounds) - 1, bisect_right(seg_ends, clamped))
        if idx > 0 and abs(clamped - seg_bounds[idx - 1][1]) < 1e-6:
            idx -= 1
        b_s, _, b_label = seg_bounds[idx]
        local_s = max(0.0, clamped - b_s)
        return b_label, round(local_s, 3), _format_hms(local_s)

    excl_merged = _merge_span_list(
        [(float(a), float(b)) for a, b in (exclude_spans or []) if float(b) > float(a)]
    )

    merged_edits = _merge_span_list(all_edits)
    merged_distractions = _merge_span_list(all_distractions)
    merged_issues = _merge_span_list([*merged_edits, *merged_distractions])

    edit_starts = [s for s, _ in merged_edits]
    edit_ends = [e for _, e in merged_edits]
    edit_prefix = [0.0]
    for s, e in merged_edits:
        edit_prefix.append(edit_prefix[-1] + (e - s))

    dist_starts = [s for s, _ in merged_distractions]
    dist_ends = [e for _, e in merged_distractions]
    dist_prefix = [0.0]
    for s, e in merged_distractions:
        dist_prefix.append(dist_prefix[-1] + (e - s))

    untouched_gaps: list[tuple[float, float]] = []
    cur = 0.0
    for is_s, is_e in merged_issues:
        if is_s > cur + 1e-6:
            untouched_gaps.append((cur, is_s))
        cur = max(cur, is_e)
    if total_dur > cur + 1e-6:
        untouched_gaps.append((cur, total_dur))
    untouched_gaps.sort(key=lambda g: (-(g[1] - g[0]), g[0]))

    longest_gap_s = (untouched_gaps[0][1] - untouched_gaps[0][0]) if untouched_gaps else 0.0
    largest_untouched_raw_min = int(longest_gap_s // 60.0)

    cand_durations = [
        d for d in YOUTUBE_ASMR_DURATIONS_MIN if d * 60.0 <= total_dur + 1e-6
    ]

    top_gaps_src = [g for g in untouched_gaps if (g[1] - g[0]) >= 60.0] or untouched_gaps
    top_untouched_gaps: list[dict[str, Any]] = []
    for g0, g1 in top_gaps_src[:5]:
        raw_dur = g1 - g0
        r_start = float(math.ceil(g0))
        r_end = float(math.floor(g1))
        if r_end <= r_start:
            r_start, r_end = round(g0, 1), round(g1, 1)
        fit_min = max((d for d in cand_durations if d * 60.0 <= raw_dur + 1e-6), default=None)
        sf_name, _, s_loc_hms = _map_cum_time(r_start)
        ef_name, _, e_loc_hms = _map_cum_time(r_end)
        top_untouched_gaps.append(
            {
                "duration_s": round(raw_dur, 1),
                "duration_min": int(raw_dur // 60.0),
                "round_asmr_label": _format_round_duration(fit_min) if fit_min else None,
                "start_s": r_start,
                "end_s": r_end,
                "start_hms": _format_hms(r_start),
                "end_hms": _format_hms(r_end),
                "start_file": sf_name,
                "start_local_hms": s_loc_hms,
                "end_file": ef_name,
                "end_local_hms": e_loc_hms,
            }
        )

    asmr_targets: list[dict[str, Any]] = []
    for d_min in cand_durations:
        win_s = float(d_min * 60)
        max_start = max(0.0, total_dur - win_s)
        cand_starts: set[float] = {0.0, round(max_start, 2)}
        for m in range(int(max_start // 60.0) + 1):
            cand_starts.add(float(m * 60))
        local_minute_starts: set[float] = set()
        if round_to_file_local_minutes:
            for b_s, b_e, _ in seg_bounds:
                seg_dur = max(0.0, b_e - b_s)
                for m in range(int(seg_dur // 60.0) + 1):
                    cand_val = b_s + float(m * 60)
                    if cand_val <= max_start + 1e-6:
                        cand_starts.add(min(max_start, cand_val))
                        local_minute_starts.add(round(min(max_start, cand_val), 3))
        for g0, g1 in untouched_gaps:
            s_ceil = float(math.ceil(g0))
            if s_ceil <= max_start + 1e-6:
                cand_starts.add(min(max_start, s_ceil))
            s_tail = float(math.floor(g1)) - win_s
            if 0.0 <= s_tail <= max_start + 1e-6:
                cand_starts.add(min(max_start, max(0.0, s_tail)))

        best_key: tuple[int, float, int, float] | None = None
        best_info: tuple[float, float, int, int, float, float] | None = None
        for s_cand in sorted(cand_starts):
            w0 = max(0.0, min(max_start, s_cand))
            w1 = w0 + win_s
            if excl_merged and any(w1 > ex0 + 1e-6 and w0 < ex1 - 1e-6 for ex0, ex1 in excl_merged):
                continue
            n_ed, edited_s = _count_and_overlap(
                merged_edits, edit_starts, edit_ends, w0, w1, prefix_dur=edit_prefix
            )
            n_dist, dist_s = _count_and_overlap(
                merged_distractions, dist_starts, dist_ends, w0, w1, prefix_dur=dist_prefix
            )
            is_local_round = round_to_file_local_minutes and (round(w0, 3) in local_minute_starts)
            if round_to_file_local_minutes:
                round_pref = 0 if is_local_round else (1 if abs(w0 % 60.0) < 1e-6 else 2)
            else:
                round_pref = 0 if abs(w0 % 60.0) < 1e-6 else 1
            score_key = (n_ed + n_dist, round(edited_s + dist_s, 3), round_pref, w0)
            if best_key is None or score_key < best_key:
                best_key = score_key
                best_info = (w0, w1, n_ed, n_dist, edited_s, dist_s)

        if best_info is None:
            continue
        w0, w1, n_ed, n_dist, edited_s, dist_s = best_info
        raw_ed = sum(1 for es, ee in all_edits if ee > w0 + 1e-9 and es < w1 - 1e-9)
        edited_pct = round(100.0 * edited_s / max(win_s, 1e-6), 3)
        dist_pct = round(100.0 * dist_s / max(win_s, 1e-6), 3)
        max_allowed = max(1, int(round(float(max_edits_per_hour) * (d_min / 60.0))))
        violations: list[str] = []
        if n_ed == 0 and n_dist == 0:
            tier = "untouched"
        elif n_ed <= max_allowed and n_dist <= max_allowed and edited_pct <= float(max_edited_pct):
            tier = "minimal_edits"
        else:
            tier = "moderate_edits"
            if n_ed > max_allowed:
                violations.append(f"edit_count {n_ed} > max_allowed {max_allowed}")
            if n_dist > max_allowed:
                violations.append(f"distraction_count {n_dist} > max_allowed {max_allowed}")
            if edited_pct > float(max_edited_pct):
                violations.append(f"edited_pct {edited_pct:.3f}% > {float(max_edited_pct):.3f}%")

        sf_name, s_loc_s, s_loc_hms = _map_cum_time(w0)
        ef_name, e_loc_s, e_loc_hms = _map_cum_time(w1)
        asmr_targets.append(
            {
                "duration_label": _format_round_duration(d_min),
                "duration_min": int(d_min),
                "tier": tier,
                "eligible": tier in {"untouched", "minimal_edits"},
                "violations": violations,
                "start_s": round(w0, 3),
                "end_s": round(w1, 3),
                "start_hms": _format_hms(w0),
                "end_hms": _format_hms(w1),
                "start_file": sf_name,
                "start_local_s": s_loc_s,
                "start_local_hms": s_loc_hms,
                "end_file": ef_name,
                "end_local_s": e_loc_s,
                "end_local_hms": e_loc_hms,
                "edit_count": int(n_ed),
                "raw_edit_span_count": int(raw_ed),
                "distraction_count": int(n_dist),
                "residual_distraction_count": int(n_dist),
                "edited_s": round(edited_s, 3),
                "edited_pct": edited_pct,
                "residual_distraction_s": round(dist_s, 3),
                "residual_distraction_pct": dist_pct,
            }
        )

    untouched_targets = [t for t in asmr_targets if t["tier"] == "untouched"]
    minimal_targets = [t for t in asmr_targets if t["tier"] in {"untouched", "minimal_edits"}]

    return {
        "total_duration_s": round(total_dur, 1),
        "total_duration_hms": _format_hms(total_dur),
        "segment_count": len(segments),
        "total_localized_edits": len(merged_edits),
        "total_localized_distractions": len(merged_distractions),
        "largest_untouched_raw_min": largest_untouched_raw_min,
        "largest_untouched_asmr_target": untouched_targets[-1] if untouched_targets else None,
        "largest_minimal_edit_asmr_target": minimal_targets[-1] if minimal_targets else None,
        "top_untouched_gaps": top_untouched_gaps,
        "asmr_targets": asmr_targets,
    }
