"""Headless waveform and spectrogram PNG renderers supporting zoomed windows and long files."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from ambient_sound.io_audio import ensure_2d_audio, load_audio_slice, probe_audio
from ambient_sound.models import DistractionEvent


def _compute_spectrogram_db(
    mono: np.ndarray,
    sample_rate: int,
    n_fft: int = 8192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mono.size < 32:
        freqs = np.array([20.0, max(100.0, sample_rate * 0.5)])
        times = np.array([0.0, max(0.01, mono.size / max(sample_rate, 1))])
        spec_db = np.full((2, 2), -90.0, dtype=np.float64)
        return freqs, times, spec_db

    nperseg = min(n_fft, max(256, (mono.size // 4) * 2)) if mono.size < n_fft else n_fft
    nperseg = min(nperseg, mono.size)
    noverlap = int(nperseg * 0.75)
    freqs, times, zxx = signal.stft(
        mono,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )
    mag = np.abs(zxx)
    spec_db = 20.0 * np.log10(np.maximum(mag, 1e-6))
    return freqs, times, spec_db


def _write_plot_meta(
    fig: Any,
    ax_top: Any,
    ax_bot: Any,
    x_start_s: float,
    x_end_s: float,
    out_path: Path,
) -> None:
    """Persist normalized figure plot bounds alongside PNG for Web UI playhead overlay."""
    fig.canvas.draw()
    p_top = ax_top.get_position()
    p_bot = ax_bot.get_position()
    meta = {
        "x0": round(float(p_bot.x0), 5),
        "x1": round(float(p_bot.x1), 5),
        "top": round(1.0 - float(p_top.y1), 5),
        "bottom": round(1.0 - float(p_bot.y0), 5),
        "t_start": round(float(x_start_s), 4),
        "t_end": round(float(max(x_start_s + 0.01, x_end_s)), 4),
    }
    meta_path = Path(f"{out_path}.meta.json")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def _plot_spectrogram_figure(
    t_wav: np.ndarray,
    wav_lo: np.ndarray,
    wav_hi: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    spec_db: np.ndarray,
    sample_rate: int,
    x_start_s: float,
    x_end_s: float,
    out_path: str | Path,
    title: str,
    events: list[DistractionEvent] | None,
    vlines: list[tuple[float, str, str]] | None = None,
) -> Path:
    max_db = float(np.max(spec_db)) if spec_db.size else 0.0
    vmin = max_db - 85.0
    vmax = max_db

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_wav, ax_spec) = plt.subplots(
        2,
        1,
        figsize=(11, 5.5),
        dpi=130,
        gridspec_kw={"height_ratios": [1.0, 2.2]},
        sharex=True,
        layout="constrained",
    )
    try:
        fig.patch.set_facecolor("#0f141c")
        for ax in (ax_wav, ax_spec):
            ax.set_facecolor("#161e29")
            ax.tick_params(colors="#b8c4d4", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#2a3647")

        if t_wav.size > 0:
            if np.allclose(wav_lo, wav_hi):
                ax_wav.plot(t_wav, wav_hi, color="#59c3c3", linewidth=0.7)
            else:
                ax_wav.fill_between(t_wav, wav_lo, wav_hi, color="#59c3c3", alpha=0.85)
            peak = max(
                0.05,
                float(max(np.max(np.abs(wav_lo)), np.max(np.abs(wav_hi)))) * 1.1,
            )
            ax_wav.set_ylim(-peak, peak)
        ax_wav.set_ylabel("Amp", color="#d8e2ec", fontsize=9)
        display_title = title or "Ambient Recording Overview"

        if events:
            in_view_clicks = sum(
                1
                for ev in events
                if ev.kind == "impulsive_click"
                and ev.end_s >= x_start_s
                and ev.start_s <= x_end_s
            )
            skip_clicks = in_view_clicks > 40
            if skip_clicks:
                display_title = f"{display_title} ({in_view_clicks} click markers omitted)"
            palette = {
                "low_freq_rumble": "#f59e0b",
                "impulsive_click": "#ef4444",
                "transient_distraction": "#f43f5e",
                "slow_pass": "#fb7185",
                "clipping": "#dc2626",
            }
            for ev in events:
                if ev.kind in {"mains_hum", "machine_drone"}:
                    continue
                if skip_clicks and ev.kind == "impulsive_click":
                    continue
                if ev.end_s < x_start_s or ev.start_s > x_end_s:
                    continue
                color = palette.get(ev.kind, "#f43f5e")
                ax_wav.axvspan(
                    max(x_start_s, ev.start_s),
                    min(x_end_s, ev.end_s),
                    color=color,
                    alpha=0.22,
                )

        ax_wav.set_title(display_title, color="#eef3f8", fontsize=11)

        pos_mask = freqs >= 20.0
        f_plot = freqs[pos_mask] if np.any(pos_mask) else freqs[1:]
        s_plot = spec_db[pos_mask, :] if np.any(pos_mask) else spec_db[1:, :]
        if f_plot.size >= 2 and times.size >= 2:
            mesh = ax_spec.pcolormesh(
                times,
                f_plot,
                s_plot,
                shading="auto",
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
            )
            ax_spec.set_yscale("log")
            ax_spec.set_ylim(max(20.0, float(f_plot[0])), max(100.0, sample_rate * 0.5))
            cbar = fig.colorbar(mesh, ax=ax_spec, pad=0.015, aspect=28)
            cbar.ax.tick_params(colors="#b8c4d4", labelsize=7)
            cbar.set_label("dB", color="#d8e2ec", fontsize=8)

        ax_spec.set_ylabel("Freq (Hz)", color="#d8e2ec", fontsize=9)
        ax_spec.set_xlabel("Time (s)", color="#d8e2ec", fontsize=9)
        ax_spec.set_xlim(x_start_s, max(x_start_s + 0.01, x_end_s))

        if vlines:
            for vx, vcolor, vlabel in vlines:
                if x_start_s <= vx <= x_end_s:
                    ax_wav.axvline(vx, color=vcolor, linestyle="--", linewidth=1.2, alpha=0.9)
                    ax_spec.axvline(vx, color=vcolor, linestyle="--", linewidth=1.2, alpha=0.9)
                    if vlabel:
                        ax_wav.text(
                            vx,
                            ax_wav.get_ylim()[1] * 0.88,
                            f" {vlabel}",
                            color=vcolor,
                            fontsize=8,
                            va="top",
                            ha="left",
                        )

        fig.savefig(str(out), facecolor=fig.get_facecolor())
        _write_plot_meta(fig, ax_wav, ax_spec, x_start_s, x_end_s, out)
    finally:
        plt.close(fig)
    return out


def _plot_diff_figure(
    times: np.ndarray,
    freqs: np.ndarray,
    s_b: np.ndarray,
    s_a: np.ndarray,
    s_r: np.ndarray,
    sample_rate: int,
    x_start_s: float,
    x_end_s: float,
    out_path: str | Path,
    title: str,
    residual_label: str = "Removed Residual (Parent - Processed)",
    patched_spans: list[tuple[float, float]] | None = None,
) -> Path:
    vmax = float(max(np.max(s_b), np.max(s_a), -40.0))
    vmin = vmax - 85.0

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 7.5),
        dpi=130,
        sharex=True,
        layout="constrained",
    )
    panels = [
        (axes[0], s_b, "Before (Parent Take)"),
        (axes[1], s_a, "After (Processed Take)"),
        (axes[2], s_r, residual_label),
    ]
    try:
        fig.patch.set_facecolor("#0f141c")
        pos_mask = freqs >= 20.0
        f_plot = freqs[pos_mask] if np.any(pos_mask) else freqs[1:]

        last_mesh = None
        for ax, spec, subtitle in panels:
            ax.set_facecolor("#161e29")
            ax.tick_params(colors="#b8c4d4", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#2a3647")
            s_plot = spec[pos_mask, :] if np.any(pos_mask) else spec[1:, :]
            if f_plot.size >= 2 and times.size >= 2:
                last_mesh = ax.pcolormesh(
                    times,
                    f_plot,
                    s_plot,
                    shading="auto",
                    cmap="magma",
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_yscale("log")
                ax.set_ylim(max(20.0, float(f_plot[0])), max(100.0, sample_rate * 0.5))
            ax.set_title(subtitle, color="#eef3f8", fontsize=9.5, loc="left")
            ax.set_ylabel("Hz", color="#d8e2ec", fontsize=8)

        if patched_spans:
            y_hi = max(100.0, sample_rate * 0.5) * 0.72
            for ps, pe in patched_spans:
                if pe < x_start_s or ps > x_end_s:
                    continue
                s_c = max(x_start_s, float(ps))
                e_c = min(x_end_s, float(pe))
                for ax in (axes[1], axes[2]):
                    ax.axvline(s_c, color="#38bdf8", linestyle="--", linewidth=0.9, alpha=0.8)
                    ax.axvline(e_c, color="#38bdf8", linestyle="--", linewidth=0.9, alpha=0.8)
                    ax.axvspan(s_c, e_c, color="#38bdf8", alpha=0.10)
                axes[1].text(
                    0.5 * (s_c + e_c),
                    y_hi,
                    "Patched",
                    color="#7dd3fc",
                    fontsize=7.5,
                    ha="center",
                    va="top",
                    bbox={"facecolor": "#0f172a", "alpha": 0.65, "edgecolor": "none", "pad": 1.5},
                )

        axes[-1].set_xlabel("Time (s)", color="#d8e2ec", fontsize=9)
        axes[-1].set_xlim(x_start_s, max(x_start_s + 0.01, x_end_s))
        if title:
            fig.suptitle(title, color="#eef3f8", fontsize=11)
        if last_mesh is not None:
            cbar = fig.colorbar(last_mesh, ax=axes, pad=0.015, aspect=40)
            cbar.ax.tick_params(colors="#b8c4d4", labelsize=8)
            cbar.set_label("dB", color="#d8e2ec", fontsize=9)

        fig.savefig(str(out), facecolor=fig.get_facecolor())
        _write_plot_meta(fig, axes[0], axes[-1], x_start_s, x_end_s, out)
    finally:
        plt.close(fig)
    return out


def render_spectrogram_png(
    audio: np.ndarray,
    sample_rate: int,
    out_path: str | Path,
    title: str = "",
    events: list[DistractionEvent] | None = None,
    time_offset_s: float = 0.0,
    vlines: list[tuple[float, str, str]] | None = None,
) -> Path:
    """Render a dark-themed waveform + log-frequency spectrogram PNG from an array."""
    arr = ensure_2d_audio(audio)
    mono = np.mean(arr, axis=1)
    duration_s = mono.size / float(max(sample_rate, 1))
    freqs, times, spec_db = _compute_spectrogram_db(mono, sample_rate)

    if mono.size > 0:
        step = max(1, mono.size // 4000)
        wav_ds = mono[::step]
        t_wav = time_offset_s + np.linspace(0.0, duration_s, num=wav_ds.size)
    else:
        wav_ds = np.zeros(0, dtype=np.float64)
        t_wav = np.zeros(0, dtype=np.float64)

    return _plot_spectrogram_figure(
        t_wav=t_wav,
        wav_lo=wav_ds,
        wav_hi=wav_ds,
        times=times + time_offset_s,
        freqs=freqs,
        spec_db=spec_db,
        sample_rate=sample_rate,
        x_start_s=time_offset_s,
        x_end_s=time_offset_s + duration_s,
        out_path=out_path,
        title=title,
        events=events,
        vlines=vlines,
    )


def render_diff_spectrogram_png(
    before: np.ndarray,
    after: np.ndarray,
    sample_rate: int,
    out_path: str | Path,
    title: str = "",
    time_offset_s: float = 0.0,
    loudness_match_residual: bool = True,
    patched_spans: list[tuple[float, float]] | None = None,
) -> Path:
    """Render a 3-panel Before / After / Removed Residual spectrogram comparison."""
    arr_b = ensure_2d_audio(before)
    arr_a = ensure_2d_audio(after)
    n = min(arr_b.shape[0], arr_a.shape[0])
    mono_b = np.mean(arr_b[:n], axis=1)
    mono_a = np.mean(arr_a[:n], axis=1)

    residual_label = "Removed Residual (Parent - Processed)"
    mono_a_cmp = mono_a
    if loudness_match_residual and mono_b.size > 0:
        rms_b = float(np.sqrt(np.mean(mono_b * mono_b)))
        rms_a = float(np.sqrt(np.mean(mono_a * mono_a)))
        if rms_b > 1e-12 and rms_a > 1e-12:
            diff_db = abs(20.0 * math.log10(rms_a / rms_b))
            if diff_db > 1.5:
                mono_a_cmp = mono_a * (rms_b / rms_a)
                residual_label = "Gain-Matched Spectral Diff (After scaled to Before RMS)"
    mono_r = mono_b - mono_a_cmp

    f_b, t_b, s_b = _compute_spectrogram_db(mono_b, sample_rate)
    _, _, s_a = _compute_spectrogram_db(mono_a, sample_rate)
    _, _, s_r = _compute_spectrogram_db(mono_r, sample_rate)
    duration_s = n / float(max(sample_rate, 1))

    return _plot_diff_figure(
        times=t_b + time_offset_s,
        freqs=f_b,
        s_b=s_b,
        s_a=s_a,
        s_r=s_r,
        sample_rate=sample_rate,
        x_start_s=time_offset_s,
        x_end_s=time_offset_s + duration_s,
        out_path=out_path,
        title=title,
        residual_label=residual_label,
        patched_spans=patched_spans,
    )


def render_file_spectrogram_png(
    path: str | Path,
    out_path: str | Path,
    title: str = "",
    events: list[DistractionEvent] | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
) -> Path:
    """Render a spectrogram PNG in bounded RAM (full resolution <= 60s, overview > 60s)."""
    src = Path(path)
    info = probe_audio(src)
    s0 = max(0.0, float(start_s)) if start_s is not None else 0.0
    s1 = min(info.duration_s, float(end_s)) if end_s is not None else info.duration_s
    span = max(0.0, s1 - s0)

    if span <= 60.0:
        arr, sr = load_audio_slice(src, start_s=s0, end_s=s1, dtype="float64")
        return render_spectrogram_png(
            arr, sr, out_path, title=title, events=events, time_offset_s=s0
        )

    # Low-RAM sampled overview for long recordings (> 60 s)
    n_cols = 800
    n_fft = 8192
    sr = info.sample_rate
    win = np.hanning(n_fft)
    win_norm = max(float(np.sum(win)), 1.0)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    col_times = np.linspace(s0, s1, num=n_cols, endpoint=False)
    spec_cols = np.empty((freqs.size, n_cols), dtype=np.float64)
    wav_lo = np.empty(n_cols, dtype=np.float64)
    wav_hi = np.empty(n_cols, dtype=np.float64)

    probe_dur = max(n_fft / float(sr), min(0.25, span / n_cols))
    for i, t_c in enumerate(col_times):
        chunk, _ = load_audio_slice(
            src, start_s=float(t_c), end_s=min(s1, float(t_c) + probe_dur), dtype="float32"
        )
        if chunk.size == 0:
            wav_lo[i] = wav_hi[i] = 0.0
            spec_cols[:, i] = -120.0
            continue
        mono = np.mean(chunk, axis=1)
        wav_lo[i] = float(np.min(mono))
        wav_hi[i] = float(np.max(mono))
        seg = mono[:n_fft] if mono.size >= n_fft else np.pad(mono, (0, n_fft - mono.size))
        mag = np.abs(np.fft.rfft(seg * win)) / win_norm
        spec_cols[:, i] = 20.0 * np.log10(np.maximum(mag, 1e-6))

    return _plot_spectrogram_figure(
        t_wav=col_times,
        wav_lo=wav_lo,
        wav_hi=wav_hi,
        times=col_times,
        freqs=freqs,
        spec_db=spec_cols,
        sample_rate=sr,
        x_start_s=s0,
        x_end_s=s1,
        out_path=out_path,
        title=title,
        events=events,
    )


def render_file_diff_spectrogram_png(
    before_path: str | Path,
    after_path: str | Path,
    out_path: str | Path,
    title: str = "",
    start_s: float | None = None,
    end_s: float | None = None,
    loudness_match_residual: bool = True,
    patched_spans: list[tuple[float, float]] | None = None,
    before_offset_s: float = 0.0,
) -> Path:
    """Render a 3-panel diff PNG in bounded RAM (full <= 60s, overview > 60s)."""
    info_b = probe_audio(before_path)
    info_a = probe_audio(after_path)
    b_off = max(0.0, float(before_offset_s))
    dur = min(max(0.0, info_b.duration_s - b_off), info_a.duration_s)
    s0 = max(0.0, float(start_s)) if start_s is not None else 0.0
    s1 = min(dur, float(end_s)) if end_s is not None else dur
    span = max(0.0, s1 - s0)

    if span <= 60.0:
        arr_b, sr = load_audio_slice(
            before_path, start_s=b_off + s0, end_s=b_off + s1, dtype="float64"
        )
        arr_a, _ = load_audio_slice(after_path, start_s=s0, end_s=s1, dtype="float64")
        return render_diff_spectrogram_png(
            arr_b,
            arr_a,
            sr,
            out_path,
            title=title,
            time_offset_s=s0,
            loudness_match_residual=loudness_match_residual,
            patched_spans=patched_spans,
        )

    n_cols = 700
    n_fft = 8192
    sr = info_b.sample_rate
    win = np.hanning(n_fft)
    win_norm = max(float(np.sum(win)), 1.0)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    col_times = np.linspace(s0, s1, num=n_cols, endpoint=False)
    s_b = np.empty((freqs.size, n_cols), dtype=np.float64)
    s_a = np.empty((freqs.size, n_cols), dtype=np.float64)
    s_r = np.empty((freqs.size, n_cols), dtype=np.float64)

    probe_dur = max(n_fft / float(sr), min(0.25, span / n_cols))
    seg_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    sum_sq_b = 0.0
    sum_sq_a = 0.0
    n_tot = 0
    for _i, t_c in enumerate(col_times):
        t_end = min(s1, float(t_c) + probe_dur)
        cb, _ = load_audio_slice(
            before_path, start_s=b_off + float(t_c), end_s=b_off + t_end, dtype="float32"
        )
        ca, _ = load_audio_slice(after_path, start_s=float(t_c), end_s=t_end, dtype="float32")
        n = min(cb.shape[0], ca.shape[0])
        if n == 0:
            seg_pairs.append((np.zeros(0), np.zeros(0)))
            continue
        mb = np.mean(cb[:n], axis=1)
        ma = np.mean(ca[:n], axis=1)
        seg_b = mb[:n_fft] if mb.size >= n_fft else np.pad(mb, (0, n_fft - mb.size))
        seg_a = ma[:n_fft] if ma.size >= n_fft else np.pad(ma, (0, n_fft - ma.size))
        seg_pairs.append((seg_b, seg_a))
        sum_sq_b += float(np.sum(seg_b * seg_b))
        sum_sq_a += float(np.sum(seg_a * seg_a))
        n_tot += n_fft

    scale = 1.0
    residual_label = "Removed Residual (Parent - Processed)"
    if loudness_match_residual and n_tot > 0:
        rms_b = math.sqrt(sum_sq_b / n_tot)
        rms_a = math.sqrt(sum_sq_a / n_tot)
        if rms_b > 1e-12 and rms_a > 1e-12 and abs(20.0 * math.log10(rms_a / rms_b)) > 1.5:
            scale = rms_b / rms_a
            residual_label = "Gain-Matched Spectral Diff (After scaled to Before RMS)"

    for i, (seg_b, seg_a) in enumerate(seg_pairs):
        if seg_b.size == 0:
            s_b[:, i] = s_a[:, i] = s_r[:, i] = -120.0
            continue
        seg_r = seg_b - seg_a * scale
        for mat, vec in ((s_b, seg_b), (s_a, seg_a), (s_r, seg_r)):
            mag = np.abs(np.fft.rfft(vec * win)) / win_norm
            mat[:, i] = 20.0 * np.log10(np.maximum(mag, 1e-6))

    return _plot_diff_figure(
        times=col_times,
        freqs=freqs,
        s_b=s_b,
        s_a=s_a,
        s_r=s_r,
        sample_rate=sr,
        x_start_s=s0,
        x_end_s=s1,
        out_path=out_path,
        title=title,
        residual_label=residual_label,
        patched_spans=patched_spans,
    )


def render_seam_spectrogram_png(
    prev_path: str | Path,
    next_path: str | Path,
    out_path: str | Path,
    window_s: float = 10.0,
    title: str = "",
) -> Path:
    """Render [-window_s, +window_s] stitched across a split boundary with SEAM marker at t=0."""
    win_s = max(1.0, min(30.0, float(window_s)))
    info_prev = probe_audio(prev_path)
    info_next = probe_audio(next_path)
    p0 = max(0.0, info_prev.duration_s - win_s)
    prev_arr, sr = load_audio_slice(
        prev_path, start_s=p0, end_s=info_prev.duration_s, dtype="float64"
    )
    next_arr, _ = load_audio_slice(
        next_path, start_s=0.0, end_s=min(win_s, info_next.duration_s), dtype="float64"
    )
    stitched = np.vstack([prev_arr, next_arr]) if prev_arr.size and next_arr.size else (
        prev_arr if prev_arr.size else next_arr
    )
    prev_dur_s = prev_arr.shape[0] / float(max(sr, 1))
    return render_spectrogram_png(
        stitched,
        sr,
        out_path,
        title=title or "Split Seam Verification (-window..+window s)",
        time_offset_s=-prev_dur_s,
        vlines=[(0.0, "#38bdf8", "Seam")],
    )
