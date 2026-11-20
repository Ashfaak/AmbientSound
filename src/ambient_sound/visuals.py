"""Headless waveform and spectrogram PNG renderers supporting zoomed windows and long files."""

from __future__ import annotations

from pathlib import Path

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
    n_fft: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mono.size < 32:
        freqs = np.array([20.0, max(100.0, sample_rate * 0.5)])
        times = np.array([0.0, max(0.01, mono.size / max(sample_rate, 1))])
        spec_db = np.full((2, 2), -90.0, dtype=np.float64)
        return freqs, times, spec_db

    nperseg = min(n_fft, mono.size)
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
        ax_wav.set_title(title or "Ambient Recording Overview", color="#eef3f8", fontsize=11)

        if events:
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
                if ev.end_s < x_start_s or ev.start_s > x_end_s:
                    continue
                color = palette.get(ev.kind, "#f43f5e")
                ax_wav.axvspan(
                    max(x_start_s, ev.start_s),
                    min(x_end_s, ev.end_s),
                    color=color,
                    alpha=0.22,
                )

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

        fig.savefig(str(out), facecolor=fig.get_facecolor())
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
        (axes[2], s_r, "Removed Residual (Parent - Processed)"),
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

        axes[-1].set_xlabel("Time (s)", color="#d8e2ec", fontsize=9)
        axes[-1].set_xlim(x_start_s, max(x_start_s + 0.01, x_end_s))
        if title:
            fig.suptitle(title, color="#eef3f8", fontsize=11)
        if last_mesh is not None:
            cbar = fig.colorbar(last_mesh, ax=axes, pad=0.015, aspect=40)
            cbar.ax.tick_params(colors="#b8c4d4", labelsize=8)
            cbar.set_label("dB", color="#d8e2ec", fontsize=9)

        fig.savefig(str(out), facecolor=fig.get_facecolor())
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
    )


def render_diff_spectrogram_png(
    before: np.ndarray,
    after: np.ndarray,
    sample_rate: int,
    out_path: str | Path,
    title: str = "",
    time_offset_s: float = 0.0,
) -> Path:
    """Render a 3-panel Before / After / Removed Residual spectrogram comparison."""
    arr_b = ensure_2d_audio(before)
    arr_a = ensure_2d_audio(after)
    n = min(arr_b.shape[0], arr_a.shape[0])
    mono_b = np.mean(arr_b[:n], axis=1)
    mono_a = np.mean(arr_a[:n], axis=1)
    mono_r = mono_b - mono_a

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
    n_fft = 2048
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
) -> Path:
    """Render a 3-panel diff PNG in bounded RAM (full <= 60s, overview > 60s)."""
    info_b = probe_audio(before_path)
    info_a = probe_audio(after_path)
    dur = min(info_b.duration_s, info_a.duration_s)
    s0 = max(0.0, float(start_s)) if start_s is not None else 0.0
    s1 = min(dur, float(end_s)) if end_s is not None else dur
    span = max(0.0, s1 - s0)

    if span <= 60.0:
        arr_b, sr = load_audio_slice(before_path, start_s=s0, end_s=s1, dtype="float64")
        arr_a, _ = load_audio_slice(after_path, start_s=s0, end_s=s1, dtype="float64")
        return render_diff_spectrogram_png(
            arr_b, arr_a, sr, out_path, title=title, time_offset_s=s0
        )

    n_cols = 700
    n_fft = 2048
    sr = info_b.sample_rate
    win = np.hanning(n_fft)
    win_norm = max(float(np.sum(win)), 1.0)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    col_times = np.linspace(s0, s1, num=n_cols, endpoint=False)
    s_b = np.empty((freqs.size, n_cols), dtype=np.float64)
    s_a = np.empty((freqs.size, n_cols), dtype=np.float64)
    s_r = np.empty((freqs.size, n_cols), dtype=np.float64)

    probe_dur = max(n_fft / float(sr), min(0.2, span / n_cols))
    for i, t_c in enumerate(col_times):
        t_end = min(s1, float(t_c) + probe_dur)
        cb, _ = load_audio_slice(before_path, start_s=float(t_c), end_s=t_end, dtype="float32")
        ca, _ = load_audio_slice(after_path, start_s=float(t_c), end_s=t_end, dtype="float32")
        n = min(cb.shape[0], ca.shape[0])
        if n == 0:
            s_b[:, i] = s_a[:, i] = s_r[:, i] = -120.0
            continue
        mb = np.mean(cb[:n], axis=1)
        ma = np.mean(ca[:n], axis=1)
        mr = mb - ma
        for mat, vec in ((s_b, mb), (s_a, ma), (s_r, mr)):
            seg = vec[:n_fft] if vec.size >= n_fft else np.pad(vec, (0, n_fft - vec.size))
            mag = np.abs(np.fft.rfft(seg * win)) / win_norm
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
    )
