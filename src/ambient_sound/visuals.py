"""Headless waveform and spectrogram PNG renderers for AmbientSound."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from ambient_sound.io_audio import ensure_2d_audio
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


def render_spectrogram_png(
    audio: np.ndarray,
    sample_rate: int,
    out_path: str | Path,
    title: str = "",
    events: list[DistractionEvent] | None = None,
) -> Path:
    """Render a dark-themed waveform + log-frequency spectrogram PNG with event overlays."""
    arr = ensure_2d_audio(audio)
    mono = np.mean(arr, axis=1)
    duration_s = mono.size / float(max(sample_rate, 1))
    freqs, times, spec_db = _compute_spectrogram_db(mono, sample_rate)

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
    )
    try:
        fig.patch.set_facecolor("#0f141c")
        for ax in (ax_wav, ax_spec):
            ax.set_facecolor("#161e29")
            ax.tick_params(colors="#b8c4d4", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#2a3647")

        # Downsample waveform display cleanly for fast plotting
        if mono.size > 0:
            step = max(1, mono.size // 4000)
            t_wav = np.linspace(0.0, duration_s, num=mono[::step].size)
            ax_wav.plot(t_wav, mono[::step], color="#59c3c3", linewidth=0.7)
            peak = max(0.05, float(np.max(np.abs(mono))) * 1.1)
            ax_wav.set_ylim(-peak, peak)
        ax_wav.set_ylabel("Amp", color="#d8e2ec", fontsize=9)
        ax_wav.set_title(title or "Ambient Recording Overview", color="#eef3f8", fontsize=11)

        if events:
            palette = {
                "low_freq_rumble": "#f59e0b",
                "impulsive_click": "#ef4444",
                "transient_distraction": "#f43f5e",
                "clipping": "#dc2626",
            }
            for ev in events:
                if ev.kind == "mains_hum":
                    continue
                color = palette.get(ev.kind, "#f43f5e")
                ax_wav.axvspan(ev.start_s, ev.end_s, color=color, alpha=0.22)

        # Plot spectrogram excluding DC bin for clean log y-scale
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
        ax_spec.set_xlim(0.0, max(0.01, duration_s))

        fig.tight_layout()
        fig.savefig(str(out), facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
    return out


def render_diff_spectrogram_png(
    before: np.ndarray,
    after: np.ndarray,
    sample_rate: int,
    out_path: str | Path,
    title: str = "",
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

    vmax = float(max(np.max(s_b), np.max(s_a), -40.0))
    vmin = vmax - 85.0
    duration_s = n / float(max(sample_rate, 1))

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
        pos_mask = f_b >= 20.0
        f_plot = f_b[pos_mask] if np.any(pos_mask) else f_b[1:]

        last_mesh = None
        for ax, spec, subtitle in panels:
            ax.set_facecolor("#161e29")
            ax.tick_params(colors="#b8c4d4", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#2a3647")
            s_plot = spec[pos_mask, :] if np.any(pos_mask) else spec[1:, :]
            if f_plot.size >= 2 and t_b.size >= 2:
                last_mesh = ax.pcolormesh(
                    t_b,
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
        axes[-1].set_xlim(0.0, max(0.01, duration_s))
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
