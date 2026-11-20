"""Audio file I/O helpers enforcing 2D float64 (num_samples, num_channels) arrays."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def ensure_2d_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize an array to contiguous float64 shaped (num_samples, num_channels)."""
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    elif arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D audio array, got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float64)


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file as (num_samples, num_channels) float64, falling back to ffmpeg."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Audio file not found: {src}")

    try:
        data, sr = sf.read(str(src), always_2d=True, dtype="float64")
        return ensure_2d_audio(data), int(sr)
    except sf.LibsndfileError as exc:
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            raise RuntimeError(
                f"Cannot read '{src.name}' via libsndfile and ffmpeg is not on PATH."
            ) from exc

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_wav = Path(tmpdir) / "decoded.wav"
            cmd = [
                ffmpeg_bin,
                "-y",
                "-v",
                "error",
                "-i",
                str(src),
                "-f",
                "wav",
                "-acodec",
                "pcm_f32le",
                str(tmp_wav),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0 or not tmp_wav.exists():
                raise RuntimeError(
                    f"ffmpeg failed decoding '{src.name}': {proc.stderr.strip()}"
                ) from exc
            data, sr = sf.read(str(tmp_wav), always_2d=True, dtype="float64")
            return ensure_2d_audio(data), int(sr)


def save_wav(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    subtype: str = "FLOAT",
) -> Path:
    """Write audio to a WAV file (defaults to 32-bit float for session headroom)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = ensure_2d_audio(audio)
    sf.write(str(out), arr, int(sample_rate), subtype=subtype)
    return out


def export_audio(
    src_wav: str | Path,
    dst_path: str | Path,
    sample_rate: int,
    audio_format: str = "wav",
) -> Path:
    """Export a session take WAV to WAV (PCM_24), FLAC (PCM_24), MP3, or OGG."""
    fmt = audio_format.strip().lower()
    src = Path(src_wav)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if fmt not in {"wav", "flac", "mp3", "ogg"}:
        raise ValueError(f"Unsupported export format '{audio_format}'. Use wav, flac, mp3, or ogg.")

    audio, sr = load_audio(src)
    # Clamp peak gently only on integer PCM / compressed exports if exceeding +/- 1.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio * (0.999 / peak)

    if fmt == "wav":
        sf.write(str(dst), audio, sr, subtype="PCM_24")
        return dst
    if fmt == "flac":
        sf.write(str(dst), audio, sr, format="FLAC", subtype="PCM_24")
        return dst

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError(f"Exporting to '{fmt}' requires ffmpeg on PATH.")

    codec_args = (
        ["-c:a", "libmp3lame", "-b:a", "320k"]
        if fmt == "mp3"
        else ["-c:a", "libvorbis", "-q:a", "7"]
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_wav = Path(tmpdir) / "export_in.wav"
        save_wav(tmp_wav, audio, sr, subtype="FLOAT")
        cmd = [ffmpeg_bin, "-y", "-v", "error", "-i", str(tmp_wav), *codec_args, str(dst)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not dst.exists():
            raise RuntimeError(f"ffmpeg export failed: {proc.stderr.strip()}")
    return dst


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
