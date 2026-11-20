"""Audio file I/O helpers supporting both 2D arrays and low-RAM streaming slices."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import soundfile as sf

from ambient_sound.models import AudioInfo


def ensure_2d_audio(audio: np.ndarray, dtype: np.dtype | type | str = np.float64) -> np.ndarray:
    """Normalize an array to contiguous floating-point shaped (num_samples, num_channels)."""
    arr = np.asarray(audio, dtype=dtype)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    elif arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D audio array, got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=dtype)


def probe_audio(path: str | Path) -> AudioInfo:
    """Inspect container metadata without loading sample buffers into RAM."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Audio file not found: {src}")

    size_bytes = src.stat().st_size
    try:
        info = sf.info(str(src))
        return AudioInfo(
            sample_rate=int(info.samplerate),
            channels=int(info.channels),
            frames=int(info.frames),
            duration_s=round(float(info.duration), 6),
            subtype=str(info.subtype),
            format_name=str(info.format),
            size_bytes=int(size_bytes),
        )
    except sf.LibsndfileError as exc:
        ffprobe_bin = shutil.which("ffprobe")
        if ffprobe_bin is None:
            raise RuntimeError(
                f"Cannot inspect '{src.name}' via libsndfile and ffprobe is not on PATH."
            ) from exc

        cmd = [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,duration,codec_name:format=format_name,duration",
            "-of",
            "json",
            str(src),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffprobe failed inspecting '{src.name}': {proc.stderr.strip()}"
            ) from exc
        payload = json.loads(proc.stdout)
        streams = payload.get("streams") or []
        if not streams:
            raise RuntimeError(f"No audio stream found in '{src.name}'.") from exc
        st = streams[0]
        fmt = payload.get("format") or {}
        sr = int(st["sample_rate"])
        ch = int(st["channels"])
        dur = float(st.get("duration") or fmt.get("duration") or 0.0)
        return AudioInfo(
            sample_rate=sr,
            channels=ch,
            frames=int(round(dur * sr)),
            duration_s=round(dur, 6),
            subtype=str(st.get("codec_name") or "unknown"),
            format_name=str(fmt.get("format_name") or src.suffix.lstrip(".")),
            size_bytes=int(size_bytes),
        )


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


def load_audio_slice(
    path: str | Path,
    start_s: float = 0.0,
    end_s: float | None = None,
    dtype: str = "float64",
) -> tuple[np.ndarray, int]:
    """Random-access read of [start_s, end_s] without loading the whole file into RAM."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Audio file not found: {src}")

    s0 = max(0.0, float(start_s))
    try:
        with sf.SoundFile(str(src), mode="r") as f:
            sr = int(f.samplerate)
            ch = int(f.channels)
            total_frames = int(f.frames)
            start_frame = min(total_frames, int(round(s0 * sr)))
            end_frame = (
                total_frames
                if end_s is None
                else max(start_frame, min(total_frames, int(round(float(end_s) * sr))))
            )
            n_frames = end_frame - start_frame
            if n_frames <= 0:
                return np.zeros((0, ch), dtype=dtype), sr
            f.seek(start_frame)
            data = f.read(n_frames, always_2d=True, dtype=dtype)
            return ensure_2d_audio(data, dtype=dtype), sr
    except sf.LibsndfileError as exc:
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            raise RuntimeError(
                f"Cannot slice '{src.name}' via libsndfile and ffmpeg is not on PATH."
            ) from exc

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_wav = Path(tmpdir) / "slice.wav"
            cmd = [ffmpeg_bin, "-y", "-v", "error", "-ss", f"{s0:.6f}"]
            if end_s is not None:
                dur = max(0.0, float(end_s) - s0)
                if dur <= 0.0:
                    info = probe_audio(src)
                    return np.zeros((0, info.channels), dtype=dtype), info.sample_rate
                cmd.extend(["-t", f"{dur:.6f}"])
            cmd.extend(["-i", str(src), "-f", "wav", "-acodec", "pcm_f32le", str(tmp_wav)])
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0 or not tmp_wav.exists():
                raise RuntimeError(
                    f"ffmpeg failed slicing '{src.name}': {proc.stderr.strip()}"
                ) from exc
            data, sr = sf.read(str(tmp_wav), always_2d=True, dtype=dtype)
            return ensure_2d_audio(data, dtype=dtype), int(sr)


def iter_audio_blocks(
    path: str | Path,
    block_seconds: float = 30.0,
    start_s: float = 0.0,
    end_s: float | None = None,
    dtype: str = "float32",
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (start_frame, block_2d) chunks sequentially from a libsndfile-readable file."""
    src = Path(path)
    with sf.SoundFile(str(src), mode="r") as f:
        sr = int(f.samplerate)
        total_frames = int(f.frames)
        s_frame = max(0, min(total_frames, int(round(max(0.0, float(start_s)) * sr))))
        e_frame = (
            total_frames
            if end_s is None
            else max(s_frame, min(total_frames, int(round(float(end_s) * sr))))
        )
        block_frames = max(256, int(round(float(block_seconds) * sr)))
        f.seek(s_frame)
        pos = s_frame
        while pos < e_frame:
            to_read = min(block_frames, e_frame - pos)
            chunk = f.read(to_read, always_2d=True, dtype=dtype)
            if chunk.size == 0:
                break
            yield pos, ensure_2d_audio(chunk, dtype=dtype)
            pos += chunk.shape[0]


def ingest_audio_to_wav(src_path: str | Path, dst_wav: str | Path) -> AudioInfo:
    """Ingest an input file into a canonical WAV take without loading full arrays into RAM."""
    src = Path(src_path)
    dst = Path(dst_wav)
    if not src.exists():
        raise FileNotFoundError(f"Audio file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        info = sf.info(str(src))
        if src.suffix.lower() == ".wav" and str(info.format).upper() == "WAV":
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            return probe_audio(dst)

        with (
            sf.SoundFile(str(src), mode="r") as fin,
            sf.SoundFile(
                str(dst),
                mode="w",
                samplerate=int(info.samplerate),
                channels=int(info.channels),
                subtype="PCM_24",
                format="WAV",
            ) as fout,
        ):
            block_frames = max(4096, int(info.samplerate) * 30)
            while True:
                chunk = fin.read(block_frames, always_2d=True, dtype="float32")
                if chunk.size == 0:
                    break
                fout.write(chunk)
        return probe_audio(dst)
    except sf.LibsndfileError as exc:
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            raise RuntimeError(
                f"Cannot ingest '{src.name}' via libsndfile and ffmpeg is not on PATH."
            ) from exc
        cmd = [
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-i",
            str(src),
            "-f",
            "wav",
            "-c:a",
            "pcm_s24le",
            str(dst),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not dst.exists():
            raise RuntimeError(
                f"ffmpeg failed ingesting '{src.name}': {proc.stderr.strip()}"
            ) from exc
        return probe_audio(dst)


def save_wav(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    subtype: str = "FLOAT",
) -> Path:
    """Write audio to a WAV file (defaults to 32-bit float for short-take headroom)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = ensure_2d_audio(audio)
    sf.write(str(out), arr, int(sample_rate), subtype=subtype)
    return out


def render_wav_bytes(
    audio: np.ndarray,
    sample_rate: int,
    subtype: str = "PCM_24",
) -> bytes:
    """Encode an in-memory audio slice as WAV bytes for sliced Web UI streaming."""
    arr = ensure_2d_audio(audio)
    if arr.size and subtype != "FLOAT":
        peak = float(np.max(np.abs(arr)))
        if peak > 1.0:
            arr = arr * (0.999 / peak)
    buf = io.BytesIO()
    sf.write(buf, arr, int(sample_rate), format="WAV", subtype=subtype)
    return buf.getvalue()


def export_audio(
    src_wav: str | Path,
    dst_path: str | Path,
    sample_rate: int,
    audio_format: str = "wav",
) -> Path:
    """Stream-export a session take WAV to WAV (PCM_24), FLAC (PCM_24), MP3, or OGG."""
    del sample_rate  # Kept for signature compatibility; actual rate is read from src_wav.
    fmt = audio_format.strip().lower()
    src = Path(src_wav)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if fmt not in {"wav", "flac", "mp3", "ogg"}:
        raise ValueError(
            f"Unsupported export format '{audio_format}'. Use wav, flac, mp3, or ogg."
        )

    info = probe_audio(src)
    peak = 0.0
    for _, block in iter_audio_blocks(src, block_seconds=30.0, dtype="float32"):
        if block.size:
            b_peak = float(np.max(np.abs(block)))
            if b_peak > peak:
                peak = b_peak
    scale = (0.999 / peak) if peak > 1.0 else 1.0

    if fmt in {"wav", "flac"}:
        out_fmt = "WAV" if fmt == "wav" else "FLAC"
        with sf.SoundFile(
            str(dst),
            mode="w",
            samplerate=info.sample_rate,
            channels=info.channels,
            subtype="PCM_24",
            format=out_fmt,
        ) as fout:
            for _, block in iter_audio_blocks(src, block_seconds=30.0, dtype="float64"):
                if scale != 1.0:
                    block = block * scale
                fout.write(block)
        return dst

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError(f"Exporting to '{fmt}' requires ffmpeg on PATH.")

    codec_args = (
        ["-c:a", "libmp3lame", "-b:a", "320k"]
        if fmt == "mp3"
        else ["-c:a", "libvorbis", "-q:a", "7"]
    )
    af_args = ["-af", f"volume={scale:.8f}"] if scale != 1.0 else []
    cmd = [ffmpeg_bin, "-y", "-v", "error", "-i", str(src), *af_args, *codec_args, str(dst)]
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
