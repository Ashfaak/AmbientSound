"""Audio file I/O helpers supporting both 2D arrays and low-RAM streaming slices."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

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
        bytes_per_sample = 3
        est_bytes = info.frames * info.channels * bytes_per_sample + 4096
        out_fmt = ("RF64" if est_bytes >= 3_800_000_000 else "WAV") if fmt == "wav" else "FLAC"
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


class CompilationCancelledError(RuntimeError):
    """Raised when a streaming copy or span compilation is cooperatively cancelled."""


def copy_file_with_sha256(
    src_path: str | Path,
    dst_path: str | Path,
    *,
    block_bytes: int = 4 * 1024 * 1024,
    on_progress: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[int, str]:
    """Stream-copy a file atomically via `.partial` while hashing SHA-256 in a single pass."""
    src = Path(src_path)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = int(src.stat().st_size)
    chunk_n = max(65536, int(block_bytes))
    tmp_dst = dst.with_name(f".{dst.name}.{uuid4().hex[:6]}.partial")
    digest = hashlib.sha256()
    written = 0
    try:
        with src.open("rb") as fin, tmp_dst.open("wb") as fout:
            while True:
                if cancel_check is not None and cancel_check():
                    raise CompilationCancelledError("File copy cancelled.")
                buf = fin.read(chunk_n)
                if not buf:
                    break
                fout.write(buf)
                digest.update(buf)
                written += len(buf)
                if on_progress is not None:
                    on_progress(written, total_bytes)
        os.replace(tmp_dst, dst)
        return written, digest.hexdigest()
    except Exception:
        if tmp_dst.exists():
            tmp_dst.unlink(missing_ok=True)
        raise


def compile_audio_span(
    segments: list[tuple[str | Path, float]],
    start_s: float,
    end_s: float,
    dst_path: str | Path,
    *,
    audio_format: str = "wav",
    subtype: str = "PCM_24",
    fade_in_s: float = 2.0,
    fade_out_s: float = 3.0,
    fade_curve: str = "cosine",
    block_seconds: float = 30.0,
    on_block: Callable[[np.ndarray], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Stream-compile a [start_s, end_s] span across ordered audio files into one output."""
    if not segments:
        raise ValueError("compile_audio_span requires at least one segment.")
    fmt = audio_format.strip().lower()
    if fmt not in {"wav", "flac", "mp3", "ogg"}:
        raise ValueError(
            f"Unsupported export format '{audio_format}'. Use wav, flac, mp3, or ogg."
        )
    sub_u = subtype.strip().upper()
    if sub_u not in {"PCM_24", "PCM_16", "FLOAT"}:
        raise ValueError(f"Unsupported subtype '{subtype}'. Use PCM_24, PCM_16, or FLOAT.")
    curve = fade_curve.strip().lower()

    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    seg_meta: list[tuple[int, Path, int, int]] = []
    sr: int | None = None
    ch: int | None = None
    cum_frames = 0
    for seg_idx, (seg_p_raw, _) in enumerate(segments):
        seg_p = Path(seg_p_raw)
        info = probe_audio(seg_p)
        if sr is None:
            sr, ch = info.sample_rate, info.channels
        elif info.sample_rate != sr or info.channels != ch:
            raise ValueError(
                f"Segment format mismatch for {seg_p.name}: "
                f"{info.sample_rate}Hz/{info.channels}ch vs {sr}Hz/{ch}ch."
            )
        seg_meta.append((seg_idx, seg_p, cum_frames, cum_frames + info.frames))
        cum_frames += info.frames

    assert sr is not None and ch is not None
    req0 = max(0, int(round(float(start_s) * sr)))
    req1 = min(cum_frames, int(round(float(end_s) * sr)))
    if req1 <= req0:
        raise ValueError(f"Requested compile window [{start_s}, {end_s}] is empty.")

    slices: list[tuple[int, Path, int, int, float, float, float, float]] = []
    for seg_idx, seg_p, g0, g1 in seg_meta:
        ov0 = max(req0, g0)
        ov1 = min(req1, g1)
        if ov1 <= ov0:
            continue
        loc0 = ov0 - g0
        take_n = ov1 - ov0
        slices.append(
            (
                seg_idx,
                seg_p,
                loc0,
                take_n,
                round(loc0 / float(sr), 3),
                round((loc0 + take_n) / float(sr), 3),
                round(ov0 / float(sr), 3),
                round(ov1 / float(sr), 3),
            )
        )

    total_frames = sum(s[3] for s in slices)
    fade_in_n = min(total_frames // 2, max(0, int(round(max(0.0, float(fade_in_s)) * sr))))
    fade_out_n = min(
        total_frames - fade_in_n, max(0, int(round(max(0.0, float(fade_out_s)) * sr)))
    )
    out_start = total_frames - fade_out_n
    block_frames = max(4096, int(round(max(1.0, float(block_seconds)) * sr)))
    bytes_per_sample = 4 if sub_u == "FLOAT" else (2 if sub_u == "PCM_16" else 3)
    est_bytes = total_frames * ch * bytes_per_sample + 4096

    def _stream_into(target_wav: Path, container: str, out_subtype: str) -> int:
        written = 0
        with sf.SoundFile(
            str(target_wav),
            mode="w",
            samplerate=sr,
            channels=ch,
            subtype=out_subtype,
            format=container,
        ) as fout:
            global_cursor = 0
            for _, seg_p, loc0, take_n, _, _, _, _ in slices:
                with sf.SoundFile(str(seg_p), mode="r") as fin:
                    fin.seek(loc0)
                    rem = take_n
                    while rem > 0:
                        if cancel_check is not None and cancel_check():
                            raise CompilationCancelledError("Compilation cancelled.")
                        blk = fin.read(min(block_frames, rem), always_2d=True, dtype="float64")
                        if blk.size == 0:
                            break
                        n_b = blk.shape[0]
                        b_end = global_cursor + n_b
                        if fade_in_n > 0 and global_cursor < fade_in_n:
                            lo = global_cursor
                            hi = min(b_end, fade_in_n)
                            t = np.arange(lo, hi, dtype=np.float64) / max(fade_in_n - 1, 1)
                            ramp = t if curve == "linear" else 0.5 * (1.0 - np.cos(math.pi * t))
                            blk[: hi - lo] *= ramp[:, np.newaxis]
                        if fade_out_n > 0 and b_end > out_start:
                            lo = max(global_cursor, out_start)
                            hi = b_end
                            t = 1.0 - (np.arange(lo, hi, dtype=np.float64) - out_start) / max(
                                fade_out_n - 1, 1
                            )
                            ramp = t if curve == "linear" else 0.5 * (1.0 - np.cos(math.pi * t))
                            blk[lo - global_cursor : hi - global_cursor] *= ramp[:, np.newaxis]
                        if out_subtype != "FLOAT":
                            blk = np.clip(blk, -1.0, 0.999999)
                        if on_block is not None:
                            on_block(blk)
                        fout.write(blk)
                        global_cursor += n_b
                        written += n_b
                        rem -= n_b
                        if on_progress is not None:
                            on_progress(written, total_frames)
        return written

    if fmt in {"wav", "flac"}:
        container = ("RF64" if est_bytes >= 3_800_000_000 else "WAV") if fmt == "wav" else "FLAC"
        tmp_dst = dst.with_name(f".{dst.name}.{uuid4().hex[:6]}.partial")
        try:
            written = _stream_into(tmp_dst, container, sub_u if fmt == "wav" else "PCM_24")
            os.replace(tmp_dst, dst)
        except Exception:
            if tmp_dst.exists():
                tmp_dst.unlink(missing_ok=True)
            raise
    else:
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            raise RuntimeError(f"Exporting to '{fmt}' requires ffmpeg on PATH.")
        container = fmt.upper()
        tmp_dst = dst.with_name(f".{dst.stem}.{uuid4().hex[:6]}.partial.{fmt}")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_wav = Path(tmpdir) / "compile_tmp.wav"
                written = _stream_into(tmp_wav, "RF64", "FLOAT")
                codec_args = (
                    ["-c:a", "libmp3lame", "-b:a", "320k"]
                    if fmt == "mp3"
                    else ["-c:a", "libvorbis", "-q:a", "7"]
                )
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(tmp_wav),
                    *codec_args,
                    str(tmp_dst),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if proc.returncode != 0 or not tmp_dst.exists():
                    raise RuntimeError(f"ffmpeg export failed: {proc.stderr.strip()}")
            os.replace(tmp_dst, dst)
        except Exception:
            if tmp_dst.exists():
                tmp_dst.unlink(missing_ok=True)
            raise

    return {
        "output_path": str(dst),
        "format": fmt,
        "container": container,
        "subtype": sub_u if fmt == "wav" else fmt.upper(),
        "size_bytes": int(dst.stat().st_size),
        "duration_s": round(written / float(sr), 3),
        "sample_rate": int(sr),
        "channels": int(ch),
        "sha256": sha256_file(dst),
        "source_slices": [
            {
                "segment_index": seg_idx,
                "file": p.name,
                "local_start_s": l0,
                "local_end_s": l1,
                "local_start_sample": int(loc0),
                "local_end_sample": int(loc0 + take_n),
                "global_start_s": g0_s,
                "global_end_s": g1_s,
            }
            for seg_idx, p, loc0, take_n, l0, l1, g0_s, g1_s in slices
        ],
    }
