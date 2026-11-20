# Changelog

All notable changes to **AmbientSound** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Phase 2: CPU ONNX audio tagging / sound event detection (`YAMNet`, `PANNs`, `Silero VAD`) for semantic distraction classification.
- Phase 3: Text-queried target distraction separation (`AudioSep` / `CLAPSep`) behind license-aware model opt-in controls.
- Phase 4: Multi-take history DAG visualizer and optional external `ferraus` CLI backend adapter.

## [0.2.0] - 2026-11-20

### Added
- Standalone [instructions.md](instructions.md) handover manual (with preflight checks, long-file streaming guidance, hive/traffic notes, and YouTube/ASMR mastering recipes).
- Bounded-RAM streaming audio I/O (`probe_audio`, `load_audio_slice`, `iter_audio_blocks`, `ingest_audio_to_wav`, `render_wav_bytes`, streaming `export_audio`) safe for 1-hour / 672 MB+ WAV recordings.
- Streaming ITU-R BS.1770-4 LUFS (`StreamingLUFS`) and chunked 4x True Peak (`chunked_true_peak_linear` / `StreamingTruePeak`) plus `analyze_file` and `analyze_region`.
- Configurable distraction detection (`sensitivity`, `min_event_s`, `max_event_s`, `freq_range_hz`), `slow_pass` (`>= 3 s`) event classification, and arbitrary-frequency tonal `machine_drone` detection (`tonal_peaks_hz`).
- Six new CPU DSP filters (`drone_notch`, `trim_range`, `cut_regions`, `fade_edges`, `soft_compress`, `true_peak_limiter`) plus file-to-file `apply_recipe_streaming`.
- Multi-span batching (`spans`) and up to `5000 ms` fades/crossfades in `band_attenuate` (with `"swell"` envelope mode) and `ambience_patch`.
- Windowed (`start_s`/`end_s`) and low-RAM long-file overview spectrogram renderers (`render_file_spectrogram_png`, `render_file_diff_spectrogram_png`).
- New `health_check` and `analyze_region` MCP tools, plus windowed `/audio`, dynamic `/residual`, and `/spectrogram` endpoints with Windowed Audition Mode in the A/B Listening Studio (`static/index.html`).

## [0.1.0] - 2026-11-20

### Added
- Initial Apache-2.0 project structure (`pyproject.toml` with Python 3.12 `uv` environment, `LICENSE`, `NOTICE`, `.gitignore`, `CONTRIBUTING.md`, `AGENTS.md`).
- Project documentation suite: `README.md`, `docs/ARCHITECTURE.md`, `docs/GAPS_AND_DESIGN.md`, `docs/LICENSES.md`, `docs/RESEARCH.md`, `docs/ROADMAP.md`, `docs/MEMORY.md`, `docs/NOTES.md`.
- Core audio I/O (`src/ambient_sound/io_audio.py`) enforcing 2D `(num_samples, num_channels)` float64 arrays, 32-bit float WAV takes, and `ffmpeg` fallback for compressed formats.
- Psychoacoustic analyzer (`src/ambient_sound/analysis.py`) computing LUFS, true peak (4x oversampled), crest factor, stereo correlation, mid/side ratio, spectral features, 50/60 Hz hum prominence, rumble ratio, impulsive clicks, and transient distraction spans.
- Linked-stereo CPU DSP engine (`src/ambient_sound/dsp.py`) with 8 filters (`highpass_rumble`, `dehum_notch`, `declick_impulse`, `spectral_denoise`, `band_attenuate`, `ambience_patch`, `parametric_eq`, `loudness_normalize`) plus exact residual extraction.
- Headless spectrogram and 3-panel before/after/residual diff PNG renderer (`src/ambient_sound/visuals.py`).
- Non-destructive session/take DAG manager and attenuation-only LUFS-matched A/B test manager (`src/ambient_sound/session.py`).
- Path-sandboxed `stdio` MCP server (`src/ambient_sound/mcp_server.py`) exposing 11 tools via `MCPServer`.
- FastAPI + Web Audio API local A/B Listening Studio (`src/ambient_sound/web_ui.py`, `src/ambient_sound/static/index.html`) with synchronized instant switching, residual auditioning, loop spans, blind mode, artifact tags, and verdict feedback submission.
- CLI (`src/ambient_sound/cli.py`) with `mcp`, `ui`, `analyze`, and synthetic `demo` commands.
- Automated unit/integration tests (`tests/`), CI workflow (`.github/workflows/ci.yml`), and license/repo hygiene verification (`scripts/check_hygiene.py`).
