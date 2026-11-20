# Changelog

All notable changes to **AmbientSound** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Phase 2: CPU ONNX audio tagging / sound event detection (`YAMNet`, `PANNs`, `Silero VAD`) for semantic distraction classification.
- Phase 3: Text-queried target distraction separation (`AudioSep` / `CLAPSep`) behind license-aware model opt-in controls.
- Phase 4: Multi-take history DAG visualizer and optional external `ferraus` CLI backend adapter.

## [0.3.0] - 2026-11-20

### Added
- Dedicated `gain` DSP filter (`-48` to `+48 dB`, streamable) plus `limit_peaks` / `limiter_release_ms` on `loudness_normalize` (in-memory & streaming) so quiet, spiky field recordings (`-50 LUFS`) can consistently master to `-14 LUFS` at `<= -1.0 dBTP`.
- `band_attenuate(envelope_mode="adaptive")` scaling spectral cut dynamically per frame above clean ambient floor, hitting peak attenuation where the distraction is actually loudest without carving hollow gaps on quiet edges.
- `StepTelemetry` (`step_metrics`) per-step RMS/peak/LUFS metrics and `ambience_patch` donor provenance (`target_start_s`, `target_end_s`, `donor_start_s`, `donor_end_s`, per-span `crossfade_ms`) recorded across both in-memory and streaming pipelines.
- EBU R128 Loudness Range (`loudness_range_lu`), 10th-percentile ambient `noise_floor_dbfs`, short-term 3 s LUFS statistics (`short_term_lufs_stats`), and downsampled `short_term_lufs_timeline` in `AudioMetrics`.
- Per-take full event archival (`takes/take_NNN/events.json`), `event_counts`, and compact bounded MCP event parameters (`event_kinds`, `max_events=50`, `include_events`, `events_total`, `events_truncated`) prioritizing macro distractions ahead of impulsive clicks.
- Disk cleanup tools `delete_take(session_id, take_id, cascade=False)` (preserving immutable `take_000` and cleaning orphaned A/B tests with non-colliding IDs) and `delete_session(session_id)`.
- Candidate-first A/B loudness matching (`match_original=False` default, opt-in `match_original=True`), earliest-timeline-root `original_take_id` auto-resolution across `trim_range` previews, and loudness-matched residual playback / diff panels when gain shifts exceed `1.5 dB`.
- `"Patched"` region shading/labels in diff spectrograms and higher-resolution `n_fft=8192` spectrograms (`~5.86 Hz` bins at `48 kHz`) that suppress dense waveform click walls (`> 40` clicks in view).

### Fixed
- Lowered distraction detector amplitude floors (`-55`/`-60` -> `-92.0 dBFS`), wired `sensitivity` into low-frequency rumble detection, widened merge gaps (`0.50 s` foreground, `0.40 s` rumble), and gated `impulsive_click` output when `min_event_s > 0.15 s`.
- Replaced one-sided neighbor medians with bilateral sideband prominence in `_detect_hum_peaks` and `_detect_tonal_peaks` (eliminating false `50 Hz` hum detections caused by steep `45 Hz` high-pass filters) and lowered tonal prominence to `> 7.0 dB` with `>= 60%` multi-segment persistence to separate steady machine drones from short passing vehicles.

## [0.2.1] - 2026-11-20

### Fixed
- Default `AMBIENTSOUND_WORKSPACE` now resolves to `~/.ambientsound` instead of `CWD/.ambientsound`, preventing session-list mismatches when the MCP server and Web UI launch from different working directories.
- Added `--workspace <path>` to `ambient-sound-ui` / `ambient-sound ui`, added `/api/info`, included `workspace_dir` in A/B MCP tool responses, and surfaced a clear missing-session banner in the A/B Studio UI instead of silently falling back to the demo session.

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
