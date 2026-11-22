# Changelog

All notable changes to **AmbientSound** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Phase 2: CPU ONNX audio tagging / sound event detection (`YAMNet`, `PANNs`, `Silero VAD`) behind optional extra/license gates (`AMBIENTSOUND_ALLOW_NONCOMMERCIAL=1`).
- Phase 3: Text-queried target distraction separation (`AudioSep` / `CLAPSep`) behind license-aware model opt-in controls.
- Phase 4: Multi-take history DAG visualizer and optional external `ferraus` CLI backend adapter.

## [0.5.3] - 2026-11-22

### Fixed
- **Gain-Invariant Piezo Beep Detection & Narrowband Protection (`engines.py`):**
  - Replaced the fixed `-88.0 dBFS` floor inside `detect_synthetic_beeps` with a recording-relative bed threshold (`max(-110.0, bed_p25 + 8.0)`) and interior-frame stability evaluation (`dur_s >= 0.045`, `len >= 4`), eliminating gain-amplified false positives after normalizing while catching real `75–120 ms` watch piezo beeps (`~2982 Hz`).
  - Updated `compute_bird_protect_mask` to evaluate zero-drift piezo rejection down to `2+` frame components and across interior frames so `48 kHz` `n_fft=4096` `spectral_match_suppress(narrowband_mode=True, protect_birds=True)` suppresses `90 ms` piezo tones instead of partially guarding them.
- **Filter Schema & `10 min` Clean Target Completeness (`dsp.py`, `analysis.py`):**
  - Exposed `"bark_distance"` in `distance_soften`'s `list_filters` `mode.choices` schema.
  - Enabled `"10 min"` ASMR clean-section targets on all recordings `>= 10 min` (previously omitted once duration exceeded 30 min).

## [0.5.2] - 2026-11-21

### Fixed
- **Gain-Normalized Bird Retention & Parent Event Offsets (`engines.py`, `mcp_server.py`):**
  - Normalized `compute_comparison_guardrails` (`bird_retention_db`) against overall RMS gain difference (`gain_offset_db`, plus unnormalized `bird_retention_raw_db`) so `+37 dB` master boost never masks real bird-energy loss below the `-1.5 dB` warning threshold.
  - Added `source_offset_s` on `analyze_take` / `analyze_region` payloads and `parent_start_s` / `parent_end_s` on serialized events whenever analyzing trimmed child takes (`source_offset_s > 0`).
- **File-Local Minute Snapping Across All Clean Tiers & Sub-Second Precision (`analysis.py`):**
  - Fixed `round_to_file_local_minutes=True` tie-breaking in `compute_clean_sections` so file-local whole minutes strictly outrank global minutes across `untouched`, `minimal_edits`, and `moderate_edits` tiers alike.
  - Preserved millisecond precision (`start_s`, `end_s`, `start_local_s`, `end_local_s`, `edited_s`, `residual_distraction_s`) instead of `0.1 s` rounding, eliminating `~30 ms` local offset drift; added `eligible` and `violations[]` reasons on `asmr_targets`.
- **Clean-Section Provenance, Prefix Naming & Non-Overlapping Selection (`session.py`, `io_audio.py`):**
  - Applied explicit `filename_prefix` consistently to full masters (`{prefix}_{stem}_master.{ext}`, tracking `"prefix_applied": bool`) in both `compile_clean_sections` and `process_sequence`.
  - Populated `source_slices[]` with real `source_file`, `session_id`, `take_id`, `segment_index`, `local_start_sample`, `local_end_sample`, and `global_start_s`/`global_end_s` (never generic `"audio.wav"`).
  - Added `distinct_regions=True` (via `compute_clean_sections(..., exclude_spans=...)`) so multi-target compilations never emit overlapping stretches when separate gaps exist.
  - Added `bed_shape` headroom warning in `process_take` telemetry/notes whenever post-shaping `true_peak_dbtp > -1.05` without a trailing limiter.

### Changed & Added
- **Single-Pass Atomic Exports, Progress Heartbeats & `wait_for_job` (`io_audio.py`, `jobs.py`, `mcp_server.py`, `models.py`):**
  - Added `copy_file_with_sha256` with atomic `.partial` sibling writes and reused cached `TakeRecord` LUFS/peak metrics when exporting unmodified WAV takes (`metrics_source="cached_take_metrics"`), cutting multi-GB master export I/O in half.
  - Added intra-item byte/frame progress heartbeats (`item_progress_pct`, `phase`, `bytes_written`, `bytes_total`, `elapsed_s`, `eta_s`), structured `items_summary` (`{label, kind, state, detail, session_id, take_id, output_path, size_bytes}`), startup `reconcile_stale_jobs`, and active `output_dir` job collision locks.
  - Added `wait_for_job` MCP tool (`26` tools total, reported by `health_check.tool_count`), defaulted `list_jobs(..., include_result=False)` / optional `get_job_status(..., include_result=False)` with `has_result` + `manifest_path`, and enriched `clean_sections_manifest.json` with `tool_version`, `created_at`, `parameters`, `sequence`, `masters_count`, `total_bytes`, `warnings`, `seams`, and `events`.

## [0.5.1] - 2026-11-21

### Added
- **Clean ASMR Section Compilation & Unified Output Folder:**
  - Added streaming `compile_audio_span(...)` in `io_audio.py` with frame-accurate single-file slicing and seamless cross-seam multi-file stitching, configurable cosine/linear global fade-in/out, single-pass `on_block` callback, SHA-256 verification, and `source_slices` provenance.
  - Added automatic `RF64` WAV container switching in `export_audio` and `compile_audio_span` whenever estimated WAV output size reaches `3.8 GB` (`>= 3_800_000_000` bytes), preventing 4 GiB RIFF header wraparound on multi-hour masters and combined ASMR exports.
  - Added `SessionManager.compile_clean_sections(...)` and MCP tools `compile_clean_sections` + async `start_compile_clean_sections_job` (raising total MCP tools from 23 to 25), writing clean section renders under `<output_dir>/clean_sections/`, optional full masters (`<output_dir>/<stem>_master.<ext>`) and stitched `<output_dir>/<prefix>_full_combined.<ext>`, and atomic `<output_dir>/clean_sections_manifest.json` with strict `allowed_roots`, `sessions_root`, and overwrite guards.
  - Upgraded `SessionManager.process_sequence(...)`, `JobManager.start_sequence_job(...)`, and MCP `process_sequence` / `start_sequence_job` with optional `output_dir`, `compile_clean_targets`, `compile_clean_tiers`, `include_combined_full`, `clean_fade_in_s`, `clean_fade_out_s`, and `filename_prefix`.

## [0.5.0] - 2026-11-21

### Added
- **Shared Pure-NumPy/SciPy Engine Layer (`src/ambient_sound/engines.py`):**
  - Unified channel-layout detection (`detect_channel_layout`, `detect_channel_layout_from_stats`), GCC-PHAT sub-sample lag estimation (`estimate_gcc_phat_lag`), windowed-sinc fractional delay (`fractional_shift_channel`), masked least-squares FIR matching (`fit_least_squares_fir`), clipping mask detection, and bidirectional Autoregressive gap interpolation (`ar_interpolate_gap`).
  - Content-defined bird protection (`compute_bird_protect_mask`) across `300 Hz – 11.5 kHz` that preserves FM birdsong syllables/trills while rejecting zero-drift synthetic piezo beeps via sub-bin parabolic frequency flatness.
  - Variable-RPM harmonic comb tracker (`track_harmonic_comb`, `build_harmonic_comb_mask`) and unified excess-capped, context-gapped (`context_gap_s`), bird-protected STFT transform engine (`apply_masked_stft_transform`).
- **New Outdoor & Nature Distraction Detectors (`analysis.py`):**
  - Added 8 new `DistractionEvent.kind` categories: `synthetic_beep` (zero-drift piezo/watch tones with cross-block fingerprinting), `click_train` (rapid mechanical zips/cable ties), `crackle_burst` (plastic bags/foliage rustle), `human_voice` (syllable-modulated harmonic formants), `dog_bark` (percussive broadband-harmonic barks), `variable_rpm_engine` (sweeping leafblowers/trimmers), `channel_biased_noise` (one-sided machinery/vehicles), and `wind_noise` (uncorrelated `<160 Hz` turbulence), plus confidence scores and approach/tail `+1.5 dB` hysteresis.
- **Psychoacoustic Pleasantness, Hardware Layout & Band Telemetry (`AudioMetrics`):**
  - Added `sharpness_acum`, `roughness_index`, `ndsi`, `spectral_slope_db_per_oct`, `channel_balance_db`, `stereo_layout_hint` (`mono`, `dual_mono`, `raw_ms`, `directional_plus_surround`, `stereo_lr`, `ambiguous`), `stereo_layout_confidence`, `pleasantness_score`, `band_metrics` (full/P10/peak-excess across standard + custom bands), `compare_takes` deltas (`bird_retention_db`, `musical_noise_index`, `guardrail_warnings`), and `assess_do_no_harm` (`cleanliness_assessment`).
- **5 New Restoration & Polishing Filters (`21` Total, In-Memory & Streaming):**
  - `declip_from_backup`: Repairs clipped peaks from a safety `-12 dB` WAV (`backup_audio_path` validated via `ensure_path_allowed`) using GCC-PHAT sub-sample alignment, LS-FIR calibration, raised-cosine splicing, and AR fallback when both tracks clip.
  - `spatial_mix`: Auto/explicit handling for `ms_decode` (Zoom SSH-6 Mid/Side), `dual_mic_blend` (directional SGH-6 + surround with GCC-PHAT delay alignment & crossover), `dual_mono_to_stereo` (mono->stereo decorrelation in both in-memory & streaming), and `lr_adjust` (`auto_balance`, `low_mono_hz`, `side_hpf_hz`).
  - `harmonic_track_suppress`: Sweeping `f0(t)` comb suppression for variable-RPM engines/leafblowers with optional L/R bias focus and FM bird protection.
  - `distance_soften`: Pushes intrusive `dog_bark` (`bark_distance`) or `human_voice` (`voice_muffle`, formant blur + HF air absorption + attack smearing) into the far distance instead of carving hollow gaps.
  - `bed_shape`: Slow-bed comfort shaper (4 s sliding floor, warmth tilt, dynamic 2–5 kHz de-harsh, floor preservation, optional gentle bed width boost) leaving FM birdsong & transients untouched.
  - Upgraded `declick_impulse` (`spans`, `method="pchip"|"ar"`, `min_flatness`), `spectral_match_suppress` (`protect_birds`, `spatial_mode="channel_excess"`, `narrowband_mode`, `context_gap_s`), and `ambience_patch` v3 (`min_donor_distance_s`, `max_donor_crest` knock screening, `loop_count`, `donor_chunks`).
- **Background Job Manager & 4 Async Sequence MCP Tools (`23` Tools Total):**
  - Added `src/ambient_sound/jobs.py` (`JobManager`) and `start_sequence_job`, `get_job_status`, `cancel_job`, `list_jobs` so multi-hour split sequences run cooperatively in background threads without MCP timeouts.
  - Added `GET /api/jobs`, 5 nature/outdoor quality tag chips, and NDSI/Sharpness/Balance/Layout/Pleasantness displays in the A/B Studio (`web_ui.py` / `static/index.html`), plus palette colors for all new event kinds in `visuals.py`.

### Fixed
- **Seam Slew-Normalized Step Gate & Clean Sections:** Added `max_sample_step_excess_db=18.0` to `verify_seams` so normal loud broadband `-16 LUFS` splits pass while true step discontinuities fail; added `round_to_file_local_minutes` in `find_clean_sections`.
- **Short-Clip Tonal Persistence & Macro Event Budgeting:** Lowered multi-segment Welch tonal persistence gating from `15 s` down to `>= 2.0 s` (`analysis.py`) to eliminate false `machine_drone` labels from brief bird chirps; added `include_clicks` auto-hiding (`> 25` clicks hidden from macro event lists unless explicitly requested) and `compact_takes` / `take_ids` on `get_session_summary`.
- **Thread-Safe Session Locking & Verified Rich Exports:** Added per-session `threading.RLock` + atomic `session.json` replacements, `seam_padding` provenance tracking, and `export_take(..., discard_take_audio=...)` returning size, duration, loudness/TP, and SHA-256 while protecting immutable `take_000`.

## [0.4.0] - 2026-11-21

### Added
- **A/B Listening Studio Scrolling Playhead & A vs B Diff (`web_ui.py` / `visuals.py` / `static/index.html`):**
  - `_plot_spectrogram_figure` and `_plot_diff_figure` emit sidecar plot-bounds metadata (`.meta.json`: `x0, x1, top, bottom, t_start, t_end`), exposed via HTTP `X-Plot-*` and `X-Time-*` headers so the live synchronized `#playheadLine` cursor scrolls smoothly across waveforms and spectrograms, supports click/drag seeking, and renders loop highlight regions directly over the active figure.
  - Added `/api/sessions/{session_id}/ab-tests/{ab_id}/diff-spectrogram?side=a|b|ab` with timeline-aligned Before/After/Removed and direct **A vs B Diff** inspection, fixing 404s on `trim_range` previews whose `diff_spectrogram_path` was `None`.
- **`spectral_match_suppress` (#16 DSP Filter, In-Memory & Streaming):**
  - STFT (`n_fft=4096`) per-bin adaptive spectral excess suppression against clean surrounding (or manual `ref_start_s`/`ref_end_s`) bed percentiles with smooth normalized 2D Gaussian attenuation and subtle clean-context texture fill (`texture_fill`, `strength`, `max_attenuation_db`), attenuating moving vehicle exhaust harmonics and temporary overlap tones while leaving non-excess hive/nature frequencies intact.
- **`ambience_patch` v2 Non-Repeating Event-Screened Donors:**
  - Single-pass `O(N)` streaming stats scan scoring full contiguous `[p0, p1]` windows by RMS match, variance CV, crest factor, sub-120 Hz rumble ratio, and proximity; tracks `used_spans` so multi-span patches never repeat the same donor segment; avoids detected `exclude_events` (`±5.0 s`); applies RMS + 2-band (`<250 Hz` / `>250 Hz`) spectral tilt match; and logs `donor_quality` (`crest_factor`, `rms_diff_db`, `overlaps_event`).
- **Continuous Sequence Workflow Tools (`19` MCP Tools Total):**
  - Added `verify_seams` (LUFS delta, `2 s` RMS delta, sample step dBFS, `40–8000 Hz` Welch spectral diff, pass/fail threshold, and stitched `[-10 s, +10 s]` seam spectrogram PNG), `calibrate_sequence` (clean-interior slice solver computing one shared master `recommended_gain_db`), and `process_sequence` (single-call seam-padded batch processor with outer edge fades, exports, `verify_seams`, `find_clean_sections`, and `discard_take_audio_after_export`).
  - Added `copy_source=False` on `create_session` (referencing WAVs in-place with zero duplicate copy), `compact_metrics=True` across read/analyze/process tools (stripping verbose `short_term_lufs_timeline`), and `texture_tick_rate_per_min` on `AudioMetrics` to warn against running global `declick_impulse` on organic ASMR texture.

### Fixed
- **A/B Playback Hang on `0 s`-Start Previews:** Fixed missing `?` in `loadAudioWindow()` windowed query strings and ensured `Original` always requests a sliced `<= 90 s` window even when `original_offset_s == 0.0` (preventing multi-GB full-file fetch hangs on start-of-file previews).
- **Detection Accuracy & `freq_range_hz` Gating (`analysis.py`):**
  - Switched `_detect_hum_peaks` to use segment-median PSD when available with `±2.5 Hz` peak bin centering and `> 7.5 dB` prominence (detecting `49.8 Hz` mains hum without false-triggering on `45 Hz` HPF slopes or misclassifying it in `tonal_peaks_hz`).
  - Added `ambient_bed_dbfs`-relative floor gating on rumble curves, honored `freq_range_hz` bounds for rumble (`skip >= 80 Hz`) and clicks (`skip <= 1500 Hz`), and added temporary per-block narrowband tone detection (`>= 12 dB`) on long recordings.
  - Updated `find_clean_sections` (`include_low_severity=False` default) to recognize `spectral_match_suppress`, separate edit coverage from low-severity residual distractions, and report `raw_edit_span_count`, `residual_distraction_count`, `residual_distraction_s`, and `residual_distraction_pct`.
  - Recorded `attenuated_spans` telemetry in streaming `band_attenuate` batches and shifted all span telemetry accurately across seam-padded streaming runs.

## [0.3.2] - 2026-11-21

### Added
- **Split-File Boundary Seam Continuity (`process_take` / `dsp.py`):** Added optional `prev_audio_path`, `next_audio_path`, and `seam_context_s=8.0` to `process_take`, `apply_recipe_streaming`, and `apply_recipe_with_seam_padding`. Neighbor boundary context primes IIR filters, dynamic envelopes, and lookahead limiting seamlessly across recorder file splits (such as Zoom H5 2 GB chunks) before trimming back to exact original sample boundaries (`trim_range`/`cut_regions` explicitly rejected when seam padding is set).
- **Default Round-Number YouTube ASMR `clean_sections` Reporting & `find_clean_sections` MCP Tool:** Added `compute_clean_sections`, `extract_localized_edit_spans`, and `extract_localized_distraction_spans` in `analysis.py`, surfaced `clean_sections` by default in `create_session`, `get_session_summary`, `process_take`, and `analyze_take`, and added `find_clean_sections` to rank untouched and minimal-edit stretches in round YouTube ASMR durations (`15 min`, `30 min`, `45 min`, `1 hr`, `1.5 hr`, `2 hr`, `3 hr`, `4 hr`, `6 hr`, `8 hr`, `10 hr`) across single takes or ordered multi-file sequences.
- Updated `instructions.md` (`v0.3.2`) with Section 5b documenting multi-file continuous split workflow, unified fixed mastering gain, outer-edge-only fades, and default clean-section reporting.

### Fixed
- **Streaming 4 GiB RIFF Safety & Immediate Temp Cleanup:** Switched all internal 32-bit float streaming temp stages in `dsp.py` (`_STREAM_TEMP_FORMAT = "RF64"`, plus `RF64` final output when `>= 3.8 GB`) to avoid the 4 GiB RIFF WAV header cap on 2–3+ hour files, and unlinked superseded temporary stage files immediately after each stage (including intermediate `loudness_normalize` limiter passes) to halve transient disk footprint.

## [0.3.1] - 2026-11-20

### Fixed
- **Tonal Persistence & Constant-Q Sidebands:** Switched whole-file tonal peak discovery (`30–8000 Hz`, up from `1000 Hz`) to elementwise segment-median Welch PSD with constant-Q bilateral sidebands (`max(6 Hz, 3% fc)` to `max(22 Hz, 12% fc)`) and `>= 6.0 dB` prominence in `>= 60%` of segments—eliminating false whole-file `machine_drone` labels on loud brief passes (like `167 Hz`) while detecting steady resonant lines above `1 kHz` (`~2686 Hz`).
- **Dual-Timescale Slow-Pass Detection:** Added a `1.0 s` smoothed curve + hysteresis edge expansion pass (`>= 2.5 s`) alongside the fast `50 ms` detector in `_foreground_events_from_curve`, catching gentle vehicle passes and pass tails that blend into noisy short-window beds.
- **Cross-Branch A/B Original Alignment:** Restricted automatic `original_take_id` resolution strictly to shared common ancestors (`set_a & set_b`), adding `original_offset_s` so even separate trimmed branches replay the time-aligned ancestor slice (never another candidate branch).
- **Streaming Per-Step LUFS Telemetry:** `_stream_local_batch` and `apply_recipe_streaming` now accumulate ITU-R BS.1770 LUFS alongside RMS/peak per step so `step_metrics` populates `input_lufs` / `output_lufs` across every streaming step.
- **Actionable MCP Errors:** Surfaced `ValueError`, `FileNotFoundError`, and `PermissionError` via `ToolError(str(exc))` so callers see child take IDs and `cascade=True` instructions on `delete_take`, sandbox errors, and missing IDs.
- **Bed vs Hiss Clarity, Narrow Tonal Events & Gain-Invariant Clicks:** Added `ambient_bed_dbfs` (50th-percentile 100 ms RMS) and `high_freq_floor_dbfs` (10th-percentile `> 6 kHz` high-passed RMS) alongside `noise_floor_dbfs`; emitted one narrowband `machine_drone` event (`±5 Hz`) per persistent peak (honoring `freq_range_hz`); lowered the click gate amplitude floor from `5e-3` to `1e-4` so click counts stay invariant across gain changes; returned `events_truncated=False` when `include_events=False`; and updated `instructions.md` Section 1 to recommend `envelope_mode="adaptive"`.

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
