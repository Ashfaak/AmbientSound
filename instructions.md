# AmbientSound MCP — Agent Operational Instructions (`v0.5.3`)

This guide configures any AI coding or audio agent (Claude Code, Cursor, Antigravity, etc.) to restore ambient and outdoor nature recordings using the **AmbientSound** MCP server cleanly and conservatively.

---

## 0. Preflight: Verify MCP Connection Before Starting
1. Call `health_check` immediately. Verify:
   - `status == "ok"` and `version == "0.5.3"`
   - `filter_count == 21` and `tool_count == 26`
   - `workspace_dir` is writable (`workspace_free_gb` sufficient — long 2 GB+ split recordings create large takes; free space via `export_take(..., discard_take_audio=True)` or `delete_take` when done)
   - Your target audio directory is inside `allowed_roots` (`AMBIENTSOUND_ALLOWED_ROOTS`).
2. Call `list_filters` once per session to inspect all 21 filters, `step_schema`, and `spans_precedence`.

---

## 1. Mission & Core Ambient Principle ("Do No Harm")
- **Ambient bed is the signal:** Rain, wind-in-leaves, creek water, beehive buzz, and birdsong are preserved — NEVER apply blind broadband denoising unless persistent electronic hiss degrades the bed.
- **Check `cleanliness_assessment` (`assess_do_no_harm`) first:** When `analyze_take` or `analyze_region` reports `recommend_cleanup == False` (clean baseline), skip destructive cleanup and use gentle whole-clip `bed_shape` and/or `loudness_normalize` mastering only.
- **Macro-first event summaries:** By default (`include_clicks=None`), when `impulsive_click > 25` (e.g., natural rain, gravel, insects, or beehives), `events` omits the click wall so macro events (`slow_pass`, `synthetic_beep`, `human_voice`, `dog_bark`, `variable_rpm_engine`, `channel_biased_noise`, `click_train`, `crackle_burst`, `wind_noise`) fit cleanly in context. Pass `include_clicks=True` only when explicitly inspecting clicks.
- **Bird protection without frequency blindspots:** Always pass `protect_birds=True` on nature clips when using `spectral_match_suppress`, `harmonic_track_suppress`, or `distance_soften`. This protects FM-modulated chirp ridges (`300 Hz – 11.5 kHz`) rather than using a crude frequency ceiling, while still suppressing zero-drift synthetic piezo tones (`synthetic_beep` at `2–6 kHz`).

---

## 2. Environment Variables
- `AMBIENTSOUND_WORKSPACE`: Shared workspace directory (default `~/.ambientsound`; must match `--workspace` passed to `ambient-sound-ui`).
- `AMBIENTSOUND_ALLOWED_ROOTS`: Path-separated allowlist for reading inputs, `backup_audio_path` safety WAVs, `prev_audio_path`/`next_audio_path` seams, and writing exports.
- `AMBIENTSOUND_UI_URL`: Base URL of the local A/B Listening Studio (default `http://127.0.0.1:8765`).

---

## 3. MCP Tool Reference (`26` Tools)
1. **Discovery & Preflight:** `health_check`, `list_filters`, `list_sessions`.
2. **Ingestion & Inspection:**
   - `create_session(audio_path, name="", copy_source=False, compact_metrics=True, include_clicks=None)`: Use `copy_source=False` on large WAV sources to avoid duplicating GBs.
   - `get_session_summary(..., take_ids=None, compact_takes=False, include_clicks=None)`: Use `compact_takes=True` or `take_ids=[...]` to keep long multi-take sessions compact.
   - `analyze_take(..., compare_to_take_id=None, include_clicks=None)` & `analyze_region(session_id, take_id, start_s=0.0, end_s=None, ..., compare_to_take_id=None)`: Always use `compare_to_take_id="take_000"` (or parent take) after cleanup to inspect gain-normalized `bird_retention_db`, `gain_offset_db`, `musical_noise_index`, `band_deltas_db`, and `guardrail_warnings` (trimmed takes also report `source_offset_s` & `parent_start_s`).
   - `render_spectrogram(..., compare_to_take_id="", start_s=None, end_s=None)`.
3. **Processing & Cleanup:** `process_take`.
4. **Hardware & Human A/B Studio Loop:** `create_ab_test(session_id, take_a_id, take_b_id, ..., match_original=False)`, `list_ab_tests`, `get_ab_feedback`. Present `ui_url` to the user after generating two candidate takes.
5. **Housekeeping & Verified Export:**
   - `export_take(session_id, take_id, output_path, format="wav", discard_take_audio=False)`: Returns `output_path`, `size_bytes`, `duration_s`, `integrated_lufs`, `true_peak_dbtp`, `sha256`, and `take_audio_discarded` (never deletes `take_000`).
   - `delete_take`, `delete_session`.
6. **Multi-File Splits, Clean Section Compilation & Background Jobs:**
   - `calibrate_sequence`, `verify_seams(..., max_sample_step_excess_db=18.0)`, `find_clean_sections(..., round_to_file_local_minutes=False)`.
   - `process_sequence` (synchronous for short batches) or `start_sequence_job` (preferred for multi-hour split recordings): pass `output_dir`, `compile_clean_targets` (e.g., `["15 min", "30 min", "1 hr", "2 hr", "largest"]`), and optional `include_combined_full=True` to export all `<output_dir>/<prefix>_<stem>_master.<ext>` files alongside compiled `<output_dir>/clean_sections/*` and `clean_sections_manifest.json` in one folder.
   - `compile_clean_sections` & `start_compile_clean_sections_job`: Compile clean YouTube ASMR stretches from already-processed takes (with optional `distinct_regions=True` to avoid overlapping sections, `include_full_masters=True`, `include_combined_full=True`, raised-cosine outer fades, and automatic `RF64` above ~3.8 GB WAV or `format="flac"` for smaller lossless files with universal player compatibility).
   - `wait_for_job(job_id, timeout_s=30.0)` / `get_job_status(job_id, include_result=False)` / `cancel_job` / `list_jobs(include_result=False)`: Sleep server-side via `wait_for_job` without shell polling, and inspect `item_progress_pct`, `bytes_written`, and `eta_s`.

---

## 4. Built-In DSP Filter Reference (`21` Filters)
- **Hardware & Spatial Prep (run FIRST if needed):**
  - `declip_from_backup(backup_audio_path, clip_threshold=0.985, crossfade_ms=2.0, fir_taps=31, max_lag_ms=250.0, ar_fallback=True)`: Aligns and calibrates `-12 dB` safety track via GCC-PHAT + least-squares FIR to repair clipped peaks (falls back to AR interpolation if backup also clipped).
  - `spatial_mix(mode="auto"|"ms_decode"|"dual_mic_blend"|"dual_mono_to_stereo"|"lr_adjust", side_gain_db=0.0, directional_weight=0.5, directional_channel=0, crossover_hz=0.0, align_delay=True, low_mono_hz=0.0, side_hpf_hz=0.0, balance_db=0.0, auto_balance=False, decorrelate_ms=0.0)`: Guided by `metrics.stereo_layout_hint` (`raw_ms` -> `ms_decode` for SSH-6; `directional_plus_surround` -> `dual_mic_blend` for SGH-6 + ambience; `dual_mono` -> optional `dual_mono_to_stereo` on mono/dual-mono feeds).
- **Outdoor Distraction Removal:**
  - `highpass_rumble`, `dehum_notch`, `drone_notch`.
  - `declick_impulse(threshold_sigma=6.0, max_width_ms=3.0, spans=None, method="pchip"|"ar", min_flatness=0.0)`: Use `spans=[...]` + `method="ar"` on `click_train` (e.g., cable ties); avoid running globally when `texture_tick_rate_per_min` is high organic detail.
  - `spectral_match_suppress(..., protect_birds=False, spatial_mode="linked"|"channel_excess", narrowband_mode=False, context_gap_s=1.5)`: Use `narrowband_mode=True` on `synthetic_beep` (Garmin/watch beeps); `spatial_mode="channel_excess"` on `channel_biased_noise` (one-sided engines); `protect_birds=True` on `crackle_burst` (plastic bags) & traffic passes.
  - `harmonic_track_suppress(f0_min_hz=55.0, f0_max_hz=300.0, max_harmonics=12, spatial_bias=True, protect_birds=True, ...)`: Sweeping `f0(t)` comb filter for `variable_rpm_engine` (leafblowers, trimmers).
  - `distance_soften(mode="distance"|"voice_muffle"|"bark_distance", attenuation_db=-14.0, air_absorption=0.65, attack_smear_ms=12.0, formant_blur=0.5, narrow_stereo=0.35, protect_birds=True, ...)`: Pushes `human_voice` and `dog_bark` back softly instead of carving holes.
  - `band_attenuate`, `spectral_denoise`, `ambience_patch(min_donor_distance_s=5.0, max_donor_crest=6.0, ...)` (logs `loop_count`, `donor_chunks`, `donor_quality`).
- **Editing, Polishing & Mastering:**
  - `trim_range`, `cut_regions`, `fade_edges`, `gain`, `parametric_eq`, `soft_compress`, `true_peak_limiter`.
  - `bed_shape(sub_rumble_cut_db=-6.0, harshness_dip_db=-3.0, warmth_tilt_db=-1.5, floor_preserve=0.85, bed_width_boost=0.0, deharsh_dynamic=True)`: Whole-clip slow stationary bed polisher — lowers harsh `2–5 kHz` glare (`sharpness_acum`) and boosts `pleasantness_score` while leaving FM birdsong & transients intact (follow with `true_peak_limiter` when headroom is near `-1.0 dBTP`).
  - `loudness_normalize(target_lufs=-14.0, true_peak_ceiling_dbtp=-1.0, limit_peaks=True)`.

---

## 5. Multi-Hour & Outdoor Nature Playbooks
1. **Inspect layout & safety track:** Check `stereo_layout_hint`, `clipped_samples`, and `ndsi` / `sharpness_acum` / `pleasantness_score`. Run `declip_from_backup` -> `spatial_mix` before distraction filters if clipping or raw M/S / directional+surround channels exist.
2. **Targeted local cleanup over global cuts:** Match filter to `event.kind`:
   - `synthetic_beep` -> `spectral_match_suppress(narrowband_mode=True, protect_birds=True)`
   - `click_train` -> `declick_impulse(spans=..., method="ar")`
   - `crackle_burst` -> `spectral_match_suppress(protect_birds=True)` or `ambience_patch`
   - `human_voice` / `dog_bark` -> `distance_soften(mode="voice_muffle" | "bark_distance", protect_birds=True)`
   - `variable_rpm_engine` -> `harmonic_track_suppress(spatial_bias=True, protect_birds=True)`
   - `channel_biased_noise` -> `spectral_match_suppress(spatial_mode="channel_excess", protect_birds=True)`
3. **Always run `compare_to_take_id` guardrails:** If `bird_retention_db < -1.5` or `musical_noise_index > 0.35`, loosen strength/attenuation or raise `texture_fill`, or use `ambience_patch`.
4. **Long split sequences & unified output folder:** Calibrate one shared gain (`calibrate_sequence`), launch `start_sequence_job` with `output_dir=...`, `compile_clean_targets=...`, `verify_seams=True`, `clean_sections=True`, and `discard_take_audio_after_export=True`, and wait via `wait_for_job` until `"completed"`. For already-processed takes, use `start_compile_clean_sections_job(..., include_full_masters=True, distinct_regions=False)` into a dedicated `output` directory inside `allowed_roots` (`format="flac"` avoids RF64 >3.8 GB containers when desired).
