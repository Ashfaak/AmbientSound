# AmbientSound MCP — Agent Operational Instructions (`v0.4.0`)

> **Handover Guide for Any AI Model / MCP Agent**
> Use this document as your complete operational manual for the **AmbientSound** MCP server—covering preflight verification, short, multi-hour (672 MB+), and multi-file split continuous (Zoom H5 2 GB+) field recordings, wildlife/hive + moving-traffic/machine cleanup, loudness-matched A/B listening feedback, clean YouTube ASMR section discovery, and mastering.

---

## 0. Preflight: Verify MCP Connection Before Starting

If the AmbientSound MCP tools (`health_check`, `create_session`, `process_take`, etc.) do **not** appear in your active tool list, **stop immediately**—do not try to improvise DSP scripts or tell the user you processed audio.

Tell the user:
1. The AmbientSound MCP server is not connected to this session.
2. Provide or write a project `.mcp.json` pointing to your local **AmbientSound** clone (the folder containing `pyproject.toml`), with `AMBIENTSOUND_ALLOWED_ROOTS` set to the folder containing their audio file(s):
   ```json
   {
     "mcpServers": {
       "ambient-sound": {
         "command": "uv",
         "args": [
           "--directory",
           "C:\\path\\to\\AmbientSound",
           "run",
           "--no-sync",
           "ambient-sound-mcp"
         ],
         "env": {
           "AMBIENTSOUND_ALLOWED_ROOTS": "C:\\path\\to\\your\\audio\\folder"
         }
       }
     }
   }
   ```
3. Ensure dependencies are installed once in the AmbientSound folder (`uv sync --python 3.12 --all-groups`), then restart/reconnect the session so the MCP server loads.
4. Once connected, **call `health_check()` first** to verify `status: "ok"`, `version: "0.4.0"`, `filter_count: 16`, `allowed_roots`, `ffmpeg_available`, and `workspace_free_gb`.
5. For A/B listening steps, remind the user to start the local studio UI in a terminal (passing `--workspace` only if `AMBIENTSOUND_WORKSPACE` differs from the default `~/.ambientsound`):
   ```bash
   uv run --no-sync ambient-sound-ui
   ```
   In the A/B Studio, the synchronized cursor scrolls live over waveforms and spectrograms (`#playheadLine`), supports clicking/dragging anywhere on the plot to seek, and provides `A Diff`, `B Diff`, and `A vs B Diff` spectrograms.

---

## 1. Mission & Core Ambient Principle

Your goal is to clean up **field and ambient soundscape recordings** (rain, bee hives, forest wind, ocean surf, room tone, cafés, night streets) by removing **distractions** without degrading the surrounding natural ambience.

Core rules:
1. **Ambience is the signal.** Avoid heavy `spectral_denoise` on rain, leaves, bees, or water—broadband gating creates metallic/swishy artifacts. Also check `metrics.texture_tick_rate_per_min`: if `>= 30/min` (e.g., hive/insect micro-ticks), **do not** run `declick_impulse` globally.
2. **Choose the right localized repair strategy:**
   - **Moderate wind thumps or isolated low swells:** **`band_attenuate`** with `envelope_mode="adaptive"` and `fade_ms` (`800–3000 ms`) so peak cut follows actual excess without hollowing quiet edges.
   - **Moving vehicle exhaust harmonics, train/bell tones, or tonal traffic overlapping `200–300 Hz` bees:** **`spectral_match_suppress`** (`n_fft=4096`). Estimates per-bin clean bed medians from surrounding context (or manual `ref_start_s`/`ref_end_s`), dynamically suppresses only the excess bins (`25–4000 Hz`), and fills deep cuts smoothly with clean context texture (`texture_fill=0.5–0.7`) so non-excess hive frequencies remain natural.
   - **Unrecoverable loud passes / microphone bumps / handling thumps:** **`ambience_patch`** v2 (`crossfade_ms` up to `5000 ms`). Automatically selects **non-repeating**, event-screened clean donors matching both RMS and 2-band spectral tilt, and reports `donor_quality` (`crest_factor`, `rms_diff_db`, `overlaps_event`) in `step_metrics`.
   - **Continuous edits inside >1 hr files:** Avoid deleting/cutting large stretches unless explicitly wanted; prefer seamless attenuation or texture patching so recording continuity stays intact.
3. **Preserve stereo width & non-destructive branching.** `take_000` and every `take_NNN` are immutable. Never overwrite takes.

---

## 2. Environment Variables

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `AMBIENTSOUND_WORKSPACE` | `~/.ambientsound` | Shared session workspace directory. |
| `AMBIENTSOUND_ALLOWED_ROOTS` | `CWD` + `workspace_dir` | OS path-separator list of directories tools may read/write. |
| `AMBIENTSOUND_UI_HOST` | `127.0.0.1` | Hostname used in `ui_url`. |
| `AMBIENTSOUND_UI_PORT` | `8765` | Port used in `ui_url`. |
| `AMBIENTSOUND_ALLOW_NONCOMMERCIAL` | `0` | Safety gate for optional future non-commercial ML weights (`0` by default). |

---

## 3. MCP Tool Reference (19 Tools)

All tools return finite JSON objects (`null` instead of `NaN`/`±inf`). Set **`compact_metrics=True`** on `create_session`, `get_session_summary`, `analyze_take`, `analyze_region`, and `process_take` on long multi-hour recordings to omit verbose `short_term_lufs_timeline` points while preserving `loudness_range_lu`, `short_term_lufs_stats`, and `noise_floor_dbfs`.

| Tool | Arguments | Purpose & Return |
| :--- | :--- | :--- |
| `health_check` | `()` | Confirms `status`, `version` (`0.4.0`), `allowed_roots`, `ffmpeg_available`, `workspace_free_gb`, `streaming_threshold_s` (`120.0 s`), and `filter_count` (`16`). |
| `list_filters` | `()` | Returns `{"filters": [...]}` with exact parameter schemas, bounds, and defaults. |
| `create_session` | `(audio_path, name="", event_kinds=None, max_events=50, include_events=True, copy_source=True, compact_metrics=False)` | Ingests audio into `take_000` (`copy_source=False` references allowed `.wav` files in-place with zero duplicate GBs), archives `events.json`, renders overview PNG (`n_fft=8192`), and returns `clean_sections`. |
| `list_sessions` | `()` | Returns `{"sessions": [...]}` summary list. |
| `get_session_summary` | `(session_id, event_kinds=None, max_events=50, include_events=True, compact_metrics=False)` | Returns `SessionSummary` (`takes`, `clean_sections`, `metrics`, `event_counts`, `step_metrics`, `ab_tests`). |
| `analyze_take` | `(session_id, take_id, sensitivity="medium", min_event_s=0.12, max_event_s=45.0, freq_range_hz=None, event_kinds=None, max_events=50, include_events=True, compact_metrics=False)` | Streaming analysis (`50/60 Hz` segment-median hum detection, `30–8000 Hz` persistent tonal peaks, per-block temporary tonal detections, `texture_tick_rate_per_min`, and `freq_range_hz` gating for rumble/clicks). |
| `analyze_region` | `(session_id, take_id, start_s=0.0, end_s=None, sensitivity="medium", min_event_s=0.12, max_event_s=45.0, freq_range_hz=None, event_kinds=None, max_events=50, include_events=True, compact_metrics=False)` | Analyzes `[start_s, end_s]` with ambient-bed-relative floor gating and returns absolute timestamps. |
| `render_spectrogram` | `(session_id, take_id, compare_to_take_id="", start_s=None, end_s=None)` | Renders overview (`>60 s` sampled) or zoomed (`<=60 s`) `n_fft=8192` PNG and timeline-aligned 3-panel diff PNG (`Patched` spans shaded). |
| `process_take` | `(session_id, parent_take_id, steps, label="", notes="", event_kinds=None, max_events=50, include_events=True, prev_audio_path=None, next_audio_path=None, seam_context_s=8.0, compact_metrics=False)` | Runs DSP recipe sequentially (`>120 s` streams file-to-file via RF64 stages with immediate temp cleanup), optionally pads split boundaries, avoids parent localized distractions during auto `ambience_patch`, and records `step_metrics` + `donor_quality`. |
| `create_ab_test` | `(session_id, take_a_id, take_b_id, focus_start_s=None, focus_end_s=None, blind=False, notes="", original_take_id=None, match_original=False)` | Creates attenuation-only A/B test (`match_original=False` matches A & B at full loudness and aligns `original_offset_s`). Always pass `focus_start_s`/`focus_end_s` on `>90 s` takes. |
| `list_ab_tests` | `(session_id)` | Returns `{"ab_tests": [...]}` with verdicts and `ui_url`s. |
| `get_ab_feedback` | `(session_id, ab_id)` | Retrieves `verdict` (`winner`, `artifact_tags`, `comment`, `submitted_at`). |
| `delete_take` | `(session_id, take_id, cascade=False)` | Deletes an abandoned non-root take (`take_000` protected) and orphaned A/B tests. |
| `delete_session` | `(session_id)` | Permanently deletes a session directory (never deletes external WAVs linked with `copy_source=False`). |
| `export_take` | `(session_id, take_id, output_path, format="wav")` | Stream-exports take to `wav` (24-bit), `flac` (24-bit), `mp3` (320k), or `ogg`. |
| `find_clean_sections` | `(session_id="", take_id="", sequence=None, max_edits_per_hour=2.0, max_edited_pct=1.0, include_low_severity=False)` | Ranks largest untouched and minimal-edit ASMR stretches in round durations (`15 min`–`10 hr`), separating edits from residual distractions (`low` severity skipped by default). |
| `verify_seams` | `(sequence, window_s=10.0)` | Measures LUFS delta, `2 s` RMS delta, sample step dBFS, `40–8000 Hz` spectral diff, pass/fail boolean, and renders stitched `[-window, +window]` seam PNGs across consecutive splits. |
| `calibrate_sequence` | `(sequence, shared_steps_before_gain, target_lufs=-14.5, ceiling_dbtp=-1.0, sample_windows_per_file=5, window_s=20.0, shared_steps_after_gain=None)` | Solves ONE exact shared master `recommended_gain_db` across a split sequence on clean interior slices before full-file processing. |
| `process_sequence` | `(items, shared_prefix_steps=None, shared_suffix_steps=None, outer_fade_in_s=0.0, outer_fade_out_s=0.0, seam_context_s=8.0, verify_seams=True, clean_sections=True, export_format="wav", discard_take_audio_after_export=False)` | Batch-processes split recordings with neighbor seam padding, outer-edge-only fades, exports, seam QC, clean-section reporting, and optional duplicate `audio.wav` cleanup after export. |

---

## 4. Built-In DSP Filter Reference (16 Filters)

> **Timeline Rule:** Steps execute sequentially. Any `trim_range` or `cut_regions` shifts/shortens audio immediately; subsequent steps operate on the new `0.0 s`-based timeline.

| Filter | Key Parameters (Defaults & Bounds) | Ambient & Mastering Usage |
| :--- | :--- | :--- |
| `highpass_rumble` | `cutoff_hz` (`60.0`, `15–300`), `order` (`4`, `1–8`) | Clears wind/traffic sub-rumble. `45–80 Hz` protects `200–300 Hz` bee buzz while removing sub-bass thumps. |
| `dehum_notch` | `base_freq_hz` (`60.0`, `20–200`), `harmonics` (`4`, `1–8`), `q` (`30.0`, `5–80`) | Zero-phase cascaded notches for `50/60 Hz` mains hum (`hum_peaks_hz`). |
| `drone_notch` | **`freqs_hz`**, `q` (`25.0`, `2–80`), `attenuation_db` (`-24.0`, `-48 to -1`), `harmonics` (`1`, `1–6`) | Proportional zero-phase peaking cuts at persistent machine frequencies (`tonal_peaks_hz`). |
| `declick_impulse` | `threshold_sigma` (`6.0`, `3–20`), `max_width_ms` (`3.0`, `0.5–12`) | PCHIP spike repair for mic clicks/pops. **Skip** when `texture_tick_rate_per_min >= 30` (desirable ASMR texture!). |
| `spectral_denoise` | `strength` (`0.45`), `floor_db` (`-18.0`), `n_fft` (`2048`), `noise_start_s`, `noise_end_s` | Steady broadband preamp hiss only. Keep `strength <= 0.30` on live ambience. |
| `spectral_match_suppress` | `start_s`, `end_s`, `low_hz` (`25`), `high_hz` (`4000`), `strength` (`1.0`, `0.1–1.5`), `max_attenuation_db` (`-32`, `-60 to -3`), `texture_fill` (`0.65`, `0–1`), `context_s` (`6`, `1–20`), `fade_ms` (`1500`, `20–5000`), `n_fft` (`4096`), `ref_start_s`, `ref_end_s`, `spans` | **Primary tool for moving cars, exhaust harmonics, train horns, and temporary tonal overlaps.** Dynamically attenuates excess per-bin magnitude above clean surrounding bed percentiles while filling deep cuts subtly with clean texture. Logs `suppressed_spans` (`peak_excess_db`, `max_cut_db`, `mean_cut_db`). |
| `band_attenuate` | `start_s`, `end_s`, `low_hz` (`40`), `high_hz` (`250`), `attenuation_db` (`-18`, `-60 to +12`), `fade_ms` (`30`, `2–5000`), `envelope_mode` (`"flat"` \| `"swell"` \| `"adaptive"`), `n_fft` (`2048`), `spans` | Targeted band cut (best `<160 Hz` for wind thumps). Use `envelope_mode="adaptive"`. Logs `attenuated_spans`. |
| `ambience_patch` | `start_s`, `end_s`, `source_start_s` (`null`), `crossfade_ms` (`40`, up to `5000`), `spans` | Replaces unrecoverable spans (loud close vehicles, mic handling thumps, starting clicks) with **non-repeating**, event-screened donors matched in RMS & 2-band spectral tilt. Logs `patched_spans` + `donor_quality`. |
| `trim_range` | `start_s` (`0.0`), `end_s` (`null`) | Extracts `[start_s, end_s]`. Sets `source_offset_s` for automatic A/B Diff and Original alignment. |
| `cut_regions` | **`regions`** (`[{"start_s", "end_s"}, ...]`), `crossfade_ms` (`1000`, `10–5000`) | Removes bad stretches with equal-power crossfades. |
| `fade_edges` | `fade_in_s` (`2.0`, `0–60`), `fade_out_s` (`3.0`, `0–60`), `curve` (`"cosine"` \| `"linear"`) | Smooth start/end fades for finished exports. |
| `gain` | `gain_db` (`0.0`, `-48 to +48`) | Clean linear sample gain stage (ideal after `calibrate_sequence`). |
| `parametric_eq` | **`freq_hz`**, **`gain_db`**, `q` (`1.4`), `filter_type` (`"peaking"` \| `"lowshelf"` \| `"highshelf"`) | Shape tone: e.g. `+1.5 dB lowshelf @ 160 Hz` and `+2 dB highshelf @ 11000 Hz`. |
| `soft_compress` | `threshold_db` (`-24`), `ratio` (`2.0`), `attack_ms` (`25`), `release_ms` (`250`), `makeup_db` (`0.0`) | Gentle linked-stereo RMS compression for consistent ASMR playback. |
| `true_peak_limiter` | `ceiling_dbtp` (`-1.0`), `release_ms` (`80.0`) | Lookahead peak limiter enforcing strict ceiling (`-1.0 dBTP`). |
| `loudness_normalize` | `target_lufs` (`-18.0`, `-40 to -6`), `true_peak_ceiling_dbtp` (`-1.0`), `limit_peaks` (`false`), `limiter_release_ms` (`80.0`) | BS.1770 normalization on single files (`limit_peaks=True` reaches `-14 LUFS` cleanly). For multi-file splits, use `calibrate_sequence` + fixed `gain` + `true_peak_limiter` instead! |

---

## 5. Multi-Hour & Continuous Split Workflow (`ZOOM` / Field Recorders)

### 5a. Low-Disk Ingestion & Discovery
1. Call `create_session(audio_path, copy_source=False, compact_metrics=True, max_events=50)` on each `.wav` file — zero duplicate copy GBs, full `events.json` archival, compact responses.
2. Scan sub-band events cleanly without click floods:
   - `freq_range_hz=[35, 160]`, `sensitivity="high"`, `min_event_s=1.5` (note: click detector is skipped when `high <= 1500 Hz`, rumble detector is skipped when `low >= 80 Hz`).
   - Full-take `analyze_take` also runs per-block temporary narrowband tone detection (`>= 12 dB`) and reports `texture_tick_rate_per_min`.
3. Prototype parameters on a `40–60 s` `trim_range` preview first: branch **Candidate A** (`spectral_match_suppress`) vs **Candidate B** (`ambience_patch`), run `create_ab_test(..., focus_start_s=0.0, focus_end_s=duration_s)` (or full-take windowed `focus_start_s`/`focus_end_s`), inspect `A Diff`, `B Diff`, and `A vs B Diff`, and call `delete_take` when finished prototyping.

### 5b. Continuous Split Sequence Processing & QC
1. **Solve one shared gain with `calibrate_sequence`:** Pass `sequence=[{"session_id": sid, "take_id": "take_000"}, ...]`, `shared_steps_before_gain` (may be `[]` if none), and `shared_steps_after_gain` (e.g., `soft_compress` + `true_peak_limiter`), targeting `-14.5 LUFS` at `-1.0 dBTP`. All source/export paths must reside inside `AMBIENTSOUND_ALLOWED_ROOTS` (and pass `--workspace` to `ambient-sound-ui` if `AMBIENTSOUND_WORKSPACE` differs from `~/.ambientsound`).
2. **Process the entire sequence in one call with `process_sequence`:**
   - Pass `items=[{"session_id": sid, "parent_take_id": "take_000", "local_steps": [...], "label": "...", "notes": "...", "export_path": ".../master.wav"}, ...]`.
   - Put shared cleanup/EQ in `shared_prefix_steps`, unique per-file spans in each item's `local_steps`, and shared `[gain(recommended_gain_db), soft_compress, true_peak_limiter]` in `shared_suffix_steps`.
   - Provide `export_path` on every item to export (`export_format`: `"wav"`, `"flac"`, `"mp3"`, `"ogg"`) — items without `export_path` are not exported, and `discard_take_audio_after_export=True` requires `export_path` on every item (reclaiming duplicate session WAVs while keeping Web UI / A/B creation intact).
   - Set `outer_fade_in_s=3.0`, `outer_fade_out_s=5.0`, `seam_context_s=8.0`, `verify_seams=True` (`window_s=10.0`), and `clean_sections=True`.
3. **Always verify seams and report clean YouTube ASMR sections:**
   - Check `seams["all_passed"]` (`lufs_delta_db <= 1.5`, `rms_delta_2s_db <= 2.0`, `spectral_diff_db <= 3.0`) and inspect generated `seam_*.png` spectrograms.
   - Report the largest **untouched** and **minimal-edit** stretches in round YouTube ASMR durations (`10 min`, `15 min`, `20 min`, `30 min`, `45 min`, `1 hr`, `1.5 hr`, `2 hr`, `3 hr`, `4 hr`, `5 hr`, `6 hr`, `8 hr`, `10 hr`) from `find_clean_sections` (note: calling `find_clean_sections(session_id=...)` without `take_id` selects that session's latest take; pass `sequence=[{"session_id": ..., "take_id": ...}]` for multi-file spans).

---

## 6. Feedback-to-Tuning Playbook

| Artifact Tag / User Comment | Cause | Adjustment |
| :--- | :--- | :--- |
| **`Metallic / swishy`** | Over-aggressive `spectral_denoise` or zero `texture_fill` on deep suppression. | Remove `spectral_denoise`; raise `spectral_match_suppress.texture_fill` (`0.6–0.8`) or switch severe short burst to `ambience_patch`. |
| **`Hollow gap`** | Static `band_attenuate` dug below bed or abrupt edge. | Switch to `spectral_match_suppress` (preferred for broadband/harmonic passes) or `band_attenuate(envelope_mode="adaptive")`. |
| **`Distraction still audible`** | Exhaust harmonics / bell tone extended above `180 Hz` or passive cut too weak. | Switch to **`spectral_match_suppress`** with wider `low_hz=25, high_hz=4000, strength=1.1, max_attenuation_db=-36, texture_fill=0.65`, or `ambience_patch` with verified clean donor (`overlaps_event=False`). |
| **`Too dull / muffled` / bees thinned** | `highpass_rumble` cut into `200–300 Hz` buzz. | Lower `highpass_rumble.cutoff_hz` (`45–60 Hz`), keep `<120 Hz` narrow wind cuts, use `spectral_match_suppress` instead of wide static notch. |
| **`Stereo image shifted`** | Donor patch came from asymmetric moment. | Check `donor_quality` and specify balanced `source_start_s` manually if needed. |
| **`Pumping / breathing`** | High compressor ratio or short fades. | Lower `soft_compress.ratio` (`1.5–1.8`), raise `release_ms` (`350–400 ms`), lengthen `fade_ms` (`1500–3000 ms`). |
