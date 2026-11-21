# AmbientSound MCP — Agent Operational Instructions (`v0.3.2`)

> **Handover Guide for Any AI Model / MCP Agent**
> Use this document as your complete operational manual for the **AmbientSound** MCP server—covering preflight verification, short, multi-hour (672 MB+), and multi-file split continuous (Zoom H5 2 GB+) field recordings, wildlife/hive + traffic/machine cleanup, loudness-matched A/B listening feedback, clean YouTube ASMR section discovery, and mastering.

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
4. Once connected, **call `health_check()` first** to verify `status: "ok"`, `allowed_roots`, `ffmpeg_available`, and `workspace_free_gb`.
5. For A/B listening steps, remind the user to start the local studio UI in a terminal (passing `--workspace` only if `AMBIENTSOUND_WORKSPACE` differs from the default `~/.ambientsound`):
   ```bash
   uv run ambient-sound-ui
   ```

---

## 1. Mission & Core Ambient Principle

Your goal is to clean up **field and ambient soundscape recordings** (rain, bee hives, forest wind, ocean surf, room tone, cafés, night streets) by removing **distractions** without degrading the surrounding natural ambience.

Core rules:
1. **Ambience is the signal.** Avoid heavy `spectral_denoise` on rain, leaves, bees, or water—broadband gating creates metallic/swishy artifacts.
2. **Patch, cut, or adaptively attenuate—don't hard-mute.**
   - Use **`band_attenuate`** with `envelope_mode="adaptive"` and `fade_ms` (`800–3000 ms`) for asymmetric `5–20 s` passing cars/motorbikes so peak reduction follows the real swell excess without hollowing quiet edges (`"swell"` is also available for symmetric bell curves).
   - Use **`ambience_patch`** (`crossfade_ms` up to `5000 ms`) to replace discrete distracting spans with clean matching texture.
   - Use **`cut_regions`** (`crossfade_ms=1000–2500 ms`) on long-form recordings (`1 hour+`) when an unrecoverable vehicle pass or speech stretch is cleaner to remove cleanly than filter.
3. **Respect overlapping frequencies honestly.** Engine/tyre rumble (`80–1000 Hz`) overlaps `200–300 Hz` honeybee buzz fundamentals. High-pass filtering at `80–120 Hz` cleans subsonic wind/rumble safely below the bees, and `drone_notch` removes narrow machine lines, **but** no pure DSP filter can remove simultaneous broadband engine noise sitting inside `200–800 Hz` without thinning the hive—prefer **`cut_regions`** or **`ambience_patch`** from clean bee-only sections for heavy vehicle passes!
4. **Preserve stereo width & non-destructive branching.** `take_000` and every `take_NNN` are immutable. Never overwrite takes.

---

## 2. Environment Variables

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `AMBIENTSOUND_WORKSPACE` | `~/.ambientsound` | Shared session workspace directory (note: long 1-hour WAVs write ~672 MB per take). |
| `AMBIENTSOUND_ALLOWED_ROOTS` | `CWD` + `workspace_dir` | OS path-separator list of directories `create_session` / `export_take` may read/write. |
| `AMBIENTSOUND_UI_HOST` | `127.0.0.1` | Hostname used in `ui_url`. |
| `AMBIENTSOUND_UI_PORT` | `8765` | Port used in `ui_url`. |
| `AMBIENTSOUND_ALLOW_NONCOMMERCIAL` | `0` | Safety gate for optional future non-commercial ML weights (`0` by default). |

---

## 3. MCP Tool Reference (16 Tools)

All tools return JSON objects (`dict`). Unmeasurable metrics (e.g., LUFS on `<0.45 s` or silence) return `null` instead of `NaN`/`±inf`. Analysis/take tools archive complete detections to `events.json` per take, include full per-kind totals (`event_counts`, `events_total`, `events_truncated`), include a `clean_sections` YouTube ASMR summary by default on takes and full-take analyses, and default to `max_events=50` (prioritizing sustained events ahead of `impulsive_click`s) so responses stay compact on long files.

| Tool | Arguments | Purpose & Return |
| :--- | :--- | :--- |
| `health_check` | `()` | Confirms `status`, `version`, `allowed_roots`, `ffmpeg_available`, `workspace_free_gb`, `streaming_threshold_s` (`120.0 s`), and `filter_count` (`15`). |
| `list_filters` | `()` | Returns `{"filters": [...]}` with exact parameter schemas, bounds, and defaults. |
| `create_session` | `(audio_path, name="", event_kinds=None, max_events=50, include_events=True)` | Ingests audio into `take_000` in bounded RAM (`shutil.copy2` for WAV; streaming `PCM_24` otherwise), runs streaming analysis (`loudness_range_lu`, `ambient_bed_dbfs`, `noise_floor_dbfs`, `high_freq_floor_dbfs`, `short_term_lufs_stats`, `short_term_lufs_timeline`), writes `events.json`, renders overview PNG (`n_fft=8192`), and includes `clean_sections`. |
| `list_sessions` | `()` | Returns `{"sessions": [...]}` summary list. |
| `get_session_summary` | `(session_id, event_kinds=None, max_events=50, include_events=True)` | Returns `SessionSummary` (`takes` with `clean_sections`, `metrics`, `event_counts`, `step_metrics`, bounded `events`, `ab_tests`). |
| `analyze_take` | `(session_id, take_id, sensitivity="medium", min_event_s=0.12, max_event_s=45.0, freq_range_hz=None, event_kinds=None, max_events=50, include_events=True)` | Full-file streaming analysis with `30–8000 Hz` median-PSD persistent tonal detection and dual-timescale `1.0 s` slow-pass hysteresis, plus `clean_sections`. Pass `min_event_s > 0.15` or `event_kinds=[...]` to skip short clicks. |
| `analyze_region` | `(session_id, take_id, start_s=0.0, end_s=None, sensitivity="medium", min_event_s=0.12, max_event_s=45.0, freq_range_hz=None, event_kinds=None, max_events=50, include_events=True)` | Analyzes `[start_s, end_s]` with `-92.0 dBFS` gate floor and returns absolute recording timestamps (`+ start_s`). |
| `render_spectrogram` | `(session_id, take_id, compare_to_take_id="", start_s=None, end_s=None)` | Renders full overview (`>60 s` sampled) or zoomed (`<=60 s`) `n_fft=8192` PNG (suppressing `>40` click markers on waveform) and gain-matched Before/After/Diff PNGs with `Patched` spans shaded. |
| `process_take` | `(session_id, parent_take_id, steps, label="", notes="", event_kinds=None, max_events=50, include_events=True, prev_audio_path=None, next_audio_path=None, seam_context_s=8.0)` | Runs DSP recipe sequentially (`>120 s` streams file-to-file via RF64 stages with immediate temp cleanup), optionally pads boundaries from adjacent split files (`seam_context_s=8.0`) for sample-continuous seams, writes immutable `take_NNN`, tracks `source_offset_s`/`timeline_modified`, records `step_metrics` & `clean_sections`, and renders visuals. |
| `create_ab_test` | `(session_id, take_a_id, take_b_id, focus_start_s=None, focus_end_s=None, blind=False, notes="", original_take_id=None, match_original=False)` | Creates attenuation-only A/B test. `original_take_id=None` auto-selects the shared common ancestor (`take_a`/`take_b` lineage intersection) and computes `original_offset_s` so trimmed previews align automatically. `match_original=False` matches LUFS between A and B. |
| `list_ab_tests` | `(session_id)` | Returns `{"ab_tests": [...]}` with verdicts and `ui_url`s. |
| `get_ab_feedback` | `(session_id, ab_id)` | Retrieves `verdict` (`winner`, `artifact_tags`, `comment`, `submitted_at`). |
| `delete_take` | `(session_id, take_id, cascade=False)` | Deletes an abandoned non-root take (`take_000` protected) and orphaned A/B tests; returns an explicit `ToolError` naming child takes and `cascade=True` if children exist. |
| `delete_session` | `(session_id)` | Permanently deletes an entire session directory and its takes. |
| `export_take` | `(session_id, take_id, output_path, format="wav")` | Stream-exports take to `wav` (24-bit), `flac` (24-bit), `mp3` (320k), or `ogg`. |
| `find_clean_sections` | `(session_id="", take_id="", sequence=None, max_edits_per_hour=2.0, max_edited_pct=1.0)` | Computes largest **untouched** and **minimal-edit** stretches in round YouTube ASMR durations (`15 min`, `30 min`, `45 min`, `1 hr`, `1.5 hr`, `2 hr`, `3 hr`, `4 hr`, `6 hr`, `8 hr`, `10 hr`) for one take or an ordered multi-file `sequence=[{"session_id": ..., "take_id": ...}]`. |

---

## 4. Built-In DSP Filter Reference (15 Filters)

> **Timeline Rule:** Steps in `process_take` execute sequentially in order. Any `trim_range` or `cut_regions` shifts/shortens audio immediately; all **subsequent** steps in the same recipe (or in child takes branching from that trimmed/cut take) use timestamps on the **new resulting timeline**, starting at `0.0 s`. Either place `trim_range` / `cut_regions` first (when prototyping or pre-trimming) or place them last if your timestamps refer to `take_000`.

| Filter | Key Parameters (Defaults & Bounds) | Ambient & Mastering Usage |
| :--- | :--- | :--- |
| `highpass_rumble` | `cutoff_hz` (`60.0`, `15–300`), `order` (`4`, `1–8`) | Clears wind/traffic sub-rumble. `45–90 Hz` protects `200–300 Hz` bee buzz while removing sub-bass thumps. |
| `dehum_notch` | `base_freq_hz` (`60.0`), `harmonics` (`4`), `q` (`30.0`) | Zero-phase cascaded notches for `50/60 Hz` mains hum (`hum_peaks_hz`). |
| `drone_notch` | **`freqs_hz`** (`list[float]`), `q` (`25.0`, `2–80`), `attenuation_db` (`-24.0`, `-48 to -1`), `harmonics` (`1`, `1–6`) | Proportional zero-phase peaking cuts at persistent machine frequencies (`tonal_peaks_hz`) without thinning nearby hive harmonics. |
| `declick_impulse` | `threshold_sigma` (`6.0`), `max_width_ms` (`3.0`) | PCHIP spike repair for mic clicks/pops. **Skip** on bee hives, crackling fire, or rain drips if micro-ticks are desirable ASMR texture! |
| `spectral_denoise` | `strength` (`0.45`), `floor_db` (`-18.0`), `n_fft` (`2048`), `noise_start_s`, `noise_end_s` | Steady broadband preamp hiss only. Keep `strength <= 0.30` on live ambience. |
| `band_attenuate` | `start_s`, `end_s`, `low_hz` (`40`), `high_hz` (`250`), `attenuation_db` (`-18`), `fade_ms` (`30`, up to `5000`), `envelope_mode` (`"flat"` \| `"swell"` \| `"adaptive"`), `spans` (`list[dict]`) | Single or **batched multi-span** time-frequency attenuation. Prefer `envelope_mode="adaptive"` for asymmetric `5–20 s` vehicle passes so peak reduction tracks the real swell peak without hollowing quiet edges. `spans` accepts per-span `low_hz`, `high_hz`, `attenuation_db`, `fade_ms`, `envelope_mode`. |
| `ambience_patch` | `start_s`, `end_s`, `source_start_s` (`null`), `crossfade_ms` (`40`, up to `5000`), `spans` (`list[dict]`) | Single or **batched multi-span** texture replacement from clean donor audio outside all spans. Each dict in `spans` may specify its own `crossfade_ms` and `source_start_s`; `step_metrics[].details.patched_spans` logs exact `donor_start_s`/`donor_end_s` used. |
| `trim_range` | `start_s` (`0.0`), `end_s` (`null`) | Extracts `[start_s, end_s]`. Sets child `source_offset_s` so auto A/B Original selection aligns automatically. |
| `cut_regions` | **`regions`** (`[{"start_s", "end_s"}, ...]`), `crossfade_ms` (`1000`, `10–5000`) | Removes severe bad stretches and stitches remaining segments with equal-power crossfades. |
| `fade_edges` | `fade_in_s` (`2.0`, `0–60`), `fade_out_s` (`3.0`, `0–60`), `curve` (`"cosine"` \| `"linear"`) | Smooth start/end fades for finished exports. |
| `gain` | `gain_db` (`0.0`, `-48 to +48`) | Clean linear sample gain stage. Place before `soft_compress` on very quiet `-45` to `-55 LUFS` sources so audio crosses compression threshold cleanly. |
| `parametric_eq` | **`freq_hz`**, **`gain_db`**, `q` (`1.4`), `filter_type` (`"peaking"` \| `"lowshelf"` \| `"highshelf"`) | Shape tone: e.g. `+2 dB lowshelf @ 160 Hz` (warmth) and `+2.5 dB highshelf @ 11000 Hz` (ASMR air). |
| `soft_compress` | `threshold_db` (`-24`), `ratio` (`2.0`), `attack_ms` (`25`), `release_ms` (`250`), `makeup_db` (`0.0`) | Gentle linked-stereo RMS compression for consistent ASMR playback without pumping. |
| `true_peak_limiter` | `ceiling_dbtp` (`-1.0`), `release_ms` (`80.0`) | Lookahead peak limiter enforcing strict ceiling (`-1.0 dBTP`). |
| `loudness_normalize` | `target_lufs` (`-18.0`, `-40 to -6`), `true_peak_ceiling_dbtp` (`-1.0`), `limit_peaks` (`false`), `limiter_release_ms` (`80.0`) | Streaming BS.1770 normalization. Set **`target_lufs=-14.0, limit_peaks=True`** for YouTube/ASMR playback on spiky field recordings so peaks limit transparently at `-1.0 dBTP` instead of stalling at `-28 LUFS`. |

---

## 5. Large-File (1-Hour / 672 MB+) & Bee-Hive Field Workflow

When `duration_s > 120.0` (e.g., an hour-long `48 kHz / 24-bit` WAV), AmbientSound automatically operates in **streaming mode** (`< 150 MB` RAM):
- **Avoid unnecessary disk bloat:** Each full 1-hour take writes ~672 MB. Prototype parameters on a `30–60 s` `trim_range` preview first, use `step_metrics` to verify each stage's gain/RMS progression, and call **`delete_take`** on discarded experiments when finished.
  1. Call `create_session(audio_path, max_events=50)` (defaults keep payloads compact and put sustained events before clicks). Inspect `metrics` (`hum_peaks_hz`, `tonal_peaks_hz`, `rumble_ratio_db`, `loudness_range_lu`, `noise_floor_dbfs`, `short_term_lufs_stats`) and `event_counts`.
  2. Call `analyze_take` or `analyze_region` with `freq_range_hz=[35, 160]`, `sensitivity="high"`, and `min_event_s=1.5` (suppresses `impulsive_click` floods!) to locate sub-hive traffic/wind swells cleanly without bee buzz triggering false positives.
  3. Call `render_spectrogram(..., start_s=..., end_s=...)` (`n_fft=8192`, ~5.9 Hz low-frequency resolution) over `20–45 s` windows around candidate events.
  4. **Prototype on a `trim_range` preview slice first:** Create `take_001 = trim_range(start_s=..., end_s=...)`, branch Candidate A (`band_attenuate(envelope_mode="adaptive")`) vs Candidate B (`ambience_patch`), and run `create_ab_test(session, take_a, take_b)` — AmbientSound automatically uses `take_001` as the time-aligned Original reference and matches loudness between A and B (`match_original=False`).
  5. Apply the winning full-file chain cleanly from `take_000` using batched `spans` (`band_attenuate` `adaptive` / `ambience_patch` with per-span `crossfade_ms`) and mastering (`gain` -> `parametric_eq` -> `soft_compress` -> `loudness_normalize(..., limit_peaks=True)` -> `fade_edges`).
  6. **A/B Studio on Long Files:** Always pass `focus_start_s` and `focus_end_s` when calling `create_ab_test` on takes `> 90 s` so the Web Studio loads fast `<= 90 s` windows and computes loudness-matched dynamic residuals on hotkey `4`.

### 5b. Split Multi-File Continuous Recordings (Zoom H5 / 2 GB+ Splits)
When a field recorder splits one continuous night recording across consecutive files (e.g. `ZOOM0004.WAV` -> `ZOOM0005.WAV` -> `...`), adjacent outputs must play seamlessly without level steps or filter clicks at the join:
1. **Pass neighbor seam padding on `process_take`:** Set `prev_audio_path` (for file `N > 1`, pointing to raw file `N-1`) and `next_audio_path` (for file `N < last`, pointing to raw file `N+1`) with `seam_context_s=8.0`. This primes IIR filters (`highpass_rumble`, `drone_notch`, `parametric_eq`), `soft_compress` RMS envelopes, and `true_peak_limiter` lookahead smoothly across every boundary before slicing back exact original sample boundaries.
2. **Use ONE unified fixed mastering gain across all segments (never independent per-file `loudness_normalize`):** Per-file normalization assigns different gains when one split segment happens to have fewer vehicles or quieter insects than its neighbor, causing an audible jump at the split seam. Instead, calibrate the shared tone chain + a single fixed post-compression `gain` + `true_peak_limiter(ceiling_dbtp=-1.0, release_ms=60.0)` on a representative preview so the combined sequence centers around `-14 LUFS`, then apply that exact chain uniformly to every file.
3. **Outer edges & continuous seams only:** Apply `fade_edges(fade_in_s=3.0, fade_out_s=0.0)` only on the first file, omit `fade_edges` on interior files, and apply `fade_edges(fade_in_s=0.0, fade_out_s=5.0)` only on the final file. Do **not** call `trim_range` or `cut_regions` across split boundaries.
4. **Always include the Clean Sections Report:** On both single finalized takes and multi-file continuous sequences, read `clean_sections` (or call `find_clean_sections(sequence=[...])`) and report by default the largest **untouched** (`0` localized edits or residual distractions) and **minimal-edit** stretches formatted in round YouTube ASMR durations (`15 min`, `30 min`, `45 min`, `1 hr`, `1.5 hr`, `2 hr`, `3 hr`, `4 hr`, `6 hr`, `8 hr`, ...) with cumulative and per-file local `HH:MM:SS` start/end times.

---

## 6. Feedback-to-Tuning Playbook

When `get_ab_feedback` returns `verdict`:

| Artifact Tag / User Comment | Cause | Adjustment |
| :--- | :--- | :--- |
| **`Metallic / swishy`** | `spectral_denoise` or over-deep broadband attenuation. | Remove `spectral_denoise` or drop `strength` to `0.15–0.25`. |
| **`Hollow gap`** | Fixed `band_attenuate` dug below bed or abrupt patch edge. | Switch to `band_attenuate(envelope_mode="adaptive")` or `ambience_patch` with longer per-span `crossfade_ms`. |
| **`Distraction still audible`** | Symmetric swell missed asymmetric peak or overlapped hive band. | Use `envelope_mode="adaptive"` so peak cut tracks the loudest frame, pad timestamps (`±1.0 s`), or `ambience_patch` from a verified clean `donor_start_s`. |
| **`Too dull / muffled` / bees thinned** | `highpass_rumble` cut into `200–300 Hz` buzz or cut too wide. | Lower `highpass_rumble.cutoff_hz` (`45–80 Hz`), keep `band_attenuate` `<180 Hz`, or add gentle `parametric_eq` `highshelf`. |
| **`Stereo image shifted`** | Donor patch came from asymmetric panned moment. | Inspect `step_metrics[].details.patched_spans` and pick a balanced `source_start_s` manually. |
| **`Pumping / breathing`** | Compressor ratio too high / fast release, or short swell fades. | Lower `soft_compress.ratio` (`1.5–1.8`), raise `release_ms` (`400 ms`), or lengthen `fade_ms` (`1500–3000 ms`). |

---

## 7. Recipe Templates

### Template A: Batched Vehicle Swells (Adaptive) + Drone Notch + Hive Protection
```json
[
  {"filter_name": "highpass_rumble", "params": {"cutoff_hz": 55.0, "order": 4}},
  {"filter_name": "drone_notch", "params": {"freqs_hz": [138.0], "q": 28.0, "attenuation_db": -22.0, "harmonics": 2}},
  {
    "filter_name": "band_attenuate",
    "params": {
      "low_hz": 40.0,
      "high_hz": 180.0,
      "attenuation_db": -20.0,
      "fade_ms": 1500.0,
      "envelope_mode": "adaptive",
      "spans": [
        {"start_s": 312.0, "end_s": 326.0},
        {"start_s": 1140.0, "end_s": 1158.0, "attenuation_db": -24.0}
      ]
    }
  }
]
```

### Template B: YouTube / ASMR Mastering Chain (`-14 LUFS`, Quiet-Source Safe)
*(Assumes a quiet ~`-45` to `-50 LUFS` field source; adjust `gain_db` so pre-compression RMS sits near `-28` to `-22 dBFS`.)*
```json
[
  {"filter_name": "gain", "params": {"gain_db": 22.0}},
  {"filter_name": "parametric_eq", "params": {"freq_hz": 160.0, "gain_db": 1.8, "q": 0.8, "filter_type": "lowshelf"}},
  {"filter_name": "parametric_eq", "params": {"freq_hz": 11000.0, "gain_db": 2.2, "q": 0.7, "filter_type": "highshelf"}},
  {"filter_name": "soft_compress", "params": {"threshold_db": -24.0, "ratio": 1.8, "attack_ms": 35.0, "release_ms": 350.0}},
  {"filter_name": "loudness_normalize", "params": {"target_lufs": -14.0, "true_peak_ceiling_dbtp": -1.0, "limit_peaks": true, "limiter_release_ms": 100.0}},
  {"filter_name": "fade_edges", "params": {"fade_in_s": 2.5, "fade_out_s": 4.0, "curve": "cosine"}}
]
```
