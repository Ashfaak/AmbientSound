# AmbientSound MCP — Agent Operational Instructions (`v0.2.0`)

> **Handover Guide for Any AI Model / MCP Agent**
> Use this document as your complete operational manual for the **AmbientSound** MCP server—covering preflight verification, short and multi-hour (672 MB+) field recordings, wildlife/hive + traffic/machine cleanup, loudness-matched A/B listening feedback, and YouTube/ASMR mastering.

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
2. **Patch, cut, or smoothly swell—don't hard-mute.**
   - Use **`band_attenuate`** with `envelope_mode="swell"` and long `fade_ms` (`800–3000 ms`) for slow `5–20 s` passing cars/motorbikes when they occupy a distinct band.
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

## 3. MCP Tool Reference (13 Tools)

All tools return JSON objects (`dict`). Unmeasurable metrics (e.g., LUFS on `<0.45 s` or silence) return `null` instead of `NaN`/`±inf`.

| Tool | Arguments | Purpose & Return |
| :--- | :--- | :--- |
| `health_check` | `()` | Confirms `status`, `version`, `allowed_roots`, `ffmpeg_available`, `workspace_free_gb`, `streaming_threshold_s` (`120.0 s`), and `filter_count` (`14`). |
| `list_filters` | `()` | Returns `{"filters": [...]}` with exact parameter schemas, bounds, and defaults. |
| `create_session` | `(audio_path, name="")` | Ingests audio into `take_000` in bounded RAM (`shutil.copy2` for WAV; streaming `PCM_24` otherwise), runs streaming analysis, renders overview PNG. |
| `list_sessions` | `()` | Returns `{"sessions": [...]}` summary list. |
| `get_session_summary` | `(session_id)` | Full `SessionSummary` (`takes`, `metrics`, `events`, `ab_tests`). |
| `analyze_take` | `(session_id, take_id, sensitivity="medium", min_event_s=0.12, max_event_s=45.0, freq_range_hz=None)` | Full-file streaming analysis with configurable sensitivity (`"low"`, `"medium"`, `"high"`), duration bounds, and optional `[low_hz, high_hz]` target band filter. |
| `analyze_region` | `(session_id, take_id, start_s=0.0, end_s=None, sensitivity="medium", min_event_s=0.12, max_event_s=45.0, freq_range_hz=None)` | Analyzes a specific `[start_s, end_s]` window and returns absolute recording timestamps (`+ start_s`). |
| `render_spectrogram` | `(session_id, take_id, compare_to_take_id="", start_s=None, end_s=None)` | Renders full overview (`>60 s` sampled) or zoomed high-res (`<=60 s` slice) waveform+spectrogram and optional Before/After/Removed Residual PNGs. |
| `process_take` | `(session_id, parent_take_id, steps, label="", notes="")` | Runs DSP recipe (automatically streams file-to-file when parent `>120 s`), writes immutable `take_NNN`, analyzes output, renders visuals. |
| `create_ab_test` | `(session_id, take_a_id, take_b_id, focus_start_s=None, focus_end_s=None, blind=False, notes="")` | Creates attenuation-only LUFS-matched A/B test (`gain_db_* <= 0.0`), `ui_url`, and active `workspace_dir`. |
| `list_ab_tests` | `(session_id)` | Returns `{"ab_tests": [...]}` with verdicts and `ui_url`s. |
| `get_ab_feedback` | `(session_id, ab_id)` | Retrieves `verdict` (`winner`, `artifact_tags`, `comment`, `submitted_at`). |
| `export_take` | `(session_id, take_id, output_path, format="wav")` | Stream-exports take to `wav` (24-bit), `flac` (24-bit), `mp3` (320k), or `ogg`. |

---

## 4. Built-In DSP Filter Reference (14 Filters)

| Filter | Key Parameters (Defaults & Bounds) | Ambient & Mastering Usage |
| :--- | :--- | :--- |
| `highpass_rumble` | `cutoff_hz` (`60.0`, `15–300`), `order` (`4`, `1–8`) | Clears wind/traffic sub-rumble. `80–120 Hz` protects `200–300 Hz` bee buzz while removing low road rumble. |
| `dehum_notch` | `base_freq_hz` (`60.0`), `harmonics` (`4`), `q` (`30.0`) | Zero-phase cascaded notches for `50/60 Hz` mains hum (`hum_peaks_hz`). |
| `drone_notch` | **`freqs_hz`** (`list[float]`), `q` (`25.0`, `2–80`), `attenuation_db` (`-24.0`, `-48 to -1`), `harmonics` (`1`, `1–6`) | Proportional zero-phase peaking cuts at arbitrary machine/engine frequencies (`tonal_peaks_hz`) without touching nearby hive harmonics. |
| `declick_impulse` | `threshold_sigma` (`6.0`), `max_width_ms` (`3.0`) | PCHIP spike repair for mic clicks, drips, static pops. |
| `spectral_denoise` | `strength` (`0.45`), `floor_db` (`-18.0`), `n_fft` (`2048`), `noise_start_s`, `noise_end_s` | Steady broadband preamp hiss only. Keep `strength <= 0.30` on live ambience. |
| `band_attenuate` | `start_s`, `end_s`, `low_hz` (`40`), `high_hz` (`250`), `attenuation_db` (`-18`), `fade_ms` (`30`, up to `5000`), `envelope_mode` (`"flat"` \| `"swell"`), `spans` (`list[dict]`) | Single or **batched multi-span** time-frequency attenuation. Use `envelope_mode="swell"` + `fade_ms=1500` for slow `5–20 s` vehicle passes. |
| `ambience_patch` | `start_s`, `end_s`, `source_start_s` (`null`), `crossfade_ms` (`40`, up to `5000`), `spans` (`list[dict]`) | Single or **batched multi-span** texture replacement from clean donor audio outside all spans. |
| `trim_range` | `start_s` (`0.0`), `end_s` (`null`) | Extracts `[start_s, end_s]`. Ideal for fast 30 s preview iteration on 1-hour files or removing mic setup/teardown. |
| `cut_regions` | **`regions`** (`[{"start_s", "end_s"}, ...]`), `crossfade_ms` (`1000`, `10–5000`) | Removes severe bad stretches and stitches remaining segments with equal-power crossfades. |
| `fade_edges` | `fade_in_s` (`2.0`, `0–60`), `fade_out_s` (`3.0`, `0–60`), `curve` (`"cosine"` \| `"linear"`) | Smooth start/end fades for finished exports. |
| `parametric_eq` | **`freq_hz`**, **`gain_db`**, `q` (`1.4`), `filter_type` (`"peaking"` \| `"lowshelf"` \| `"highshelf"`) | Shape tone: e.g. `+2 dB lowshelf @ 160 Hz` (warmth) and `+2.5 dB highshelf @ 11000 Hz` (ASMR air). |
| `soft_compress` | `threshold_db` (`-24`), `ratio` (`2.0`), `attack_ms` (`25`), `release_ms` (`250`), `makeup_db` (`0.0`) | Gentle linked-stereo RMS compression for consistent ASMR playback without pumping. |
| `true_peak_limiter` | `ceiling_dbtp` (`-1.0`), `release_ms` (`80.0`) | Lookahead peak limiter enforcing strict ceiling (`-1.0 dBTP`) for YouTube loudness. |
| `loudness_normalize` | `target_lufs` (`-18.0`, `-40 to -6`), `true_peak_ceiling_dbtp` (`-1.0`) | 2-pass streaming BS.1770 normalization. Use **`-14.0 LUFS`** for YouTube/ASMR, or `-18` to `-22 LUFS` for quiet background ambience. |

---

## 5. Large-File (1-Hour / 672 MB+) & Bee-Hive Field Workflow

When `duration_s > 120.0` (e.g., an hour-long `48 kHz / 24-bit` WAV), AmbientSound automatically operates in **streaming mode** (`< 150 MB` RAM):
- **Do NOT try full-file trial-and-error first.** Writing 1-hour takes writes ~672 MB per take on disk and takes ~10–30 s per pass. Instead:
  1. Call `create_session(audio_path)` and inspect `take_000` `metrics` (`hum_peaks_hz`, `tonal_peaks_hz`, `rumble_ratio_db`, `events`). Notice `slow_pass` events (`>= 3.0 s`) flag passing vehicles, and `machine_drone` flags continuous tonal machinery.
  2. Call `analyze_take` or `analyze_region` with `freq_range_hz=[35, 160]` and `sensitivity="high"` to hunt specifically for sub-hive traffic/vehicle swells **without** bee buzz triggering false positives!
  3. Call `render_spectrogram(..., start_s=..., end_s=...)` over `20–45 s` windows around interesting timestamps (`<= 60 s` renders full high-resolution STFT; `> 60 s` renders a fast sampled overview).
  4. **Prototype on a 30–60 s preview slice first!** Create a child take with `trim_range(start_s=..., end_s=...)` covering a tough vehicle pass + clean bees, branch Candidate A (`highpass_rumble` + `drone_notch` + `band_attenuate` swell) vs Candidate B (`highpass_rumble` + `drone_notch` + `ambience_patch` or `cut_regions`), and run a quick A/B test on the preview take!
  5. Once the user approves the recipe/parameters in the preview A/B test, apply the full-file chain cleanly from `take_000` using **batched `spans`** (for `band_attenuate` / `ambience_patch`) or **`cut_regions`**, followed by mastering (`parametric_eq` + `soft_compress` + `loudness_normalize` + `true_peak_limiter` + `fade_edges`).
  6. **A/B Studio on Long Files:** When `create_ab_test` runs on a take `> 90 s`, always pass `focus_start_s` and `focus_end_s` around a concrete cleaned event—the browser automatically enters **Windowed Audition Mode** (loading up to `90 s` slices + dynamic sliced residual via hotkey `4` so the browser never runs out of memory).

---

## 6. Feedback-to-Tuning Playbook

When `get_ab_feedback` returns `verdict`:

| Artifact Tag / User Comment | Cause | Adjustment |
| :--- | :--- | :--- |
| **`Metallic / swishy`** | `spectral_denoise` or over-deep broadband attenuation. | Remove `spectral_denoise` or drop `strength` to `0.15–0.25`. |
| **`Hollow gap`** | Deep wide `band_attenuate` or abrupt patch boundary. | Switch to `ambience_patch(crossfade_ms=1000)` or `cut_regions`, or soften `band_attenuate` (`-10` to `-14 dB`, `envelope_mode="swell"`). |
| **`Distraction still audible`** | Range/threshold too narrow or frequencies overlap hive. | Use `analyze_region(..., freq_range_hz=[35, 160])` to pad timestamps (`±1.0 s`), add `drone_notch` on `tonal_peaks_hz`, or cut/patch severe vehicle passes. |
| **`Too dull / muffled` / bees thinned** | `highpass_rumble` cut into `200–300 Hz` buzz or cut too wide. | Keep `highpass_rumble.cutoff_hz <= 110 Hz`, narrow `band_attenuate` below `180 Hz`, or restore air with `parametric_eq` `highshelf`. |
| **`Stereo image shifted`** | Donor patch came from asymmetric panned moment. | Provide explicit balanced `source_start_s` and raise `crossfade_ms` (`500–1500 ms`). |
| **`Pumping / breathing`** | Compressor ratio too high / fast release, or short swell fades. | Lower `soft_compress.ratio` (`1.5–1.8`), raise `release_ms` (`400 ms`), or lengthen `fade_ms` (`1500–3000 ms`). |

---

## 7. Recipe Templates

### Template A: Batched Vehicle Swells + Machine Drone + Hive Protection
```json
[
  {"filter_name": "highpass_rumble", "params": {"cutoff_hz": 95.0, "order": 4}},
  {"filter_name": "drone_notch", "params": {"freqs_hz": [138.0], "q": 28.0, "attenuation_db": -22.0, "harmonics": 2}},
  {"filter_name": "declick_impulse", "params": {"threshold_sigma": 6.0, "max_width_ms": 3.0}},
  {
    "filter_name": "band_attenuate",
    "params": {
      "low_hz": 70.0,
      "high_hz": 220.0,
      "attenuation_db": -16.0,
      "fade_ms": 1800.0,
      "envelope_mode": "swell",
      "spans": [
        {"start_s": 312.0, "end_s": 326.0},
        {"start_s": 1140.0, "end_s": 1158.0}
      ]
    }
  }
]
```

### Template B: YouTube / ASMR Mastering Chain (`-14 LUFS`, Warmth + Air)
```json
[
  {"filter_name": "parametric_eq", "params": {"freq_hz": 160.0, "gain_db": 1.8, "q": 0.8, "filter_type": "lowshelf"}},
  {"filter_name": "parametric_eq", "params": {"freq_hz": 11000.0, "gain_db": 2.2, "q": 0.7, "filter_type": "highshelf"}},
  {"filter_name": "soft_compress", "params": {"threshold_db": -24.0, "ratio": 1.8, "attack_ms": 35.0, "release_ms": 350.0}},
  {"filter_name": "loudness_normalize", "params": {"target_lufs": -14.0, "true_peak_ceiling_dbtp": -1.2}},
  {"filter_name": "true_peak_limiter", "params": {"ceiling_dbtp": -1.0, "release_ms": 100.0}},
  {"filter_name": "fade_edges", "params": {"fade_in_s": 2.5, "fade_out_s": 4.0, "curve": "cosine"}}
]
```
