# Engineering & DSP Tuning Notes (`NOTES.md`)

Working notes on DSP design choices, tuning heuristics for ambient soundscapes, and session experiments.

---

## 1. DSP Implementation Notes (v0.1.0)

### Linked-Stereo Spectral Processing
- Computing independent spectral masks per channel causes subtle stereo shimmer ("phasiness") in wide stereo field recordings (rain, rustling leaves, ocean surf).
- Both `spectral_denoise` and `band_attenuate` compute a single cross-channel magnitude estimate (`mean` across channels), derive one smooth 2D `(freq_bins, frames)` gain mask, and multiply all channels by the exact same gain matrix.
- This keeps inter-channel level/phase relationships undisturbed.

### Zero-Phase Filtering (`sosfiltfilt` / `filtfilt`)
- `highpass_rumble` and `dehum_notch` use forward-backward filtering (`sosfiltfilt` / `filtfilt`) so low-frequency phase response stays linear and exact sample alignment with the parent take is preserved for `residual = parent - processed`.
- Note: Filter order in `sosfiltfilt` doubles effective rolloff slope (`order=4` -> 48 dB/octave equivalent). Keep `order=2` or `order=4` depending on how steep the wind cutoff needs to be.

### `ambience_patch` Auto-Donor Selection
- When `source_start_s` is omitted, `ambience_patch` scans sliding windows across the recording outside `[start_s, end_s]` (with a padding margin) and scores candidate windows by:
  - Peak-to-RMS ratio (penalizing spiky transients)
  - High-frequency derivative peaks (penalizing clicks)
  - Closeness of RMS to the median ambient bed
- Boundary matching scales donor amplitude subtly toward local boundary RMS and applies equal-power (`cos`/`sin`) crossfade edges (`crossfade_ms`, default `40 ms`) to prevent clicks or abrupt texture steps.

---

## 2. Ambient Tuning Heuristics for the AI Agent

| Recording Environment | Common Distractions | Recommended First Pass | Watch Out For (Residual Check) |
| :--- | :--- | :--- | :--- |
| **Rain / Surf** | Subsonic wind buffeting, stray traffic horn, mic drips | `highpass_rumble(55 Hz)` + `declick_impulse` + `ambience_patch` on horn | Heavy `spectral_denoise` turns rain into metallic static — avoid unless constant hiss exists. |
| **Forest / Birds** | Distant road hum, footsteps, cough, wind gust | `highpass_rumble(70 Hz)` + `band_attenuate` (`<250 Hz` during car pass) or `ambience_patch` | Keep `band_attenuate` below bird frequencies (`>1.5 kHz`) untouched. |
| **Indoor Room Tone / Café** | 50/60 Hz HVAC/mains hum, chair scrape, clear throat | `dehum_notch(50 or 60 Hz, harmonics=4)` + `ambience_patch` on scrape | Compare 50 Hz vs 60 Hz prominence in `analyze_take` (`hum_peaks_hz`) before notching! |
| **Windy Exterior** | Heavy `<120 Hz` gusts, mic diaphragm pop | `highpass_rumble(85-110 Hz)` + `declick_impulse` | If gusts saturate the mic (`clipping`), `ambience_patch` works better than filtering alone. |

---

## 3. Experiment Log

### 2026-11-20 — Synthetic Demo & v0.2.0 Streaming Architecture Notes
- Demo fixture simulates: stereo pink-noise rain bed + `60 Hz` + `120 Hz` hum + impulsive mic click + `950 Hz` distraction burst (`2.4s–3.1s`).
- Candidate A compares surgical `band_attenuate` (`750–1200 Hz`, `-22 dB`) against Candidate B `ambience_patch` (`2.35s–3.15s`).
- Listen to `Residual` (`hotkey 4`) in the A/B Studio to verify rain texture outside the distraction stays intact.
- **Bee-Hive & Passing-Vehicle Notes (`ZOOM0001.WAV` class):**
  - Honeybee hive buzzing concentrates fundamental energy around `200–300 Hz` plus upper harmonics, overlapping engine/tyre passes (`80–1000 Hz`).
  - Use `analyze_region(..., freq_range_hz=[35, 160], sensitivity="high", min_event_s=1.5)` to locate vehicle swells cleanly without thousands of `impulsive_click` entries crowding responses. Note that natural hive crackles/ticks are often desirable ASMR detail—avoid running `declick_impulse` indiscriminately on clean hive textures.
  - For asymmetric 5–20 s vehicle passes, prefer `band_attenuate(envelope_mode="adaptive", fade_ms=1500+)` or `ambience_patch` from clean bee donors.
  - For YouTube ASMR delivery (`-14 LUFS`) from quiet (`-45` to `-55 LUFS`) sources, stage `gain` before `soft_compress` (so transients reach compressor threshold) and/or enable `loudness_normalize(target_lufs=-14.0, true_peak_ceiling_dbtp=-1.0, limit_peaks=True)` + `fade_edges`.

### 2026-11-21 — `v0.3.2` Multi-File Split Continuity, RF64 Temps & Clean Sections
- **2 GB+ Split Recording Temp Sizes:** A 2h 04m stereo 48 kHz `PCM_24` split (`~2.15 GB`) expands to `~2.86 GB` as internal `FLOAT` temp stages (`4` bytes/sample), staying near or above standard RIFF limits once combined with multi-stage buffers. Using `_STREAM_TEMP_FORMAT = "RF64"` for all internal float temp stages and unlinking superseded `stage_NNN.wav` files immediately after each step keeps temp overhead minimal and prevents 4 GiB RIFF header corruption.
- **Unified Fixed Gain Across Continuous Splits:** Independent `loudness_normalize` per file shifts adjacent files differently based on local distraction density, introducing an audible step at the file join. Calibrating a single fixed post-compression `gain` + `true_peak_limiter` + `prev_audio_path`/`next_audio_path` seam context (`8.0 s`) preserves sample-level envelope continuity across split files.
- **Clean Section Discovery:** Scoring localized edits (`band_attenuate`, `spectral_match_suppress`, `ambience_patch`, cut joins) and residual localized distractions (`low_freq_rumble`, `transient_distraction`, `slow_pass`, `clipping`) while excluding `impulsive_click` and uniform global mastering identifies true untouched and minimal-edit stretches in round YouTube ASMR durations via `compute_clean_sections`.

### 2026-11-22 — `v0.5.3` Real-World Park Recording Detector & Narrowband Fixes
- **Gain-Invariant Piezo Floor & `n_fft=4096` `90 ms` Beep Bypass:** Testing `ZOOM0001.WAV` (Cape Town park sunrise, `-45.1 LUFS` raw -> `-17.1 LUFS` master) revealed two subtleties: (1) a static `-88 dBFS` noise gate in `detect_synthetic_beeps` multiplied false bird/impeller hits `10×` (`155 -> 1688`) once `+31 dB` mastering gain lifted quiet chirps above `-88 dBFS`; switching to `max(-110.0, band_bed_p25 + 8.0)` makes detection gain-invariant. (2) At `48 kHz` / `n_fft=4096` (`21.3 ms` hop), a `90 ms` Garmin Instinct 2 beep (`~2982 Hz`) spanned only 2–3 peak frames and skipped the `len >= 4` piezo check in `compute_bird_protect_mask`; lowering the check to `len >= 2` and evaluating interior frames allows `spectral_match_suppress(narrowband_mode=True, protect_birds=True)` to remove short piezo tones cleanly.

### 2026-11-21 — `v0.5.2` Multi-Hour Compile Verification & Guardrail Hardening
- **Empirical Field Audit (`v0.5.1` Beehive 5-File Run):** Streaming `compile_clean_sections` over the 9 h 44 m 50 s (`35,089.67 s`, `24-bit 48 kHz` stereo) beehive sequence exported all 5 full masters (`~9.4 GB`, `-14.42` to `-14.90 LUFS`) plus `15 min`, `30 min`, `1 hr` (`RIFF`), `2 hr` (`RIFF`, 2 slices), and `4 hr` (`RF64`, `4,147,200,104` B, 3 slices) clean sections.
- **Root-Cause Fixes Landed in `v0.5.2`:**
  - *Single-Pass WAV Master Copy (`copy_file_with_sha256`):* Exporting an unmodified WAV master previously spent ~16 s copying + ~120 s re-reading LUFS/peak/SHA-256 per 2.15 GB file; streaming copy+hash while reusing cached `TakeRecord` LUFS/true-peak cuts master export latency in half with `>= 2.0 s` intra-item heartbeats.
  - *Gain-Normalized `bird_retention_db`:* Normalizing processed audio RMS to the reference slice prior to bird-mask energy comparison (`gain_offset_db`, plus `bird_retention_raw_db`) ensures `+37 dB` master gains never mask genuine bird attenuation.
  - *File-Local Snapping & Provenance:* Strict preference hierarchy (`file-local minute < global minute < non-minute`) fixes `round_to_file_local_minutes=True` across `minimal_edits`/`moderate_edits` tiers, `0.001 s` rounding removes ~30 ms drift, `source_slices[]` identifies exact `source_file` / sample offsets, and `distinct_regions=True` prevents overlapping clean ASMR outputs.

### 2026-11-21 — `v0.5.1` Streaming Clean-Section Compilation & RF64 Output
- **Cross-Seam Clean ASMR Compilation (`compile_audio_span`):** Splitting/stitching multi-hour field sequences across file boundaries runs frame-accurately via block streaming (`block_seconds=30.0`) with single-pass ITU-R BS.1770 LUFS (`StreamingLUFS`) + true-peak accumulation (`StreamingTruePeak`) and outer raised-cosine fade-in/fade-out windows (`fade_in_s=2.0`, `fade_out_s=3.0`) while keeping internal split seams bit-exact (`> 2` GB RAM never allocated).
- **RF64 Auto-Container Threshold (`>= 3.8 GB`):** Combined 5+ hour 48 kHz stereo `PCM_24` WAV files exceed standard 4 GiB RIFF headers (~5.1 GB for ~9.7 hr). Both `export_audio` and `compile_audio_span` automatically select `"RF64"` whenever estimated uncompressed WAV byte count reaches `3_800_000_000` bytes.

### 2026-11-21 — `v0.5.0` Outdoor/Nature Restoration, Hardware Spatial/Backup & Async Jobs
- **Content-Defined FM Bird Protection vs Synthetic Beeps:** Rather than cutting off suppression above `3.5 kHz` (which protects 4 kHz watch beeps and misses 1.5–3 kHz bird phrases), `compute_bird_protect_mask` identifies modulated chirp ridges across `300 Hz – 11.5 kHz` and uses sub-bin parabolic frequency tracking (`f_ptp`) to reject zero-drift electronic piezo tones (`f_center >= 1400 Hz` or `mid_std < 1.4 dB`). In `spectral_match_suppress(narrowband_mode=True)`, using unblurred `np.maximum(mag_mean, mag_s_mean)` keeps pinpoint spike excess accurate before Gaussian smoothing.
- **Short-Clip Multi-Segment Tonal Persistence (`>= 2.0 s`):** Extending multi-segment Welch PSD median persistence down to `>= 2.0 s` recordings prevents brief FM bird chirps from falsely populating `tonal_peaks_hz` / `machine_drone` on 2–15 s audition regions while still catching true persistent tones.
- **Hardware Workflows (`declip_from_backup` & `spatial_mix`):** GCC-PHAT estimates sub-sample offset (`lag_samples`) between clipped main tracks and `-12 dB` safety tracks (`fractional_shift_channel` via 31-tap Blackman-windowed sinc), fits masked least-squares FIR eq/gain calibration on unclipped frames, splices clipped runs with raised cosines, and falls back to 16th-order bidirectional AR interpolation if the safety channel also clipped. `detect_channel_layout` infers `raw_ms` (Zoom SSH-6 M/S with figure-8 bass rolloff), `directional_plus_surround` (Zoom SGH-6 + surround), `dual_mono`, or `stereo_lr` for deterministic chunk-safe `spatial_mix`.
- **Sweeping Harmonics, Distancing & Comfort Shaping:** `harmonic_track_suppress` tracks `f0(t)` combs (`variable_rpm_engine` leafblowers) with optional L/R bias weighting and `protect_birds`; `distance_soften` recedes barks and speech via HF air absorption, onset smearing, and formant blur; `bed_shape` warms and de-harshes only the 4 s slow stationary bed (`sharpness_acum`, `pleasantness_score`) without touching FM transients.
- **Slew-Normalized Seam Step Gate (`max_sample_step_excess_db=18.0`):** Continuous `-16 LUFS` broadband recordings naturally exceed `-30 dBFS` sample-to-sample steps between adjacent samples; gating sample steps against the 2 s boundary `np.diff` RMS catches true DC/splice glitches while avoiding false failures on healthy loud beds.

### 2026-11-21 — `v0.4.0` Spectral Excess Suppression, Non-Repeating Donors & Studio Playhead
- **Adaptive STFT Spectral Excess Suppression (`spectral_match_suppress`, `n_fft=4096`):** Unlike static broadband `band_attenuate` cuts that leave moving vehicle exhaust harmonics (`50–300 Hz`) audible or hollow out non-distracted hive bands (`200–300 Hz`), estimating per-bin clean neighbor medians (`bed_med`) and 80th-percentile envelopes (`bed_p80`) attenuates only time-frequency bins exceeding the clean bed and subtly fills deeply suppressed bins (`texture_fill`) from clean context. Benchmarks on three overlapping field passes reduced narrowband excess peaks:
  - Moving car exhaust pass (`90–130 s` slice): max narrowband excess `+13.2 dB -> +3.3 dB` (`50–100 Hz: +12.1 -> +0.1 dB`, `100–180 Hz: +5.2 -> -0.2 dB`).
  - Train / horn pass (`4720–4780 s` slice): max narrowband excess `+17.0 dB -> +7.8 dB` (`50–100 Hz: +15.6 -> +5.5 dB`, `100–180 Hz: +6.6 -> +0.1 dB`).
  - Passing vehicle (`940–995 s` slice): max narrowband excess `+12.7 dB -> -0.3 dB` (`50–100 Hz: +10.2 -> -1.8 dB`, `100–180 Hz: +9.1 -> -1.3 dB`).
- **`ambience_patch` v2 Donor Pool:** Full-file single-pass `float32` scanning computes `(win_rms, win_peak, win_low_ratio)` in ~288 KB RAM for hour-long files, scoring contiguous multi-second donor candidates while masking out localized `exclude_events` (`±5.0 s`) and previously chosen `used_spans` so repeated patches in one recipe never cycle identical donor loops, matched with RMS + 2-band (`250 Hz` split) spectral tilt equalization.
- **UI Windowed Audition & Playhead Sync:** PNG sidecar `.meta.json` coordinates exposed via `X-Plot-X0`/`X-Plot-X1`/`X-Time-Start`/`X-Time-End` HTTP headers map `#playheadLine` pixel-accurately onto the Matplotlib axes box across live playback, loop bounds, and click/drag seeks. Fixing the missing `?` in `loadAudioWindow()` and always slicing `Original` even at `original_offset_s == 0.0` unblocked playback on start-of-file previews.

### 2026-11-20 — `v0.3.1` Calibration & Detector Notes
- **Median-PSD & Constant-Q Tonal Detection:** A brief loud vehicle harmonic (e.g., `167 Hz`) inflates the mean Welch PSD across an entire long file even when absent 95% of the time. Computing elementwise median PSD across segments prevents transient bursts from seeding false whole-file `machine_drone` candidates, while constant-Q sidebands (`inner = max(6, 0.03 * fc)`, `outer = max(22, 0.12 * fc)`) extend reliable steady tonal detection up to `8000 Hz`.
- **Dual-Timescale `slow_pass` + Hysteresis:** Short `50 ms` RMS windows have high variance on live nature beds (`mad ~ 1.8–2.2 dB`), raising thresholds too high for distant `4–7 dB` vehicles. Adding a `1.0 s` uniform-smoothed track (`mad ~ 0.4–0.6 dB`) with hysteresis edge expansion (`med + 1.8 dB`) captures full vehicle swells and trailing tails cleanly.
- **Ambient Bed vs High-Frequency Hiss Floor:** On continuous soundscapes (rain, hives, surf), 10th-percentile broadband RMS measures the continuous ambient bed rather than quiet gaps; pairing `ambient_bed_dbfs` with `> 6 kHz` `high_freq_floor_dbfs` distinguishes true bed level from preamp hiss.

### 2026-11-20 — `v0.3.0` Calibration & Detector Notes
- **Bilateral Spectral Prominence:** Steep high-pass filters (`45 Hz`, `order=4`) roll off the lower sideband around `50 Hz`, causing one-sided median prominence (`peak - combined_neighbor_median`) to mistake the filter shoulder for `50 Hz` mains hum. Computing `min(peak - left_median, peak - right_median)` eliminates false hum triggers while detecting real 50/60 Hz tones.
- **Persistent Tonal Drone Separation:** Lowering tonal prominence to `> 7.0 dB` catches steady `7–11 dB` machine drones, while requiring bilateral prominence `>= 4.5 dB` in `>= 60%` of time segments separates continuous drones across long files from brief 15 s passing vehicles.
- **Detector Floor `-92.0 dBFS`:** Quiet `-50 LUFS` field recordings filtered into narrow sub-bands routinely fall below `-55` to `-60 dBFS`. Moving the noise gate floor to `-92.0 dBFS` restores sensitivity and band-limited event discovery on quiet sources.
