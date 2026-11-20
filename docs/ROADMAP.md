# Roadmap

## v0.1.0 — Core CPU DSP + MCP + Loudness-Matched A/B Studio
- [x] Apache-2.0 project scaffold, license hygiene checks, CI workflow, documentation & memory suite
- [x] 32-bit float non-destructive take DAG + exact `residual.wav` computation
- [x] Psychoacoustic analyzer (LUFS, 4x True Peak, stereo correlation, 50/60 Hz hum peaks, rumble ratio, click & transient event detectors)
- [x] 8 linked-stereo CPU DSP filters including `band_attenuate` and `ambience_patch`
- [x] Headless waveform + spectrogram & 3-panel before/after/residual diff PNG renderer
- [x] Path-sandboxed `stdio` MCP server (`MCPServer`) with 11 tools
- [x] FastAPI + Web Audio API local A/B studio (time-locked, attenuation-only LUFS match, loop spans, blind mode, verdict feedback log)

## v0.2.0 — Large-File Streaming, Editing, Mastering & Configurable Detection
- [x] Low-RAM streaming I/O (`probe_audio`, `load_audio_slice`, `ingest_audio_to_wav`), `StreamingLUFS`, `chunked_true_peak_linear`, and `apply_recipe_streaming` (`>120 s` takes)
- [x] Configurable `analyze_take` / `analyze_region` (`sensitivity`, `freq_range_hz`, `slow_pass` >= 3 s, `machine_drone` arbitrary tonal peak detection)
- [x] 6 new filters (`drone_notch`, `trim_range`, `cut_regions`, `fade_edges`, `soft_compress`, `true_peak_limiter`) + multi-span `spans` + `swell` envelope + `5000 ms` fades
- [x] Windowed & sampled-overview spectrograms (`start_s`/`end_s`) + Windowed Audition Mode (`<=90 s` slices, dynamic sliced residual) + `health_check` MCP tool

## v0.3.0 — Adaptive Mastering, Timeline-Aware A/B & Field Reliability (Current)
- [x] `gain` filter + `loudness_normalize(limit_peaks=True)` peak-limited `-14 LUFS` mastering (in-memory & streaming)
- [x] `band_attenuate(envelope_mode="adaptive")`, donor provenance (`ambience_patch`), and per-step `StepTelemetry` (`step_metrics`)
- [x] Bilateral hum/drone prominence, `>= 60%` multi-segment tonal persistence, `-92.0 dBFS` quiet-recording floor, `min_event_s` click gating, LRA (`loudness_range_lu`), `noise_floor_dbfs`, and short-term LUFS timeline
- [x] Per-take `events.json` archival, compact MCP payloads (`max_events`, `event_kinds`, `include_events`), `n_fft=8192` spectrograms, `delete_take` / `delete_session`
- [x] Timeline-aware `original_take_id` LCA auto-resolution, candidate-first A/B matching (`match_original=False`), and gain-matched dynamic residual auditioning

## v0.4.0 — Semantic Distraction Detection (CPU ONNX)
- [ ] Optional `ambient-sound[ml-cpu]` extra (`onnxruntime`, Apache-2.0 / MIT compatible)
- [ ] Silero VAD (`MIT`) integration to label speech vs non-speech distractions
- [ ] YAMNet (`Apache-2.0`) or PANNs (`MIT`) window tagging to classify events (`Speech`, `Siren`, `Dog`, `Traffic`, `Rain`, `Wind`)
- [ ] Interactive waveform/spectrogram clicking in Web UI to mark manual distraction spans & send directly to MCP context

## v0.5.0 — Neural Separation & Ecosystem Bridges
- [ ] Inverted speech subtraction via open-source speech extraction (`DeepFilterNet`, MIT/Apache-2.0)
- [ ] Text-queried sound extraction (`AudioSep` / `CLAPSep`, MIT) as optional GPU/high-RAM backend
- [ ] Strict license opt-in guard (`AMBIENTSOUND_ALLOW_NONCOMMERCIAL=1`) for any non-commercial or restricted community weights
- [ ] Multi-take history tree graph visualization in the A/B Studio
- [ ] Optional external `ferraus` CLI backend adapter (`Apache-2.0 OR MIT`) for ultra-fast Rust processing
