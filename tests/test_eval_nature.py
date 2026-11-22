"""Synthetic ground-truth evaluation tests for AmbientSound v0.5.0 outdoor/nature workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal

from ambient_sound.analysis import analyze_audio, compute_clean_sections, summarize_events
from ambient_sound.config import Settings
from ambient_sound.dsp import (
    apply_recipe,
    apply_recipe_streaming,
    apply_recipe_with_telemetry,
    list_filter_specs,
)
from ambient_sound.engines import (
    assess_do_no_harm,
    compute_bird_protect_mask,
    compute_comparison_guardrails,
    detect_channel_layout,
    detect_synthetic_beeps,
    fractional_shift_channel,
)
from ambient_sound.io_audio import load_audio, save_wav
from ambient_sound.models import DistractionEvent, FilterStep
from ambient_sound.session import SessionManager


def _rms_db(x: np.ndarray) -> float:
    return 20.0 * float(np.log10(max(float(np.sqrt(np.mean(x * x))), 1e-12)))


def _pinkish_bed(n: int, sr: int, seed: int = 1, level: float = 0.008) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, 1.0, size=(n, 2))
    sos = signal.butter(1, 350.0, btype="lowpass", fs=sr, output="sos")
    low = signal.sosfilt(sos, white, axis=0)
    bed = 0.7 * low + 0.18 * white
    cur = float(np.sqrt(np.mean(bed * bed))) + 1e-12
    return bed * (level / cur)


def _add_fm_bird_chirp(
    stereo: np.ndarray,
    sr: int,
    start_s: float,
    dur_s: float = 0.14,
    f0_hz: float = 2800.0,
    f1_hz: float = 5200.0,
    amp: float = 0.08,
) -> None:
    i0 = max(0, int(round(start_s * sr)))
    n_c = min(stereo.shape[0] - i0, max(32, int(round(dur_s * sr))))
    if n_c <= 0:
        return
    t = np.arange(n_c, dtype=np.float64) / sr
    freq_inst = f0_hz + (f1_hz - f0_hz) * np.sin(np.pi * t / max(dur_s, 1e-4))
    phase = 2.0 * np.pi * np.cumsum(freq_inst) / sr
    env = np.hanning(n_c)
    chirp = amp * env * np.sin(phase)
    stereo[i0 : i0 + n_c, 0] += chirp
    stereo[i0 : i0 + n_c, 1] += chirp * 0.92


def test_detect_and_remove_garmin_beep_preserves_birds() -> None:
    sr = 44100
    n = sr * 6
    audio = _pinkish_bed(n, sr, seed=101, level=0.006)
    _add_fm_bird_chirp(audio, sr, 0.8, 0.15, 2900.0, 5100.0, amp=0.07)
    _add_fm_bird_chirp(audio, sr, 4.7, 0.15, 3100.0, 5400.0, amp=0.07)

    # Zero-drift 4096 Hz Garmin watch beep burst inside [2.10, 2.55]
    t = np.arange(n, dtype=np.float64) / sr
    for b0, b1 in ((2.10, 2.24), (2.34, 2.48)):
        m = (t >= b0) & (t < b1)
        s0, s1 = int(b0 * sr), int(b1 * sr)
        env = np.hanning(s1 - s0)
        tone = 0.14 * env * np.sin(2.0 * np.pi * 4096.0 * t[m])
        audio[s0:s1, 0] += tone
        audio[s0:s1, 1] += tone

    _, events = analyze_audio(audio, sr, sensitivity="high", min_event_s=0.08)
    assert any(ev.kind == "synthetic_beep" for ev in events)

    proc = apply_recipe(
        audio,
        sr,
        [
            FilterStep(
                filter_name="spectral_match_suppress",
                params={
                    "start_s": 2.0,
                    "end_s": 2.6,
                    "low_hz": 2200.0,
                    "high_hz": 6500.0,
                    "narrowband_mode": True,
                    "protect_birds": True,
                    "context_s": 1.2,
                    "context_gap_s": 0.25,
                    "fade_ms": 40.0,
                    "max_attenuation_db": -45.0,
                },
            )
        ],
    )
    before_beep = _rms_db(audio[int(2.12 * sr) : int(2.22 * sr)])
    after_beep = _rms_db(proc[int(2.12 * sr) : int(2.22 * sr)])
    assert (before_beep - after_beep) >= 12.0

    # Bird syllables outside the beep must remain virtually untouched
    bird_b = audio[int(0.75 * sr) : int(1.0 * sr)]
    bird_a = proc[int(0.75 * sr) : int(1.0 * sr)]
    guard = compute_comparison_guardrails(bird_b, bird_a, sr)
    if guard["bird_retention_db"] is not None:
        assert guard["bird_retention_db"] >= -1.5


def test_detect_and_repair_cable_tie_and_plastic_bag() -> None:
    sr = 44100
    n = sr * 6
    rng = np.random.default_rng(202)
    audio = _pinkish_bed(n, sr, seed=202, level=0.005)
    _add_fm_bird_chirp(audio, sr, 5.1, 0.14, 3000.0, 5200.0, amp=0.07)

    # Cable-tie zip: rapid broadband click train (~20 Hz) in 1.1..1.8 s
    for k in range(14):
        idx = int((1.12 + k * 0.048) * sr)
        burst = 0.45 * np.exp(-np.arange(40) / 6.0) * rng.choice([-1.0, 1.0])
        audio[idx : idx + 40, 0] += burst
        audio[idx : idx + 40, 1] += burst * 0.9

    # Plastic bag rustle: broadband HF crackle in 3.2..4.1 s
    sos_hf = signal.butter(4, 3000.0, btype="highpass", fs=sr, output="sos")
    hf_len = int(0.85 * sr)
    hf = signal.sosfilt(sos_hf, rng.normal(0.0, 0.08, size=(hf_len, 2)), axis=0)
    s_b = int(3.25 * sr)
    audio[s_b : s_b + hf_len] += hf * np.hanning(hf_len)[:, np.newaxis]

    _, events = analyze_audio(audio, sr, sensitivity="high", min_event_s=0.1)
    kinds = {ev.kind for ev in events}
    assert "click_train" in kinds or "impulsive_click" in kinds
    assert "crackle_burst" in kinds

    proc = apply_recipe(
        audio,
        sr,
        [
            FilterStep(
                filter_name="declick_impulse",
                params={
                    "spans": [{"start_s": 1.05, "end_s": 1.9}],
                    "method": "ar",
                    "threshold_sigma": 4.0,
                    "max_width_ms": 8.0,
                },
            ),
            FilterStep(
                filter_name="spectral_match_suppress",
                params={
                    "start_s": 3.2,
                    "end_s": 4.2,
                    "low_hz": 1800.0,
                    "high_hz": 22000.0,
                    "protect_birds": True,
                    "texture_fill": 0.25,
                    "context_s": 1.2,
                    "context_gap_s": 0.25,
                    "fade_ms": 60.0,
                },
            ),
        ],
    )
    zip_pk_before = float(np.max(np.abs(audio[int(1.1 * sr) : int(1.85 * sr)])))
    zip_pk_after = float(np.max(np.abs(proc[int(1.1 * sr) : int(1.85 * sr)])))
    assert zip_pk_after < zip_pk_before * 0.65

    bag_before = _rms_db(audio[int(3.4 * sr) : int(3.9 * sr)])
    bag_after = _rms_db(proc[int(3.4 * sr) : int(3.9 * sr)])
    assert (bag_before - bag_after) >= 6.0


def test_detect_and_soften_dog_bark_and_human_voice() -> None:
    sr = 44100
    n = sr * 8
    t = np.arange(n, dtype=np.float64) / sr
    audio = _pinkish_bed(n, sr, seed=303, level=0.005)

    # Percussive harmonic dog barks at 1.2, 1.65, 2.1 s
    for b_t in (1.2, 1.65, 2.1):
        i0 = int(b_t * sr)
        ln = int(0.14 * sr)
        tt = np.arange(ln, dtype=np.float64) / sr
        env = np.exp(-tt * 18.0) * (1.0 - np.exp(-tt * 200.0))
        bark = 0.22 * env * (
            np.sin(2.0 * np.pi * 440.0 * tt)
            + 0.7 * np.sin(2.0 * np.pi * 880.0 * tt)
            + 0.4 * np.sin(2.0 * np.pi * 1320.0 * tt)
        )
        audio[i0 : i0 + ln, 0] += bark
        audio[i0 : i0 + ln, 1] += bark

    # Modulated human speech segment 4.3..6.3 s
    s_v0, s_v1 = int(4.3 * sr), int(6.3 * sr)
    tv = t[s_v0:s_v1] - 4.3
    syll = 0.5 + 0.5 * np.sin(2.0 * np.pi * 4.2 * tv)
    voice = 0.14 * syll * np.hanning(tv.size) * (
        np.sin(2.0 * np.pi * 150.0 * tv)
        + 0.8 * np.sin(2.0 * np.pi * 450.0 * tv)
        + 0.6 * np.sin(2.0 * np.pi * 900.0 * tv)
        + 0.5 * np.sin(2.0 * np.pi * 1500.0 * tv)
    )
    audio[s_v0:s_v1, 0] += voice
    audio[s_v0:s_v1, 1] += voice

    _, events = analyze_audio(audio, sr, sensitivity="high", min_event_s=0.1)
    kinds = {ev.kind for ev in events}
    assert "dog_bark" in kinds or "transient_distraction" in kinds
    assert "human_voice" in kinds

    proc = apply_recipe(
        audio,
        sr,
        [
            FilterStep(
                filter_name="distance_soften",
                params={
                    "spans": [
                        {
                            "start_s": 1.1,
                            "end_s": 2.4,
                            "mode": "bark_distance",
                            "attenuation_db": -16.0,
                            "context_s": 1.0,
                            "context_gap_s": 0.2,
                            "fade_ms": 80.0,
                        },
                        {
                            "start_s": 4.2,
                            "end_s": 6.4,
                            "mode": "voice_muffle",
                            "attenuation_db": -16.0,
                            "context_s": 1.0,
                            "context_gap_s": 0.2,
                            "fade_ms": 80.0,
                        },
                    ],
                    "protect_birds": True,
                },
            )
        ],
    )
    assert (_rms_db(audio[int(1.15 * sr) : int(2.3 * sr)]) - _rms_db(
        proc[int(1.15 * sr) : int(2.3 * sr)]
    )) >= 5.0
    assert (_rms_db(audio[s_v0:s_v1]) - _rms_db(proc[s_v0:s_v1])) >= 5.0


def test_detect_and_suppress_variable_rpm_leafblower(tmp_path: Path) -> None:
    sr = 44100
    n = sr * 8
    t = np.arange(n, dtype=np.float64) / sr
    audio = _pinkish_bed(n, sr, seed=404, level=0.005)
    _add_fm_bird_chirp(audio, sr, 0.8, 0.14, 2900.0, 5000.0, amp=0.08)

    # Sweeping RPM comb (120 -> 230 -> 140 Hz), left-channel biased (+9 dB on L)
    s0, s1 = int(2.0 * sr), int(5.4 * sr)
    tb = t[s0:s1] - 2.0
    f0_t = 120.0 + 100.0 * np.sin(np.pi * tb / 3.4)
    phase = 2.0 * np.pi * np.cumsum(f0_t) / sr
    env = np.hanning(tb.size)
    blower = np.zeros_like(tb)
    for k in range(1, 7):
        blower += (0.11 / (0.6 + 0.4 * k)) * np.sin(k * phase)
    blower *= env
    audio[s0:s1, 0] += blower * 1.4
    audio[s0:s1, 1] += blower * 0.45

    _, events = analyze_audio(audio, sr, sensitivity="high", min_event_s=0.3)
    assert any(ev.kind == "variable_rpm_engine" for ev in events)

    steps = [
        FilterStep(
            filter_name="harmonic_track_suppress",
            params={
                "start_s": 1.9,
                "end_s": 5.5,
                "f0_min_hz": 95.0,
                "f0_max_hz": 260.0,
                "max_harmonics": 10,
                "spatial_bias": True,
                "protect_birds": True,
                "context_s": 1.2,
                "context_gap_s": 0.2,
                "fade_ms": 80.0,
                "max_attenuation_db": -30.0,
            },
        )
    ]
    proc_mem = apply_recipe(audio, sr, steps)

    in_wav = save_wav(tmp_path / "blower_in.wav", audio, sr, subtype="FLOAT")
    out_wav = tmp_path / "blower_out.wav"
    apply_recipe_streaming(in_wav, out_wav, steps, subtype="FLOAT")
    proc_stream, _ = load_audio(out_wav)

    for proc in (proc_mem, proc_stream):
        loss = _rms_db(audio[s0:s1, 0]) - _rms_db(proc[s0:s1, 0])
        assert loss >= 6.0
        guard = compute_comparison_guardrails(
            audio[int(0.7 * sr) : int(1.1 * sr)],
            proc[int(0.7 * sr) : int(1.1 * sr)],
            sr,
        )
        if guard["bird_retention_db"] is not None:
            assert guard["bird_retention_db"] >= -1.5


def test_detect_and_suppress_one_sided_spatial_machine() -> None:
    sr = 44100
    n = sr * 6
    rng = np.random.default_rng(505)
    audio = _pinkish_bed(n, sr, seed=505, level=0.006)

    # One-sided L machine band-limited noise (180..750 Hz) during 1.5..4.5 s
    s0, s1 = int(1.5 * sr), int(4.5 * sr)
    sos_m = signal.butter(4, [180.0, 750.0], btype="bandpass", fs=sr, output="sos")
    mach = signal.sosfilt(sos_m, rng.normal(0.0, 0.25, size=s1 - s0))
    audio[s0:s1, 0] += mach * np.hanning(s1 - s0)

    _, events = analyze_audio(audio, sr, sensitivity="high", min_event_s=0.25)
    assert any(ev.kind == "channel_biased_noise" for ev in events)

    proc = apply_recipe(
        audio,
        sr,
        [
            FilterStep(
                filter_name="spectral_match_suppress",
                params={
                    "start_s": 1.4,
                    "end_s": 4.6,
                    "low_hz": 120.0,
                    "high_hz": 1200.0,
                    "spatial_mode": "channel_excess",
                    "protect_birds": True,
                    "context_s": 1.0,
                    "context_gap_s": 0.2,
                    "fade_ms": 80.0,
                },
            )
        ],
    )
    l_before = _rms_db(audio[int(2.0 * sr) : int(4.0 * sr), 0])
    l_after = _rms_db(proc[int(2.0 * sr) : int(4.0 * sr), 0])
    assert (l_before - l_after) >= 6.0


def test_detect_channel_layout_and_spatial_mix_modes(tmp_path: Path) -> None:
    sr = 44100
    n = sr * 3
    rng = np.random.default_rng(606)

    # 1) Raw Mid/Side layout (SSH-6 M/S): high M RMS vs figure-8 S (with bass rolloff)
    mid = rng.normal(0.0, 0.08, size=n)
    sos_fig8 = signal.butter(1, 200.0, btype="highpass", fs=sr, output="sos")
    side = signal.sosfilt(sos_fig8, rng.normal(0.0, 0.015, size=n))
    ms_sig = np.column_stack([mid, side])
    hint_ms = detect_channel_layout(ms_sig, sr)
    assert hint_ms["layout"] == "raw_ms"
    decoded = apply_recipe(
        ms_sig, sr, [FilterStep(filter_name="spatial_mix", params={"mode": "ms_decode"})]
    )
    assert decoded.shape == (n, 2)
    assert abs(_rms_db(decoded[:, 0]) - _rms_db(decoded[:, 1])) < 2.0

    # 2) Directional + surround (SGH-6 + ambience): low correlation, HF diff
    ch_dir = rng.normal(0.0, 0.05, size=n)
    sos_lp = signal.butter(2, 600.0, btype="lowpass", fs=sr, output="sos")
    ch_sur = signal.sosfilt(sos_lp, fractional_shift_channel(ch_dir, 6.0) * 0.8) + rng.normal(
        0.0, 0.01, size=n
    )
    dual = np.column_stack([ch_dir, ch_sur])
    blended, tele = apply_recipe_with_telemetry(
        dual,
        sr,
        [
            FilterStep(
                filter_name="spatial_mix",
                params={"mode": "dual_mic_blend", "align_delay": True, "directional_weight": 0.65},
            )
        ],
    )
    assert blended.shape == (n, 2)
    assert "effective_mode" in tele[0].details

    # 3) Mono -> Stereo upmix (both in-memory and streaming!)
    mono = rng.normal(0.0, 0.04, size=(n, 1))
    up_steps = [
        FilterStep(
            filter_name="spatial_mix",
            params={"mode": "dual_mono_to_stereo", "decorrelate_ms": 6.0},
        )
    ]
    up_mem = apply_recipe(mono, sr, up_steps)
    assert up_mem.shape == (n, 2)

    in_mono = save_wav(tmp_path / "mono_in.wav", mono, sr, subtype="FLOAT")
    out_st = tmp_path / "stereo_out.wav"
    out_info = apply_recipe_streaming(in_mono, out_st, up_steps, subtype="PCM_24")
    assert out_info.channels == 2
    loaded_st, _ = load_audio(out_st)
    assert loaded_st.shape == (n, 2)


def test_declip_from_backup_sub_sample_and_streaming(tmp_path: Path) -> None:
    sr = 44100
    n = sr * 3
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(707)
    clean_base = (
        0.25 * np.sin(2.0 * np.pi * 210.0 * t)
        + 0.15 * np.sin(2.0 * np.pi * 530.0 * t)
        + rng.normal(0.0, 0.02, size=n)
    )
    # Add two transient surges driving main track into digital clipping
    surges = 1.65 * np.exp(-((t - 0.9) / 0.035) ** 2) * np.sin(2.0 * np.pi * 340.0 * t)
    surges += 1.55 * np.exp(-((t - 2.1) / 0.035) ** 2) * np.sin(2.0 * np.pi * 410.0 * t)
    ideal_main = clean_base + surges
    clipped_main = np.clip(ideal_main, -0.99, 0.99)
    main_2d = np.column_stack([clipped_main, clipped_main * 0.95])

    # Safety track (-12 dB) shifted by +1.7 samples
    bk_l = fractional_shift_channel(ideal_main * 0.25, 1.7)
    bk_r = fractional_shift_channel(ideal_main * 0.95 * 0.25, 1.7)
    bk_path = save_wav(tmp_path / "backup.wav", np.column_stack([bk_l, bk_r]), sr, subtype="FLOAT")

    steps = [
        FilterStep(
            filter_name="declip_from_backup",
            params={
                "backup_audio_path": str(bk_path),
                "clip_threshold": 0.98,
                "crossfade_ms": 3.0,
                "fir_taps": 31,
            },
        )
    ]
    rep_mem, tele_mem = apply_recipe_with_telemetry(main_2d, sr, steps)
    assert tele_mem[0].details["repaired_regions"] >= 2
    assert float(np.max(np.abs(rep_mem))) > 1.2

    in_main = save_wav(tmp_path / "clipped_main.wav", main_2d, sr, subtype="FLOAT")
    out_rep = tmp_path / "repaired_stream.wav"
    tele_str: list = []
    apply_recipe_streaming(in_main, out_rep, steps, subtype="FLOAT", telemetry_out=tele_str)
    rep_str, _ = load_audio(out_rep)
    assert tele_str[0].details["repaired_regions"] >= 2
    assert float(np.max(np.abs(rep_str))) > 1.2

    ideal_2d = np.column_stack([ideal_main, ideal_main * 0.95])
    err_before = float(np.sqrt(np.mean((main_2d - ideal_2d) ** 2)))
    err_after = float(np.sqrt(np.mean((rep_str - ideal_2d) ** 2)))
    assert err_after < err_before * 0.35


def test_bed_shape_improves_sharpness_preserves_birds(tmp_path: Path) -> None:
    sr = 44100
    n = sr * 4
    rng = np.random.default_rng(808)
    bed = _pinkish_bed(n, sr, seed=808, level=0.01)
    sos_hiss = signal.butter(4, [2400.0, 6200.0], btype="bandpass", fs=sr, output="sos")
    harsh = signal.sosfilt(sos_hiss, rng.normal(0.0, 0.08, size=(n, 2)), axis=0)
    audio = bed + harsh
    _add_fm_bird_chirp(audio, sr, 1.5, 0.16, 2900.0, 5200.0, amp=0.09)
    _add_fm_bird_chirp(audio, sr, 2.8, 0.16, 3200.0, 5500.0, amp=0.09)

    m_before, _ = analyze_audio(audio, sr)
    steps = [
        FilterStep(
            filter_name="bed_shape",
            params={
                "sub_rumble_cut_db": -6.0,
                "harshness_dip_db": -7.0,
                "warmth_tilt_db": -3.0,
                "floor_preserve": 0.85,
            },
        )
    ]
    proc_mem = apply_recipe(audio, sr, steps)
    in_wav = save_wav(tmp_path / "harsh_in.wav", audio, sr, subtype="FLOAT")
    out_wav = tmp_path / "harsh_out.wav"
    apply_recipe_streaming(in_wav, out_wav, steps, subtype="FLOAT")
    proc_str, _ = load_audio(out_wav)

    for proc in (proc_mem, proc_str):
        m_after, _ = analyze_audio(proc, sr)
        assert m_after.sharpness_acum is not None and m_before.sharpness_acum is not None
        assert m_after.sharpness_acum < m_before.sharpness_acum
        guard = compute_comparison_guardrails(audio, proc, sr)
        if guard["bird_retention_db"] is not None:
            assert guard["bird_retention_db"] >= -1.5


def test_ambience_patch_v3_donor_distance_and_crest_screening(tmp_path: Path) -> None:
    sr = 44100
    n = sr * 12
    audio = _pinkish_bed(n, sr, seed=909, level=0.01)
    # Plant loud knocks at 3.2 s and 8.2 s inside the normal vicinity of target 5.0..6.2 s
    audio[int(3.2 * sr) : int(3.2 * sr) + 12, :] += 0.55
    audio[int(8.2 * sr) : int(8.2 * sr) + 12, :] += 0.55
    # Target distraction 5.0..6.2 s
    audio[int(5.0 * sr) : int(6.2 * sr), :] += 0.08 * np.sin(
        2.0 * np.pi * 320.0 * (np.arange(int(1.2 * sr)) / sr)
    )[:, np.newaxis]

    steps = [
        FilterStep(
            filter_name="ambience_patch",
            params={
                "start_s": 5.0,
                "end_s": 6.2,
                "min_donor_distance_s": 3.0,
                "max_donor_crest": 4.5,
                "crossfade_ms": 40.0,
            },
        )
    ]
    _, tele_mem = apply_recipe_with_telemetry(audio, sr, steps)
    span_m = tele_mem[0].details["patched_spans"][0]
    assert isinstance(span_m["loop_count"], int)
    assert isinstance(span_m["donor_chunks"], int)
    # Must choose a donor outside [2.0, 9.2] (5.0-3.0 .. 6.2+3.0)
    assert span_m["donor_end_s"] <= 2.2 or span_m["donor_start_s"] >= 9.0

    in_wav = save_wav(tmp_path / "patch_in.wav", audio, sr, subtype="FLOAT")
    out_wav = tmp_path / "patch_out.wav"
    tele_str: list = []
    apply_recipe_streaming(in_wav, out_wav, steps, subtype="FLOAT", telemetry_out=tele_str)
    span_s = tele_str[0].details["patched_spans"][0]
    assert isinstance(span_s["loop_count"], int)
    assert isinstance(span_s["donor_chunks"], int)
    assert span_s["donor_end_s"] <= 2.2 or span_s["donor_start_s"] >= 9.0


def test_verify_seams_slew_excess_gate(tmp_path: Path) -> None:
    settings = Settings(workspace_dir=tmp_path / "ws", allowed_roots=[tmp_path])
    mgr = SessionManager(settings)
    sr = 44100
    rng = np.random.default_rng(1010)
    # Loud continuous -16 LUFS broadband noise across split (sample step above -30 dBFS normally)
    full = rng.normal(0.0, 0.16, size=(sr * 4, 2))
    p_a = save_wav(tmp_path / "seam_a.wav", full[: sr * 2], sr, subtype="PCM_24")
    p_b = save_wav(tmp_path / "seam_b.wav", full[sr * 2 :], sr, subtype="PCM_24")
    s_a = mgr.create_session(p_a, name="SeamA")
    s_b = mgr.create_session(p_b, name="SeamB")

    report_ok = mgr.verify_seams([(s_a.session_id, "take_000"), (s_b.session_id, "take_000")])
    assert "thresholds" in report_ok
    assert report_ok["seams"][0]["sample_step_excess_db"] is not None
    assert report_ok["all_passed"] is True

    # Hard artificial discontinuity (+0.65 step) should fail slew-excess gate
    bad_b = full[sr * 2 :].copy()
    bad_b[0, :] = -full[sr * 2 - 1, :] + 0.65
    p_bad = save_wav(tmp_path / "seam_bad.wav", bad_b, sr, subtype="FLOAT")
    s_bad = mgr.create_session(p_bad, name="SeamBad")
    report_bad = mgr.verify_seams([(s_a.session_id, "take_000"), (s_bad.session_id, "take_000")])
    assert report_bad["all_passed"] is False


def test_macro_event_summary_and_clean_sections_local_minutes() -> None:
    events = [
        DistractionEvent(
            id=f"ev_{i:03d}",
            kind="impulsive_click",
            start_s=float(i) * 2.0,
            end_s=float(i) * 2.0 + 0.02,
            severity="low",
        )
        for i in range(32)
    ]
    events.extend(
        [
            DistractionEvent(
                id="ev_sp0",
                kind="slow_pass",
                start_s=20.0,
                end_s=26.0,
                severity="high",
            ),
            DistractionEvent(
                id="ev_sp1",
                kind="synthetic_beep",
                start_s=50.0,
                end_s=50.5,
                severity="medium",
            ),
        ]
    )

    macro, counts, total_mac, trunc_mac = summarize_events(events)
    assert counts["impulsive_click"] == 32
    assert total_mac == 34
    assert trunc_mac is True
    assert len(macro) == 2
    assert all(ev.kind != "impulsive_click" for ev in macro)

    explicit, _, total_all, trunc_all = summarize_events(events, include_clicks=True, max_events=50)
    assert total_all == 34
    assert trunc_all is False
    assert len(explicit) == 34

    sec = compute_clean_sections(
        [
            {
                "label": "file1.wav",
                "session_id": "s1",
                "take_id": "take_001",
                "duration_s": 1825.0,
                "edit_spans": [(1500.0, 1510.0)],
                "distraction_spans": [],
            },
            {
                "label": "file2.wav",
                "session_id": "s2",
                "take_id": "take_001",
                "duration_s": 3700.0,
                "edit_spans": [(45.0, 50.0)],
                "distraction_spans": [],
            },
        ],
        round_to_file_local_minutes=True,
    )
    assert sec["asmr_targets"]
    top = sec["largest_untouched_asmr_target"]
    assert top is not None
    assert top["start_file"] == "file2.wav"
    assert top["start_local_hms"] == "00:01:00"


def test_do_no_harm_assessment_clean_vs_dirty() -> None:
    sr = 44100
    n = sr * 4
    clean = _pinkish_bed(n, sr, seed=1212, level=0.008)
    _add_fm_bird_chirp(clean, sr, 1.5, 0.15, 2900.0, 5000.0, amp=0.015)
    m_clean, ev_clean = analyze_audio(clean, sr, sensitivity="medium")
    a_clean = assess_do_no_harm(m_clean, ev_clean)
    assert a_clean["recommend_cleanup"] is False

    dirty = clean.copy()
    t = np.arange(n, dtype=np.float64) / sr
    dirty += 0.25 * np.sin(2.0 * np.pi * 30.0 * t)[:, np.newaxis]
    s0, s1 = int(2.0 * sr), int(2.2 * sr)
    dirty[s0:s1, :] += 0.18 * np.sin(2.0 * np.pi * 4096.0 * t[s0:s1])[:, np.newaxis]
    m_dirty, ev_dirty = analyze_audio(dirty, sr, sensitivity="high")
    a_dirty = assess_do_no_harm(m_dirty, ev_dirty)
    assert a_dirty["recommend_cleanup"] is True
    assert "Cleanup warranted" in a_dirty["do_no_harm_reason"]


def test_v052_bird_retention_gain_normalized_and_local_minutes_all_tiers() -> None:
    sr = 44100
    n = sr * 4
    ref = _pinkish_bed(n, sr, seed=1401, level=0.005)
    _add_fm_bird_chirp(ref, sr, 1.2, 0.2, 3200.0, 5200.0, amp=0.025)
    # +30 dB uniform gain alone must NOT inflate bird_retention_db
    boosted = np.clip(ref * 31.6227766, -0.99, 0.99)
    guard_same = compute_comparison_guardrails(ref, boosted, sr)
    assert guard_same["bird_retention_db"] is not None
    assert abs(guard_same["bird_retention_db"]) < 0.3
    assert guard_same["gain_offset_db"] is not None and guard_same["gain_offset_db"] > 28.0
    assert not any("dropped" in w.lower() for w in guard_same["guardrail_warnings"])

    # Dropping bird energy while boosting overall bed gain must STILL trip the <-1.5 dB guardrail
    bed_only = _pinkish_bed(n, sr, seed=1401, level=0.005) * 31.6227766
    guard_lost = compute_comparison_guardrails(ref, np.clip(bed_only, -0.99, 0.99), sr)
    assert guard_lost["bird_retention_db"] is not None and guard_lost["bird_retention_db"] < -1.5
    assert any("dropped" in w.lower() for w in guard_lost["guardrail_warnings"])

    # File-local minute rounding applies strictly across minimal_edits & moderate_edits too
    rep = compute_clean_sections(
        [
            {
                "label": "SPLIT_A.WAV",
                "session_id": "s1",
                "take_id": "take_001",
                "duration_s": 3700.123,
                "edit_spans": [(10.0, 12.0), (1500.0, 1502.0)],
                "distraction_spans": [(30.0, 32.0)],
            },
            {
                "label": "SPLIT_B.WAV",
                "session_id": "s2",
                "take_id": "take_001",
                "duration_s": 4000.0,
                "edit_spans": [(70.0, 72.0)],
                "distraction_spans": [],
            },
        ],
        round_to_file_local_minutes=True,
    )
    t_1hr = next(t for t in rep["asmr_targets"] if t["duration_min"] == 60)
    assert t_1hr["start_local_s"] % 60.0 < 1e-3
    assert "eligible" in t_1hr and isinstance(t_1hr["violations"], list)


def test_v053_beep_gain_invariance_narrowband_and_10min_target() -> None:
    sr = 48000
    n = sr * 5
    # Quiet field recording (-45 dBFS bed) with 90 ms 2982 Hz watch beep + FM birds
    quiet = _pinkish_bed(n, sr, seed=503, level=0.0025)
    _add_fm_bird_chirp(quiet, sr, 0.8, 0.16, 2600.0, 4800.0, amp=0.012)
    t = np.arange(n, dtype=np.float64) / sr
    b0, b1 = int(2.20 * sr), int(2.29 * sr)
    env = np.hanning(b1 - b0)
    tone = 0.045 * np.clip(env * 2.5, 0.0, 1.0) * np.sin(2.0 * np.pi * 2982.0 * t[b0:b1])
    quiet[b0:b1, 0] += tone
    quiet[b0:b1, 1] += tone

    beeps_q = detect_synthetic_beeps(np.mean(quiet, axis=1), sr, sensitivity="medium")
    beeps_boost = detect_synthetic_beeps(np.mean(quiet * 10.0, axis=1), sr, sensitivity="medium")
    assert any(b.freq_band_hz[1] >= 2940.0 and b.freq_band_hz[0] <= 3020.0 for b in beeps_q)
    assert len(beeps_q) == len(beeps_boost)

    # At n_fft=4096 / 48 kHz, 90 ms 2982 Hz piezo tone must NOT be protected by bird mask
    f, tm, zxx = signal.stft(
        np.mean(quiet, axis=1), fs=sr, nperseg=4096, noverlap=3072, window="hann"
    )
    pmask = compute_bird_protect_mask(f, tm, np.abs(zxx))
    bin_2982 = int(np.argmin(np.abs(f - 2982.0)))
    in_beep = (tm >= 2.21) & (tm <= 2.28)
    assert float(np.mean(pmask[bin_2982, in_beep])) < 0.15

    proc = apply_recipe(
        quiet,
        sr,
        [
            FilterStep(
                filter_name="spectral_match_suppress",
                params={
                    "start_s": 2.0,
                    "end_s": 2.5,
                    "low_hz": 2700.0,
                    "high_hz": 3300.0,
                    "narrowband_mode": True,
                    "protect_birds": True,
                    "n_fft": 4096,
                    "max_attenuation_db": -36.0,
                },
            )
        ],
    )
    sos_bp = signal.butter(4, [2920.0, 3040.0], btype="bandpass", fs=sr, output="sos")
    before_bp = _rms_db(signal.sosfiltfilt(sos_bp, quiet[b0:b1, 0]))
    after_bp = _rms_db(signal.sosfiltfilt(sos_bp, proc[b0:b1, 0]))
    assert (before_bp - after_bp) >= 15.0

    # 35-min recording must include 10 min target in asmr_targets
    rep35 = compute_clean_sections(
        [{"label": "Z1.WAV", "session_id": "s1", "take_id": "t1", "duration_s": 2123.367}]
    )
    assert any(item["duration_min"] == 10 for item in rep35["asmr_targets"])

    # list_filter_specs exposes 'bark_distance' choice
    ds_spec = next(sp for sp in list_filter_specs() if sp["name"] == "distance_soften")
    assert "bark_distance" in ds_spec["params"]["mode"]["choices"]


