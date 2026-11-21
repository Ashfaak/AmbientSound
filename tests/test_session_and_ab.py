"""Tests for non-destructive session DAG, A/B loudness matching, and feedback logging."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import ambient_sound.analysis as analysis_mod
import ambient_sound.session as session_mod
from ambient_sound.config import Settings
from ambient_sound.io_audio import compile_audio_span, load_audio, save_wav
from ambient_sound.models import DistractionEvent
from ambient_sound.session import SessionManager
from ambient_sound.visuals import (
    render_diff_spectrogram_png,
    render_file_spectrogram_png,
    render_spectrogram_png,
)


def test_session_take_dag_and_blind_ab_lifecycle(tmp_path: Path) -> None:
    settings = Settings(workspace_dir=tmp_path / ".ambientsound", allowed_roots=[tmp_path])
    manager = SessionManager(settings)

    sr = 44100
    n = sr * 2
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(5)
    audio = rng.normal(0.0, 0.03, size=(n, 2)) + (0.04 * np.sin(2.0 * np.pi * 60.0 * t))[:, None]
    src_wav = save_wav(tmp_path / "field.wav", audio, sr)

    summary = manager.create_session(src_wav, name="Forest Rain")
    assert len(summary.takes) == 1
    orig_take = summary.takes[0]
    assert orig_take.take_id == "take_000"
    assert Path(orig_take.audio_path).exists()
    assert Path(orig_take.spectrogram_path).exists()

    take_a = manager.process_take(
        summary.session_id,
        "take_000",
        [{"filter_name": "dehum_notch", "params": {"base_freq_hz": 60.0}}],
        label="De-hum 60Hz",
    )
    take_b = manager.process_take(
        summary.session_id,
        "take_000",
        [
            {"filter_name": "highpass_rumble", "params": {"cutoff_hz": 70.0}},
            {"filter_name": "dehum_notch", "params": {"base_freq_hz": 60.0}},
        ],
        label="HPF + De-hum",
    )
    assert take_a.take_id == "take_001"
    assert take_b.take_id == "take_002"
    assert Path(take_a.residual_path).exists()
    assert Path(take_a.diff_spectrogram_path).exists()

    ab = manager.create_ab_test(
        summary.session_id,
        take_a.take_id,
        take_b.take_id,
        focus_start_s=0.2,
        focus_end_s=1.5,
        blind=True,
        notes="Check low-end warmth vs hum cleanup.",
    )
    # Loudness matching must be attenuation-only (never positive boost)
    assert ab.gain_db_a <= 0.0
    assert ab.gain_db_b <= 0.0
    assert ab.gain_db_original <= 0.0

    judged = manager.submit_ab_verdict(
        summary.session_id,
        ab.ab_id,
        winner="b",
        artifact_tags=["Natural ambience"],
        comment="B sounds cleaner with no warmth loss.",
    )
    assert judged.verdict is not None
    assert judged.verdict.winner == "b"

    log_file = settings.workspace_dir / "sessions" / summary.session_id / "session_feedback_log.md"
    assert log_file.exists()
    assert "Natural ambience" in log_file.read_text(encoding="utf-8")

    exported = manager.export_take(
        summary.session_id, take_b.take_id, tmp_path / "out.flac", "flac"
    )
    assert exported.exists()

    with pytest.raises(PermissionError):
        manager.export_take(summary.session_id, take_b.take_id, Path("C:/Windows/Temp/outside.wav"))


def test_streaming_session_pipeline_and_windowed_visuals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(session_mod, "STREAMING_THRESHOLD_S", 1.5)
    settings = Settings(workspace_dir=tmp_path / ".ambientsound", allowed_roots=[tmp_path])
    manager = SessionManager(settings)

    sr = 44100
    n = sr * 3
    rng = np.random.default_rng(17)
    audio = rng.normal(0.0, 0.03, size=(n, 2))
    src_wav = save_wav(tmp_path / "long_field.wav", audio, sr, subtype="PCM_24")

    summary = manager.create_session(src_wav, name="Long Hive")
    t1 = manager.process_take(
        summary.session_id,
        "take_000",
        [{"filter_name": "highpass_rumble", "params": {"cutoff_hz": 80.0}}],
        label="Streamed HPF",
    )
    assert t1.residual_path is None
    assert manager.can_stream_residual(summary.session_id, t1.take_id) is True

    res_slice, res_sr = manager.load_take_residual_slice(
        summary.session_id, t1.take_id, start_s=0.5, end_s=1.5
    )
    assert res_sr == sr
    assert abs(res_slice.shape[0] - sr) <= 2

    vis = manager.render_take_visuals(
        summary.session_id,
        t1.take_id,
        compare_to_take_id="take_000",
        start_s=0.5,
        end_s=2.0,
    )
    assert Path(vis["spectrogram_path"]).exists()
    assert Path(vis["diff_spectrogram_path"]).exists()

    # Exercise > 60s overview spectrogram renderer quickly with a lightweight 8kHz file
    sr_lo = 8000
    long_mono = np.random.default_rng(2).normal(0.0, 0.02, size=(sr_lo * 65, 1))
    long_lo_wav = save_wav(tmp_path / "overview_65s.wav", long_mono, sr_lo, subtype="PCM_16")
    ov_png = render_file_spectrogram_png(long_lo_wav, tmp_path / "overview.png")
    assert ov_png.exists()


def test_default_workspace_uses_shared_home_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AMBIENTSOUND_WORKSPACE", raising=False)
    monkeypatch.delenv("AMBIENTSOUND_ALLOWED_ROOTS", raising=False)
    cfg = Settings.from_env()
    assert cfg.workspace_dir == (Path.home() / ".ambientsound").resolve()
    assert cfg.workspace_dir in cfg.allowed_roots


def test_timeline_lca_candidate_matching_and_residual_gain(tmp_path: Path) -> None:
    settings = Settings(workspace_dir=tmp_path / ".ambientsound", allowed_roots=[tmp_path])
    manager = SessionManager(settings)

    sr = 44100
    n = sr * 4
    rng = np.random.default_rng(29)
    # Quiet spiky ~ -44 LUFS recording
    quiet = rng.normal(0.0, 0.005, size=(n, 2))
    for k in range(1, 8):
        idx = int(k * 0.45 * sr)
        quiet[idx : idx + 6, :] += 0.06
    src_wav = save_wav(tmp_path / "quiet_spiky.wav", quiet, sr, subtype="PCM_24")

    summary = manager.create_session(src_wav, name="Hive Preview")
    t1 = manager.process_take(
        summary.session_id,
        "take_000",
        [{"filter_name": "trim_range", "params": {"start_s": 0.5, "end_s": 3.5}}],
        label="Trimmed Preview",
    )
    assert abs(t1.source_offset_s - 0.5) < 1e-3

    t2 = manager.process_take(
        summary.session_id,
        t1.take_id,
        [
            {"filter_name": "highpass_rumble", "params": {"cutoff_hz": 50.0}},
            {
                "filter_name": "loudness_normalize",
                "params": {"target_lufs": -14.0, "limit_peaks": True},
            },
        ],
        label="Master -14",
    )
    t3 = manager.process_take(
        summary.session_id,
        t1.take_id,
        [
            {
                "filter_name": "loudness_normalize",
                "params": {"target_lufs": -16.0, "limit_peaks": True},
            }
        ],
        label="Master -16",
    )

    ab_cand = manager.create_ab_test(summary.session_id, t2.take_id, t3.take_id)
    assert ab_cand.original_take_id == t1.take_id
    assert ab_cand.match_original is False
    assert -4.0 <= ab_cand.gain_db_a <= 0.0
    assert ab_cand.gain_db_b <= 0.0
    assert ab_cand.gain_db_original <= 0.0

    ab_orig = manager.create_ab_test(
        summary.session_id, t2.take_id, t3.take_id, match_original=True
    )
    assert ab_orig.gain_db_a < -15.0

    # Pure gain take residual should be gain-matched near zero dynamically
    t4 = manager.process_take(
        summary.session_id,
        t1.take_id,
        [{"filter_name": "gain", "params": {"gain_db": 18.0}}],
        label="Gain +18 dB",
    )
    res_slice, _ = manager.load_take_residual_slice(
        summary.session_id, t4.take_id, start_s=0.2, end_s=1.2
    )
    assert float(np.sqrt(np.mean(res_slice * res_slice))) < 1e-3


def test_events_json_delete_take_session_and_visuals(tmp_path: Path) -> None:
    settings = Settings(workspace_dir=tmp_path / ".ambientsound", allowed_roots=[tmp_path])
    manager = SessionManager(settings)

    sr = 44100
    rng = np.random.default_rng(71)
    audio = rng.normal(0.0, 0.03, size=(sr * 3, 2))
    src_wav = save_wav(tmp_path / "cleanup.wav", audio, sr)
    summary = manager.create_session(src_wav, name="Disk Cleanup")

    t0 = summary.takes[0]
    assert t0.events_path is not None and Path(t0.events_path).exists()
    assert len(manager.load_take_events(summary.session_id, "take_000")) >= len(t0.events)

    t1 = manager.process_take(
        summary.session_id,
        "take_000",
        [{"filter_name": "highpass_rumble", "params": {"cutoff_hz": 60.0}}],
        label="Parent T1",
    )
    t2 = manager.process_take(
        summary.session_id,
        t1.take_id,
        [{"filter_name": "ambience_patch", "params": {"start_s": 0.5, "end_s": 0.9}}],
        label="Child T2",
    )
    manager.create_ab_test(summary.session_id, t1.take_id, t2.take_id)

    with pytest.raises(ValueError):
        manager.delete_take(summary.session_id, "take_000")
    with pytest.raises(ValueError):
        manager.delete_take(summary.session_id, t1.take_id, cascade=False)

    t_sib = manager.process_take(
        summary.session_id,
        "take_000",
        [{"filter_name": "gain", "params": {"gain_db": -1.0}}],
        label="Sibling T3",
    )
    assert t_sib.take_id == "take_003"

    # Deleting non-last intermediate take_001 with cascade keeps take_003 intact; next is take_004
    res_del = manager.delete_take(summary.session_id, t1.take_id, cascade=True)
    assert set(res_del["deleted_take_ids"]) == {t1.take_id, t2.take_id}
    assert res_del["deleted_ab_ids"] == ["ab_001"]
    assert res_del["remaining_takes"] == 2

    t_next = manager.process_take(
        summary.session_id,
        "take_000",
        [{"filter_name": "gain", "params": {"gain_db": -2.0}}],
        label="After Delete",
    )
    assert t_next.take_id == "take_004"

    # Dense click skipping & patched span annotation rendering
    dense_clicks = [
        DistractionEvent(
            id=f"clk_{i}",
            start_s=0.05 * i,
            end_s=0.05 * i + 0.005,
            kind="impulsive_click",
            severity="medium",
        )
        for i in range(50)
    ]
    spec_out = render_spectrogram_png(
        audio, sr, tmp_path / "dense_spec.png", title="Dense Clicks", events=dense_clicks
    )
    diff_out = render_diff_spectrogram_png(
        audio,
        audio * 4.0,
        sr,
        tmp_path / "patched_diff.png",
        patched_spans=[(0.5, 0.9)],
    )
    assert spec_out.exists()
    assert diff_out.exists()

    del_sess = manager.delete_session(summary.session_id)
    assert del_sess == {"session_id": summary.session_id, "deleted": True}
    assert not (settings.workspace_dir / "sessions" / summary.session_id).exists()


def test_cross_branch_trimmed_ab_resolves_shared_ancestor_and_offset(tmp_path: Path) -> None:
    settings = Settings(workspace_dir=tmp_path / ".ambientsound", allowed_roots=[tmp_path])
    manager = SessionManager(settings)

    sr = 44100
    rng = np.random.default_rng(103)
    audio = rng.normal(0.0, 0.02, size=(sr * 4, 2))
    src_wav = save_wav(tmp_path / "branch.wav", audio, sr)
    summary = manager.create_session(src_wav, name="Cross Branch")

    b1 = manager.process_take(
        summary.session_id,
        "take_000",
        [
            {"filter_name": "trim_range", "params": {"start_s": 0.5, "end_s": 3.5}},
            {"filter_name": "highpass_rumble", "params": {"cutoff_hz": 50.0}},
        ],
        label="Branch A",
    )
    b2 = manager.process_take(
        summary.session_id,
        "take_000",
        [
            {"filter_name": "trim_range", "params": {"start_s": 0.5, "end_s": 3.5}},
            {"filter_name": "gain", "params": {"gain_db": 4.0}},
        ],
        label="Branch B",
    )

    ab = manager.create_ab_test(summary.session_id, b1.take_id, b2.take_id)
    assert ab.original_take_id == "take_000"
    assert abs(ab.original_offset_s - 0.5) < 1e-3


def test_v040_copy_source_false_and_sequence_pipeline(tmp_path: Path) -> None:
    settings = Settings(workspace_dir=tmp_path / ".ambientsound", allowed_roots=[tmp_path])
    manager = SessionManager(settings)

    sr = 16000
    n = sr * 3
    t = np.arange(n * 2, dtype=np.float64) / sr
    rng = np.random.default_rng(404)
    bed = rng.normal(0.0, 0.012, size=(n * 2, 2))
    buzz = (0.025 * np.sin(2.0 * np.pi * 240.0 * t))[:, None]
    full = bed + buzz

    raw_dir = tmp_path / "raw_files"
    p1 = save_wav(raw_dir / "ZOOM0004.WAV", full[:n], sr, subtype="PCM_24")
    p2 = save_wav(raw_dir / "ZOOM0005.WAV", full[n:], sr, subtype="PCM_24")

    s1 = manager.create_session(p1, name="Split 1", copy_source=False)
    s2 = manager.create_session(p2, name="Split 2", copy_source=False)
    assert Path(s1.takes[0].audio_path) == p1
    assert not (
        settings.workspace_dir / "sessions" / s1.session_id / "takes" / "take_000" / "audio.wav"
    ).exists()

    vis = manager.render_take_visuals(s1.session_id, "take_000", start_s=0.2, end_s=1.2)
    assert Path(vis["spectrogram_path"]).parent != raw_dir

    seq = [(s1.session_id, "take_000"), (s2.session_id, "take_000")]
    cal = manager.calibrate_sequence(
        seq,
        shared_steps_before_gain=[
            {"filter_name": "highpass_rumble", "params": {"cutoff_hz": 50.0}}
        ],
        target_lufs=-16.0,
        ceiling_dbtp=-1.0,
        sample_windows_per_file=2,
        window_s=1.0,
        shared_steps_after_gain=[
            {"filter_name": "true_peak_limiter", "params": {"ceiling_dbtp": -1.0}}
        ],
    )
    gain_db = cal["recommended_gain_db"]
    assert abs(cal["projected_sequence_lufs"] - (-16.0)) < 1.2

    out_dir = tmp_path / "masters"
    proc = manager.process_sequence(
        items=[
            {"session_id": s1.session_id, "export_path": str(out_dir / "ZOOM0004_master.wav")},
            {"session_id": s2.session_id, "export_path": str(out_dir / "ZOOM0005_master.wav")},
        ],
        shared_prefix_steps=[{"filter_name": "highpass_rumble", "params": {"cutoff_hz": 50.0}}],
        shared_suffix_steps=[
            {"filter_name": "gain", "params": {"gain_db": gain_db}},
            {"filter_name": "true_peak_limiter", "params": {"ceiling_dbtp": -1.0}},
        ],
        outer_fade_in_s=0.2,
        outer_fade_out_s=0.2,
        seam_context_s=0.8,
        discard_take_audio_after_export=True,
    )
    assert proc["count"] == 2
    assert proc["seams"]["all_passed"] is True
    assert Path(proc["seams"]["seams"][0]["spectrogram_path"]).exists()
    assert (out_dir / "ZOOM0004_master.wav").exists()
    assert (out_dir / "ZOOM0005_master.wav").exists()
    # Internal take_001/audio.wav discarded while session take still points to exported WAV
    assert not (
        settings.workspace_dir / "sessions" / s1.session_id / "takes" / "take_001" / "audio.wav"
    ).exists()
    sl, _ = manager.load_take_audio_slice(s1.session_id, "take_001", start_s=0.1, end_s=0.6)
    assert sl.shape[0] > 0

    # Deleting session with copy_source=False must NEVER delete the external raw WAV
    manager.delete_session(s1.session_id)
    assert p1.exists()


def test_v051_compile_clean_sections_and_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analysis_mod, "YOUTUBE_ASMR_DURATIONS_MIN", (1,))
    settings = Settings(workspace_dir=tmp_path / ".ambientsound", allowed_roots=[tmp_path])
    manager = SessionManager(settings)

    sr = 8000
    n_each = sr * 35
    total_n = n_each * 2
    t = np.arange(total_n, dtype=np.float64) / sr
    rng = np.random.default_rng(501)
    # Continuous low-level drone + noise across split boundary so 60 s span stitches cleanly
    full = rng.normal(0.0, 0.015, size=(total_n, 2)) + (
        0.04 * np.sin(2.0 * np.pi * 220.0 * t)
    )[:, None]

    raw_dir = tmp_path / "raw_split"
    p1 = save_wav(raw_dir / "ZOOM0010.WAV", full[:n_each], sr, subtype="PCM_24")
    p2 = save_wav(raw_dir / "ZOOM0011.WAV", full[n_each:], sr, subtype="PCM_24")
    s1 = manager.create_session(p1, name="Split 10", copy_source=False)
    s2 = manager.create_session(p2, name="Split 11", copy_source=False)

    out_dir = tmp_path / "clean_out"
    res = manager.compile_clean_sections(
        sequence=[(s1.session_id, "take_000"), (s2.session_id, "take_000")],
        output_dir=out_dir,
        targets=["1 min"],
        include_full_masters=True,
        include_combined_full=True,
        filename_prefix="beehive",
        fade_in_s=0.1,
        fade_out_s=0.1,
    )
    assert res["compiled_count"] == 1
    assert res["masters_count"] == 2
    assert res["total_bytes"] > 0
    assert res["tool_version"]
    assert res["parameters"]["filename_prefix"] == "beehive"
    sec = res["compiled_sections"][0]
    assert sec["spans_multiple_files"] is True
    assert sec["requested_target"] == "1 min"
    assert sec["fade_in_s"] == 0.1 and sec["fade_out_s"] == 0.1
    assert sec["peak_measured_on"] == "output"
    assert len(sec["seams"]) == 1
    assert sec["seams"][0]["from_file"] == "ZOOM0010.WAV"
    assert sec["seams"][0]["to_file"] == "ZOOM0011.WAV"
    assert len(sec["source_slices"]) == 2
    assert sec["source_slices"][0]["source_file"] == "ZOOM0010.WAV"
    assert sec["source_slices"][1]["source_file"] == "ZOOM0011.WAV"
    assert len(res["full_masters"]) == 2
    assert Path(res["full_masters"][0]["output_path"]).name == "beehive_ZOOM0010_master.wav"
    assert res["full_masters"][0]["prefix_applied"] is True
    assert res["full_masters"][0]["metrics_source"] == "cached_take_metrics"
    assert res["combined_full"] is not None
    assert Path(res["manifest_path"]).exists()

    compiled_arr, compiled_sr = load_audio(sec["output_path"])
    assert compiled_sr == sr
    assert compiled_arr.shape[0] == int(round(sec["duration_s"] * sr))
    # Outer edges obey fades while interior preserves continuous signal across seam
    assert float(np.max(np.abs(compiled_arr[:4]))) < 1e-4
    assert float(np.max(np.abs(compiled_arr[-4:]))) < 1e-4

    # Distinct-regions check: two 1-minute spans across a 130 s recording must not overlap
    monkeypatch.setattr(analysis_mod, "YOUTUBE_ASMR_DURATIONS_MIN", (1, 2))
    out_dist = tmp_path / "distinct_out"
    p_d1 = save_wav(raw_dir / "DIST01.WAV", full, sr, subtype="PCM_24")
    p_d2 = save_wav(raw_dir / "DIST02.WAV", full, sr, subtype="PCM_24")
    sd1 = manager.create_session(p_d1, name="D1", copy_source=False)
    sd2 = manager.create_session(p_d2, name="D2", copy_source=False)
    res_dist = manager.compile_clean_sections(
        sequence=[(sd1.session_id, "take_000"), (sd2.session_id, "take_000")],
        output_dir=out_dist,
        targets=["1 min", "2 min"],
        distinct_regions=True,
        fade_in_s=0.05,
        fade_out_s=0.05,
    )
    assert res_dist["compiled_count"] == 1
    assert len(res_dist["skipped_targets"]) == 1

    # Direct multi-slice span check verifies exact interior sample continuity across stitch point
    direct_out = tmp_path / "direct_stitch.wav"
    direct_info = compile_audio_span(
        segments=[(p1, 35.0), (p2, 35.0)],
        start_s=30.0,
        end_s=40.0,
        dst_path=direct_out,
        audio_format="wav",
        fade_in_s=0.05,
        fade_out_s=0.05,
    )
    stitched, _ = load_audio(direct_out)
    assert direct_info["duration_s"] == 10.0
    expected_mid = full[sr * 32 : sr * 38]
    assert float(np.max(np.abs(stitched[sr * 2 : sr * 8] - expected_mid))) < 1e-5

    # Safety guards: forbid compiling into workspace/sessions or outside allowed_roots
    with pytest.raises(ValueError):
        manager.compile_clean_sections(
            sequence=[(s1.session_id, "take_000")],
            output_dir=settings.workspace_dir / "sessions" / "forbidden",
            targets=["largest"],
        )
    with pytest.raises(PermissionError):
        manager.compile_clean_sections(
            sequence=[(s1.session_id, "take_000")],
            output_dir="C:/Windows/Temp/outside_clean_out",
            targets=["largest"],
        )

