"""End-to-end tests for the AmbientSound MCP tool surface and path sandboxing."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError

import ambient_sound.analysis as analysis_mod
from ambient_sound.config import Settings
from ambient_sound.io_audio import save_wav
from ambient_sound.mcp_server import create_mcp_server


def _call_tool_json(server: Any, name: str, arguments: dict[str, Any]) -> Any:
    result = asyncio.run(server.call_tool(name, arguments))
    assert not result.is_error
    if result.structured_content is not None:
        return result.structured_content
    texts = [block.text for block in result.content if hasattr(block, "text")]
    assert texts, f"Tool '{name}' returned empty content."
    if len(texts) == 1:
        return json.loads(texts[0])
    return [json.loads(t) for t in texts]


def test_mcp_tool_workflow_and_sandboxing(tmp_path: Path) -> None:
    settings = Settings(workspace_dir=tmp_path / ".ambientsound", allowed_roots=[tmp_path])
    mcp = create_mcp_server(settings)

    tools = asyncio.run(mcp.list_tools())
    tool_names = {t.name for t in tools}
    assert {
        "health_check",
        "list_filters",
        "create_session",
        "list_sessions",
        "get_session_summary",
        "analyze_take",
        "analyze_region",
        "render_spectrogram",
        "process_take",
        "create_ab_test",
        "list_ab_tests",
        "get_ab_feedback",
        "delete_take",
        "delete_session",
        "export_take",
        "find_clean_sections",
        "verify_seams",
        "calibrate_sequence",
        "process_sequence",
        "start_sequence_job",
        "compile_clean_sections",
        "start_compile_clean_sections_job",
        "get_job_status",
        "wait_for_job",
        "cancel_job",
        "list_jobs",
    }.issubset(tool_names)
    assert len(tool_names) == 26

    health = _call_tool_json(mcp, "health_check", {})
    assert health["status"] == "ok"
    assert health["filter_count"] == 21
    assert health["tool_count"] == 26

    filters_res = _call_tool_json(mcp, "list_filters", {})
    assert filters_res["filter_count"] == 21
    assert "step_schema" in filters_res
    assert "spans_precedence" in filters_res
    filters = filters_res["filters"]
    assert isinstance(filters, list)
    assert any(f["name"] == "ambience_patch" for f in filters)
    assert any(f["name"] == "spectral_match_suppress" for f in filters)
    assert any(f["name"] == "bed_shape" for f in filters)

    sr = 44100
    rng = np.random.default_rng(21)
    audio = rng.normal(0.0, 0.03, size=(sr * 2, 2))
    src = save_wav(tmp_path / "ambience.wav", audio, sr)

    session = _call_tool_json(
        mcp,
        "create_session",
        {
            "audio_path": str(src),
            "name": "Creek",
            "copy_source": False,
            "compact_metrics": True,
        },
    )
    sid = session["session_id"]
    assert session["takes"][0]["take_id"] == "take_000"
    assert session["takes"][0]["metrics"]["short_term_lufs_timeline"] == []
    assert Path(session["takes"][0]["audio_path"]) == src
    assert "clean_sections" in session["takes"][0]

    sessions_res = _call_tool_json(mcp, "list_sessions", {})
    assert any(s["session_id"] == sid for s in sessions_res["sessions"])

    analyzed = _call_tool_json(mcp, "analyze_take", {"session_id": sid, "take_id": "take_000"})
    assert analyzed["metrics"]["sample_rate"] == sr
    assert "clean_sections" in analyzed

    reg_analyzed = _call_tool_json(
        mcp,
        "analyze_region",
        {
            "session_id": sid,
            "take_id": "take_000",
            "start_s": 0.2,
            "end_s": 1.4,
            "sensitivity": "high",
            "freq_range_hz": [40.0, 300.0],
        },
    )
    assert abs(reg_analyzed["metrics"]["duration_s"] - 1.2) < 0.05

    take1 = _call_tool_json(
        mcp,
        "process_take",
        {
            "session_id": sid,
            "parent_take_id": "take_000",
            "steps": [{"filter_name": "highpass_rumble", "params": {"cutoff_hz": 55.0}}],
            "label": "HPF 55Hz",
        },
    )
    take2 = _call_tool_json(
        mcp,
        "process_take",
        {
            "session_id": sid,
            "parent_take_id": "take_000",
            "steps": [
                {
                    "filter_name": "ambience_patch",
                    "params": {"start_s": 0.5, "end_s": 0.9},
                }
            ],
            "label": "Patched",
        },
    )

    vis = _call_tool_json(
        mcp,
        "render_spectrogram",
        {
            "session_id": sid,
            "take_id": take1["take_id"],
            "compare_to_take_id": "take_000",
            "start_s": 0.2,
            "end_s": 1.2,
        },
    )
    assert Path(vis["spectrogram_path"]).exists()
    assert Path(vis["diff_spectrogram_path"]).exists()

    ab = _call_tool_json(
        mcp,
        "create_ab_test",
        {
            "session_id": sid,
            "take_a_id": take1["take_id"],
            "take_b_id": take2["take_id"],
            "focus_start_s": 0.4,
            "focus_end_s": 1.1,
            "blind": False,
        },
    )
    assert "ui_url" in ab
    assert ab["workspace_dir"] == str(settings.workspace_dir)
    assert ab["ab_id"] == "ab_001"

    ab_list_res = _call_tool_json(mcp, "list_ab_tests", {"session_id": sid})
    assert len(ab_list_res["ab_tests"]) == 1

    feedback = _call_tool_json(
        mcp, "get_ab_feedback", {"session_id": sid, "ab_id": ab["ab_id"]}
    )
    assert feedback["verdict"] is None

    summary = _call_tool_json(
        mcp,
        "get_session_summary",
        {"session_id": sid, "include_events": False},
    )
    assert len(summary["takes"]) == 3
    assert summary["takes"][0]["events"] == []
    assert summary["takes"][0]["events_truncated"] is False
    assert "event_counts" in summary["takes"][0]
    assert "events_total" in summary["takes"][0]
    assert "step_metrics" in take1 and len(take1["step_metrics"]) == 1
    assert "clean_sections" in take1
    assert "clean_sections" in summary["takes"][0]

    clean_single = _call_tool_json(
        mcp, "find_clean_sections", {"session_id": sid, "take_id": take1["take_id"]}
    )
    assert clean_single["segment_count"] == 1
    assert clean_single["total_duration_s"] == summary["takes"][1]["metrics"]["duration_s"]

    src2 = save_wav(tmp_path / "ambience_part2.wav", audio, sr)
    sess2 = _call_tool_json(mcp, "create_session", {"audio_path": str(src2), "name": "Creek 2"})
    sid2 = sess2["session_id"]
    take_seam = _call_tool_json(
        mcp,
        "process_take",
        {
            "session_id": sid,
            "parent_take_id": "take_000",
            "steps": [{"filter_name": "highpass_rumble", "params": {"cutoff_hz": 55.0}}],
            "label": "Seam Padded",
            "next_audio_path": str(src2),
            "seam_context_s": 1.0,
        },
    )
    assert take_seam["metrics"]["sample_rate"] == sr
    clean_seq = _call_tool_json(
        mcp,
        "find_clean_sections",
        {
            "sequence": [
                {"session_id": sid, "take_id": take_seam["take_id"]},
                {"session_id": sid2, "take_id": "take_000"},
            ]
        },
    )
    assert clean_seq["segment_count"] == 2
    assert abs(clean_seq["total_duration_s"] - 4.0) < 0.05

    cal = _call_tool_json(
        mcp,
        "calibrate_sequence",
        {
            "sequence": [
                {"session_id": sid, "take_id": "take_000"},
                {"session_id": sid2, "take_id": "take_000"},
            ],
            "shared_steps_before_gain": [
                {"filter_name": "highpass_rumble", "params": {"cutoff_hz": 50.0}}
            ],
            "target_lufs": -16.0,
            "sample_windows_per_file": 2,
            "window_s": 1.0,
        },
    )
    assert isinstance(cal["recommended_gain_db"], float)
    seq_proc = _call_tool_json(
        mcp,
        "process_sequence",
        {
            "items": [
                {"session_id": sid, "export_path": str(tmp_path / "seq_1.wav")},
                {"session_id": sid2, "export_path": str(tmp_path / "seq_2.wav")},
            ],
            "shared_prefix_steps": [
                {"filter_name": "highpass_rumble", "params": {"cutoff_hz": 50.0}}
            ],
            "outer_fade_in_s": 0.1,
            "outer_fade_out_s": 0.1,
            "seam_context_s": 0.5,
        },
    )
    assert seq_proc["count"] == 2
    assert seq_proc["seams"]["pair_count"] == 1
    seam_chk = _call_tool_json(
        mcp,
        "verify_seams",
        {
            "sequence": [
                {"session_id": sid, "take_id": seq_proc["items"][0]["take_id"]},
                {"session_id": sid2, "take_id": seq_proc["items"][1]["take_id"]},
            ],
            "window_s": 1.0,
        },
    )
    assert seam_chk["pair_count"] == 1
    _call_tool_json(mcp, "delete_session", {"session_id": sid2})

    analyzed_filtered = _call_tool_json(
        mcp,
        "analyze_take",
        {
            "session_id": sid,
            "take_id": "take_000",
            "event_kinds": ["slow_pass"],
            "max_events": 10,
        },
    )
    assert analyzed_filtered["events_truncated"] is False
    assert all(e["kind"] == "slow_pass" for e in analyzed_filtered["events"])

    take_child = _call_tool_json(
        mcp,
        "process_take",
        {
            "session_id": sid,
            "parent_take_id": take1["take_id"],
            "steps": [{"filter_name": "gain", "params": {"gain_db": -1.0}}],
            "label": "Child of take1",
        },
    )
    with pytest.raises(ToolError) as exc_info:
        asyncio.run(
            mcp.call_tool(
                "delete_take",
                {"session_id": sid, "take_id": take1["take_id"], "cascade": False},
            )
        )
    assert take_child["take_id"] in str(exc_info.value)
    assert "cascade=True" in str(exc_info.value)

    del_t2 = _call_tool_json(
        mcp, "delete_take", {"session_id": sid, "take_id": take2["take_id"]}
    )
    assert take2["take_id"] in del_t2["deleted_take_ids"]

    summary_compact = _call_tool_json(
        mcp,
        "get_session_summary",
        {
            "session_id": sid,
            "include_events": False,
            "compact_takes": True,
            "take_ids": [take1["take_id"]],
        },
    )
    assert len(summary_compact["takes"]) == 1
    assert "recipe" not in summary_compact["takes"][0]

    reg_cmp = _call_tool_json(
        mcp,
        "analyze_region",
        {
            "session_id": sid,
            "take_id": take1["take_id"],
            "start_s": 0.2,
            "end_s": 1.6,
            "compare_to_take_id": "take_000",
        },
    )
    assert "cleanliness_assessment" in reg_cmp
    assert "band_metrics" in reg_cmp["metrics"]
    assert "comparison" in reg_cmp
    assert "bird_retention_db" in reg_cmp["comparison"]

    out_wav = tmp_path / "exported_clean.wav"
    exported = _call_tool_json(
        mcp,
        "export_take",
        {
            "session_id": sid,
            "take_id": take1["take_id"],
            "output_path": str(out_wav),
            "discard_take_audio": True,
        },
    )
    assert Path(exported["output_path"]).exists()
    assert exported["size_bytes"] > 0
    assert len(exported["sha256"]) == 64
    assert exported["integrated_lufs"] is not None
    assert exported["take_audio_discarded"] is True

    # Root take_000 audio must NEVER be discarded even if requested
    out_root = tmp_path / "exported_root.wav"
    exported_root = _call_tool_json(
        mcp,
        "export_take",
        {
            "session_id": sid,
            "take_id": "take_000",
            "output_path": str(out_root),
            "discard_take_audio": True,
        },
    )
    assert exported_root["take_audio_discarded"] is False

    # declip_from_backup path outside allowed_roots must fail via ToolError
    with pytest.raises(ToolError):
        asyncio.run(
            mcp.call_tool(
                "process_take",
                {
                    "session_id": sid,
                    "parent_take_id": "take_000",
                    "steps": [
                        {
                            "filter_name": "declip_from_backup",
                            "params": {"backup_audio_path": "C:/Windows/Temp/outside_bk.wav"},
                        }
                    ],
                },
            )
        )

    del_sess = _call_tool_json(mcp, "delete_session", {"session_id": sid})
    assert del_sess["deleted"] is True

    # Path traversal outside allowed_roots must fail via ToolError
    with pytest.raises(ToolError):
        asyncio.run(
            mcp.call_tool(
                "create_session",
                {"audio_path": "C:/Windows/System32/drivers/etc/hosts"},
            )
        )

    # Path traversal job_id must fail via ToolError
    with pytest.raises(ToolError):
        asyncio.run(mcp.call_tool("get_job_status", {"job_id": "../escape"}))


def test_mcp_async_sequence_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    from ambient_sound.jobs import JobManager
    from ambient_sound.models import JobRecord

    ws = tmp_path / "workspace"
    monkeypatch.setenv("AMBIENTSOUND_WORKSPACE", str(ws))
    monkeypatch.setenv("AMBIENTSOUND_ALLOWED_ROOTS", str(tmp_path))

    mcp = create_mcp_server()
    sr = 44100
    rng = np.random.default_rng(77)
    files = []
    for k in range(2):
        arr = rng.normal(0.0, 0.02, size=(sr * 2, 2))
        p = save_wav(tmp_path / f"split_{k}.wav", arr, sr)
        sess = _call_tool_json(
            mcp, "create_session", {"audio_path": str(p), "name": f"Split{k}"}
        )
        files.append(sess["session_id"])

    # calibrate_sequence must reject out-of-root declip_from_backup paths
    with pytest.raises(ToolError):
        asyncio.run(
            mcp.call_tool(
                "calibrate_sequence",
                {
                    "sequence": [{"session_id": files[0], "take_id": "take_000"}],
                    "shared_steps_before_gain": [
                        {
                            "filter_name": "declip_from_backup",
                            "params": {"backup_audio_path": "C:/Windows/Temp/outside_cal.wav"},
                        }
                    ],
                },
            )
        )

    # export_take must refuse to overwrite immutable take_000 audio
    take0_wav = ws / "sessions" / files[0] / "takes" / "take_000" / "audio.wav"
    with pytest.raises(ToolError):
        asyncio.run(
            mcp.call_tool(
                "export_take",
                {
                    "session_id": files[0],
                    "take_id": "take_000",
                    "output_path": str(take0_wav),
                },
            )
        )

    # Peer running job record must NOT be corrupted by list_jobs(reconcile_stale=False)
    peer_jm = JobManager(ws)
    peer_jm._save_job(
        JobRecord(
            job_id="job_peerlive",
            status="running",
            owner_pid=999999,
            created_at="2026-11-21T00:00:00+00:00",
            updated_at="2026-11-21T00:00:00+00:00",
            total_items=2,
        )
    )
    ro_jobs = peer_jm.list_jobs(reconcile_stale=False)
    assert any(j.job_id == "job_peerlive" and j.status == "running" for j in ro_jobs)

    job = _call_tool_json(
        mcp,
        "start_sequence_job",
        {
            "items": [
                {
                    "session_id": files[0],
                    "export_path": str(tmp_path / "out_0.wav"),
                    "label": "seq0",
                },
                {
                    "session_id": files[1],
                    "export_path": str(tmp_path / "out_1.wav"),
                    "label": "seq1",
                },
            ],
            "shared_prefix_steps": [
                {"filter_name": "highpass_rumble", "params": {"cutoff_hz": 55.0}}
            ],
            "verify_seams": True,
            "clean_sections": True,
        },
    )
    job_id = job["job_id"]
    deadline = time.monotonic() + 20.0
    cur = job
    while cur["status"] in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(0.1)
        cur = _call_tool_json(mcp, "get_job_status", {"job_id": job_id})

    assert cur["status"] == "completed"
    assert cur["progress_pct"] == 100.0
    assert cur["result"]["count"] == 2
    assert (tmp_path / "out_0.wav").exists()
    assert (tmp_path / "out_1.wav").exists()

    listed = _call_tool_json(mcp, "list_jobs", {"limit": 5})
    assert any(j["job_id"] == job_id for j in listed["jobs"])

    cancelled = _call_tool_json(mcp, "cancel_job", {"job_id": job_id})
    assert cancelled["status"] == "completed"


def test_v051_mcp_compile_clean_sections_and_sequence_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analysis_mod, "YOUTUBE_ASMR_DURATIONS_MIN", (1,))
    ws = tmp_path / ".ambientsound"
    settings = Settings(workspace_dir=ws, allowed_roots=[tmp_path])
    mcp = create_mcp_server(settings)

    sr = 8000
    n_each = sr * 35
    total_n = n_each * 2
    t = np.arange(total_n, dtype=np.float64) / sr
    rng = np.random.default_rng(512)
    full = rng.normal(0.0, 0.015, size=(total_n, 2)) + (
        0.03 * np.sin(2.0 * np.pi * 190.0 * t)
    )[:, None]

    raw_dir = tmp_path / "raw_mcp"
    p1 = save_wav(raw_dir / "ZOOM0020.WAV", full[:n_each], sr, subtype="PCM_24")
    p2 = save_wav(raw_dir / "ZOOM0021.WAV", full[n_each:], sr, subtype="PCM_24")
    s1 = _call_tool_json(
        mcp, "create_session", {"audio_path": str(p1), "name": "S20", "copy_source": False}
    )
    s2 = _call_tool_json(
        mcp, "create_session", {"audio_path": str(p2), "name": "S21", "copy_source": False}
    )

    seq_out = tmp_path / "out_seq"
    proc = _call_tool_json(
        mcp,
        "process_sequence",
        {
            "items": [
                {"session_id": s1["session_id"]},
                {"session_id": s2["session_id"]},
            ],
            "shared_prefix_steps": [
                {"filter_name": "highpass_rumble", "params": {"cutoff_hz": 45.0}}
            ],
            "output_dir": str(seq_out),
            "compile_clean_targets": ["1 min"],
            "include_combined_full": True,
            "filename_prefix": "beehive",
            "clean_fade_in_s": 0.1,
            "clean_fade_out_s": 0.1,
            "discard_take_audio_after_export": True,
        },
    )
    assert proc["count"] == 2
    assert (seq_out / "beehive_ZOOM0020_master.wav").exists()
    assert (seq_out / "beehive_ZOOM0021_master.wav").exists()
    assert proc["compiled_clean_sections"]["compiled_count"] == 1
    assert (seq_out / "clean_sections_manifest.json").exists()
    assert Path(proc["compiled_clean_sections"]["combined_full"]["output_path"]).exists()

    # Trimmed child take reports source_offset_s & parent_start_s on detected events
    trimmed = _call_tool_json(
        mcp,
        "process_take",
        {
            "session_id": s1["session_id"],
            "parent_take_id": "take_000",
            "steps": [{"filter_name": "trim_range", "params": {"start_s": 5.0, "end_s": 15.0}}],
            "label": "Trimmed offset test",
        },
    )
    an_trim = _call_tool_json(
        mcp,
        "analyze_take",
        {"session_id": s1["session_id"], "take_id": trimmed["take_id"]},
    )
    assert abs(an_trim["source_offset_s"] - 5.0) < 1e-3
    if an_trim["events"]:
        assert "parent_start_s" in an_trim["events"][0]

    sync_out = tmp_path / "out_sync"
    seq_pairs = [
        {"session_id": s1["session_id"], "take_id": proc["items"][0]["take_id"]},
        {"session_id": s2["session_id"], "take_id": proc["items"][1]["take_id"]},
    ]
    comp = _call_tool_json(
        mcp,
        "compile_clean_sections",
        {
            "sequence": seq_pairs,
            "output_dir": str(sync_out),
            "targets": ["largest"],
            "include_full_masters": True,
            "fade_in_s": 0.1,
            "fade_out_s": 0.1,
        },
    )
    assert comp["compiled_count"] == 1
    assert len(comp["full_masters"]) == 2
    assert Path(comp["full_masters"][0]["output_path"]).name == "ZOOM0020_master.wav"
    assert comp["full_masters"][0]["prefix_applied"] is False
    assert (sync_out / "clean_sections_manifest.json").exists()

    async_out = tmp_path / "out_async"
    job = _call_tool_json(
        mcp,
        "start_compile_clean_sections_job",
        {
            "sequence": seq_pairs,
            "output_dir": str(async_out),
            "targets": ["1 min"],
            "fade_in_s": 0.1,
            "fade_out_s": 0.1,
        },
    )
    assert job["kind"] == "compile_clean_sections"
    cur = _call_tool_json(
        mcp,
        "wait_for_job",
        {"job_id": job["job_id"], "timeout_s": 20.0, "poll_interval_s": 0.1},
    )
    assert cur["status"] == "completed"
    assert cur["progress_pct"] == 100.0
    assert cur["result"]["compiled_count"] == 1
    assert len(cur["result"]["compiled_sections"]) == 1
    assert cur["items_summary"][0]["state"] == "done"
    assert cur["items_summary"][0]["kind"] == "section"
    assert (async_out / "clean_sections_manifest.json").exists()

    compact_status = _call_tool_json(
        mcp,
        "get_job_status",
        {"job_id": job["job_id"], "include_result": False},
    )
    assert compact_status["result"] is None
    assert compact_status["has_result"] is True
    assert compact_status["manifest_path"].endswith("clean_sections_manifest.json")

    jobs_compact = _call_tool_json(mcp, "list_jobs", {"limit": 5})
    assert jobs_compact["jobs"][0]["result"] is None
    assert jobs_compact["jobs"][0]["has_result"] is True

    # Active output_dir lock rejects duplicate live jobs on the same folder
    import os

    from ambient_sound.jobs import JobManager
    from ambient_sound.models import JobRecord

    busy_dir = (tmp_path / "busy_out").resolve()
    jm = JobManager(ws)
    live_peer_pid = os.getppid()
    jm._save_job(
        JobRecord(
            job_id="job_activelock",
            kind="compile_clean_sections",
            status="running",
            owner_pid=live_peer_pid,
            created_at="2026-11-21T00:00:00+00:00",
            updated_at="2026-11-21T00:00:00+00:00",
            output_dir=str(busy_dir),
        )
    )
    with pytest.raises(ToolError):
        asyncio.run(
            mcp.call_tool(
                "start_compile_clean_sections_job",
                {"sequence": seq_pairs, "output_dir": str(busy_dir)},
            )
        )
    # Remove synthetic live-PID stub so workspace job list stays clean
    jm._job_path("job_activelock").unlink(missing_ok=True)



