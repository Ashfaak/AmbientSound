"""End-to-end tests for the AmbientSound MCP tool surface and path sandboxing."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError

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
    }.issubset(tool_names)

    health = _call_tool_json(mcp, "health_check", {})
    assert health["status"] == "ok"
    assert health["filter_count"] == 15

    filters_res = _call_tool_json(mcp, "list_filters", {})
    filters = filters_res["filters"]
    assert isinstance(filters, list)
    assert any(f["name"] == "ambience_patch" for f in filters)

    sr = 44100
    rng = np.random.default_rng(21)
    audio = rng.normal(0.0, 0.03, size=(sr * 2, 2))
    src = save_wav(tmp_path / "ambience.wav", audio, sr)

    session = _call_tool_json(mcp, "create_session", {"audio_path": str(src), "name": "Creek"})
    sid = session["session_id"]
    assert session["takes"][0]["take_id"] == "take_000"
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

    out_wav = tmp_path / "exported_clean.wav"
    exported = _call_tool_json(
        mcp,
        "export_take",
        {"session_id": sid, "take_id": take1["take_id"], "output_path": str(out_wav)},
    )
    assert Path(exported["output_path"]).exists()

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
