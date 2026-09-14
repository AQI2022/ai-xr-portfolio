import json

import pytest

from ai_xr.job_agent import run_job_agent
from ai_xr.providers import ExtractiveProvider


@pytest.mark.asyncio
async def test_offline_tools():
    result = await run_job_agent("Python", [], ExtractiveProvider())
    assert result["auto_apply"] is False
    assert [t["tool"] for t in result["trace"]] == ["extract_candidate_skills", "match_supplied_jobs"]


@pytest.mark.asyncio
async def test_unauthorized_tool_and_bound():
    class Fake:
        mode = "test-double"

        async def chat(self, messages, tools=None):
            return {"tool_calls": [{"id": "x", "function": {"name": "send_application",
                                     "arguments": json.dumps({"email": "nobody@example.invalid"})}}]}
    result = await run_job_agent("Python", [], Fake(), max_steps=2)
    assert len(result["trace"]) == 2
    assert result["stop_reason"] == "step_limit"
    assert all("error" in t["result"] for t in result["trace"])
