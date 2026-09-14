import pytest


def test_mcp_tools_are_registered():
    pytest.importorskip("mcp")
    from ai_xr.mcp_server import analyze_resume_skills
    assert '"Python"' in analyze_resume_skills("Python")
