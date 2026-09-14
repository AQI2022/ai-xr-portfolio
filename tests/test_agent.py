import json

import pytest

from ai_xr.agent import execute_tool, run_agent
from ai_xr.rag import Retriever, chunk_document
from ai_xr.store import Store


@pytest.fixture
def services(tmp_path):
    store = Store(tmp_path/"agent.sqlite")
    doc, chunks = chunk_document("x.txt", b"Unity uses scene transforms and anchors.")
    store.replace_document(doc, chunks)
    return store, Retriever(store)


async def test_unknown_tool_cannot_execute(services):
    store, retriever = services
    assert "error" in await execute_tool("run_shell", {"cmd": "anything"}, retriever, store, "session123")


async def test_invalid_scene_arguments_are_rejected(services):
    store, retriever = services
    result = await execute_tool("propose_scene_action", {"action": "delete", "target": "/"}, retriever, store, "session123")
    assert "error" in result


async def test_confirmation_is_session_bound_and_single_use(services):
    store, retriever = services
    result = await execute_tool("propose_scene_action", {"action": "rotate", "target": "demo_cube", "value": 45}, retriever, store, "session123")
    assert result["status"] == "pending_confirmation"
    assert store.consume_pending(result["action_id"], "another123") is None
    assert store.consume_pending(result["action_id"], "session123")["value"] == 45
    assert store.consume_pending(result["action_id"], "session123") is None


async def test_real_tool_protocol_round_trip(services):
    store, retriever = services
    class ProtocolProvider:
        mode = "scripted-test-provider"
        calls = 0
        async def chat(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                return {"content": None, "tool_calls": [{"id": "call1", "type": "function",
                    "function": {"name": "search_knowledge", "arguments": '{"query":"Unity"}'}}]}
            assert messages[-1]["role"] == "tool"
            assert json.loads(messages[-1]["content"])["citations"]
            return {"content": "Use transforms [S1]."}
    result = await run_agent("Unity", "session123", store, retriever, ProtocolProvider())
    assert result["stop_reason"] == "model_complete" and len(result["trace"]) == 1
    assert len(store.history("session123")) == 2
    assert store.history("another123") == []


async def test_step_budget_prevents_infinite_loop(services):
    store, retriever = services
    class Endless:
        mode = "test"
        async def chat(self, messages, tools=None):
            return {"tool_calls": [{"id": "loop", "function": {"name":"search_knowledge","arguments":'{"query":"Unity"}'}}]}
    result = await run_agent("Unity", "session123", store, retriever, Endless(), max_steps=2)
    assert result["stop_reason"] == "step_limit" and len(result["trace"]) == 2


def test_history_is_bounded_and_deletable(services):
    store, _ = services
    for i in range(40):
        store.append_memory("session123", "user", str(i))
    assert len(store.history("session123", 100)) == 24
    store.delete_memory("session123")
    assert store.history("session123") == []
