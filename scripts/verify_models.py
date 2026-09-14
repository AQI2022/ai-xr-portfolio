"""Run real local model integrations; no test doubles and no external LLM key."""
import asyncio
import json
from pathlib import Path

import httpx

from ai_xr.api import create_app
from ai_xr.config import Settings


async def main():
    app = create_app(Settings(provider="transformers", local_model="models/qwen2.5-0.5b-instruct"))
    transport = httpx.ASGITransport(app=app)
    report = {"provider": "local Qwen2.5-0.5B-Instruct", "test_doubles": False}
    async with httpx.AsyncClient(transport=transport, base_url="http://local", timeout=180) as client:
        response = await client.post("/rag/query", json={"question": "Unity如何避免网络调用阻塞主线程？", "top_k": 2})
        response.raise_for_status()
        report["rag"] = response.json()
        response = await client.post("/jobs/match", json={"resume": "I use Python, Unity and FastAPI to build RAG applications."})
        response.raise_for_status()
        report["jobs"] = response.json()
        response = await client.post("/jobs/agent", json={"resume": "Python Unity FastAPI RAG"})
        response.raise_for_status()
        report["job_agent"] = response.json()
        response = await client.post("/agent/chat", json={"message": '请调用propose_scene_action，action为rotate，target为demo_cube，value为45。', "session_id": "real-model-test"})
        response.raise_for_status()
        report["agent"] = response.json()
        for pending in report["agent"]["pending_actions"]:
            result = await client.post("/agent/confirm", json={"session_id": "real-model-test", "action_id": pending["action_id"]})
            result.raise_for_status()
            report["confirmation"] = result.json()
        response = await client.post("/speech/synthesize", json={"text": "Hello. This is the mixed reality assistant. Please confirm before rotating the cube."})
        report["tts"] = {"http_status": response.status_code, "bytes": len(response.content)}
        if response.status_code == 200:
            Path("data/integration-speech.wav").write_bytes(response.content)
            response = await client.post("/speech/transcribe", files={"file": ("speech.wav", response.content, "audio/wav")})
            report["stt"] = {"http_status": response.status_code, "result": response.json()}
    Path("evidence/integration-models.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rag_citation_valid": report["rag"]["citation_valid"], "agent_stop": report["agent"]["stop_reason"], "tts": report["tts"], "stt": report.get("stt")}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
