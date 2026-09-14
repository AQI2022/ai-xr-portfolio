import pytest
from fastapi.testclient import TestClient

from ai_xr.api import create_app
from ai_xr.config import Settings


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(Settings(data_dir=tmp_path, app_api_key="", max_upload_bytes=10000)))


def test_full_upload_query_flow(client):
    response = client.post("/rag/documents", files={"file": ("test.txt", "HoloLens 空间锚点记录位置。".encode(), "text/plain")})
    assert response.status_code == 200
    result = client.post("/rag/query", json={"question":"HoloLens 空间锚点"}).json()
    assert result["citations"][0]["title"] == "test.txt"
    assert "[S1]" in result["answer"]


def test_upload_limit(client):
    assert client.post("/rag/documents", files={"file": ("large.txt", b"x"*10001)}).status_code == 413


def test_corrupt_docx_returns_user_error(client):
    assert client.post("/rag/documents", files={"file": ("broken.docx", b"notzip")}).status_code == 422


def test_query_input_validation(client):
    assert client.post("/rag/query", json={"question":"", "top_k":0}).status_code == 422


def test_optional_api_key(tmp_path):
    client = TestClient(create_app(Settings(data_dir=tmp_path, app_api_key="test-secret")))
    assert client.post("/rag/query", json={"question":"Unity"}).status_code == 401
    assert client.post("/rag/query", json={"question":"Unity"}, headers={"X-API-Key":"test-secret"}).status_code == 200
    assert client.get("/health").status_code == 200


def test_confirmation_protocol(client):
    proposal = client.post("/agent/propose", json={"session_id":"session123", "action":"rotate", "value":45}).json()
    payload = {"session_id":"session123", "action_id":proposal["action_id"]}
    assert client.post("/agent/confirm", json=payload).status_code == 200
    assert client.post("/agent/confirm", json=payload).status_code == 409


def test_sse_contract(client):
    result = client.post("/rag/stream", json={"question":"Unity"})
    assert "text/event-stream" in result.headers["content-type"]
    assert "event: status" in result.text and "event: result" in result.text


def test_demo_and_health(client):
    assert "AI XR Lab" in client.get("/").text
    assert client.get("/health").json()["provider"] == "extractive"
