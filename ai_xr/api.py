import asyncio
import hmac
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from pydantic import BaseModel, Field

from .agent import ActionArgs, execute_tool, run_agent
from .config import Settings
from .jobs import Job, llm_skill_evidence, rank_jobs, search_jobs
from .job_agent import run_job_agent
from .providers import ProviderError, make_provider
from .rag import Retriever, answer_question, chunk_document
from .speech import Transcriber
from .store import Store
from .vision import Detector, describe_image


class Query(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)


class SpeechText(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class Chat(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex, pattern=r"^[a-zA-Z0-9_-]{8,80}$")
    scene: dict | None = None


class JobMatch(BaseModel):
    resume: str = Field(min_length=1, max_length=30000)
    jobs: list[Job] = Field(default_factory=list, max_length=100)


class Confirmation(BaseModel):
    session_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{8,80}$")
    action_id: str = Field(pattern=r"^[a-f0-9]{32}$")


class Proposal(ActionArgs):
    session_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{8,80}$")


def create_app(settings=None, provider=None):
    settings = (settings or Settings()).prepare()
    store = Store(settings.data_dir / "portfolio.sqlite")
    retriever = Retriever(store, settings.embedding_model, settings.reranker_model)
    provider = provider or make_provider(settings)
    detector, transcriber = Detector(settings.yolo_model), Transcriber()
    app = FastAPI(title="AI XR Portfolio", version="0.1.0")
    app.state.store, app.state.retriever, app.state.provider = store, retriever, provider

    async def authorize(x_api_key: str = Header(default="")):
        if settings.app_api_key and not hmac.compare_digest(x_api_key, settings.app_api_key):
            raise HTTPException(401, "Invalid API key")
    protected = [Depends(authorize)]

    @app.exception_handler(ProviderError)
    async def provider_error(request, error):
        return JSONResponse(status_code=503, content={"detail": str(error)})

    @app.get("/", include_in_schema=False)
    def demo():
        return FileResponse(Path(__file__).parent / "demo.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "provider": provider.mode, "retrieval": retriever.mode,
                "yolo_ready": Path(settings.yolo_model).is_file(), "auth_enabled": bool(settings.app_api_key)}

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    async def read_upload(file):
        content = await file.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(413, "Upload exceeds configured limit")
        return content

    @app.post("/rag/documents", dependencies=protected)
    async def upload(file: UploadFile = File(...)):
        content = await read_upload(file)
        try:
            doc_id, chunks = await asyncio.to_thread(chunk_document, file.filename or "document.txt", content)
            await asyncio.to_thread(store.replace_document, doc_id, chunks)
        except Exception as error:
            raise HTTPException(422, "Unable to parse document; check file format and text content") from error
        return {"document_id": doc_id, "chunks": len(chunks)}

    @app.post("/rag/query", dependencies=protected)
    async def query(body: Query):
        return await answer_question(body.question, retriever, provider, body.top_k)

    @app.post("/rag/stream", dependencies=protected)
    async def stream(body: Query):
        async def events():
            yield "event: status\ndata: " + json.dumps({"stage": "retrieving"}) + "\n\n"
            try:
                result = await answer_question(body.question, retriever, provider, body.top_k)
                # Stage/result streaming. Do not label buffered final text as model token streaming.
                yield "event: result\ndata: " + json.dumps(result, ensure_ascii=False) + "\n\n"
            except ProviderError as error:
                yield "event: error\ndata: " + json.dumps({"detail": str(error)}) + "\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/jobs/match", dependencies=protected)
    async def match(body: JobMatch):
        jobs = [j.model_dump(mode="json") for j in body.jobs] or store.list_jobs()
        extraction = await llm_skill_evidence(body.resume, provider)
        ranking = rank_jobs(body.resume, jobs)
        return {"extraction": extraction, "ranking": ranking, "trace": [
            {"tool": "extract_skills", "output_count": len(extraction["skills"])},
            {"tool": "rank_jobs", "output_count": len(ranking)},
            {"tool": "gap_analysis", "status": "complete"}], "auto_apply": False}

    @app.post("/jobs/import", dependencies=protected)
    def import_jobs(jobs: list[Job]):
        if len(jobs) > 100:
            raise HTTPException(422, "At most 100 jobs per batch")
        store.save_jobs([job.model_dump(mode="json") for job in jobs])
        return {"saved": len(jobs)}

    @app.post("/jobs/agent", dependencies=protected)
    async def job_agent(body: JobMatch):
        jobs = [j.model_dump(mode="json") for j in body.jobs] or store.list_jobs()
        return await run_job_agent(body.resume, jobs, provider, settings.search_api_key)

    @app.post("/jobs/search", dependencies=protected)
    async def job_search(body: Query):
        try:
            return {"results": await search_jobs(body.question, settings.search_api_key)}
        except ValueError as error:
            raise HTTPException(503, str(error)) from error

    @app.post("/agent/chat", dependencies=protected)
    async def chat(body: Chat):
        if body.scene and len(json.dumps(body.scene)) > 4000:
            raise HTTPException(422, "Scene context too large")
        return await run_agent(body.message, body.session_id, store, retriever, provider, body.scene)

    @app.post("/agent/propose", dependencies=protected)
    async def propose(body: Proposal):
        return await execute_tool("propose_scene_action", body.model_dump(exclude={"session_id"}),
                                  retriever, store, body.session_id)

    @app.post("/agent/confirm", dependencies=protected)
    def confirm(body: Confirmation):
        action = store.consume_pending(body.action_id, body.session_id)
        if not action:
            raise HTTPException(409, "Action expired, already used or belongs to another session")
        return {"status": "confirmed", "action": action}

    @app.delete("/agent/memory/{session_id}", dependencies=protected)
    def clear_memory(session_id: str):
        store.delete_memory(session_id)
        return {"deleted": True}

    @app.post("/vision/detect", dependencies=protected)
    async def detect(file: UploadFile = File(...)):
        try:
            return await asyncio.to_thread(detector.detect, await read_upload(file))
        except (FileNotFoundError, ImportError) as error:
            raise HTTPException(503, str(error)) from error
        except (ValueError, OSError) as error:
            raise HTTPException(422, "Invalid or oversized image") from error

    @app.post("/vision/describe", dependencies=protected)
    async def describe(file: UploadFile = File(...)):
        try:
            return await describe_image(await read_upload(file), provider)
        except ValueError as error:
            raise HTTPException(503, str(error)) from error

    @app.post("/speech/transcribe", dependencies=protected)
    async def transcribe(file: UploadFile = File(...)):
        content = await read_upload(file)
        with tempfile.TemporaryDirectory(prefix="xr-audio-", dir=settings.data_dir) as folder:
            path = Path(folder) / "audio.bin"
            path.write_bytes(content)
            try:
                return await asyncio.to_thread(transcriber.transcribe, path)
            except ImportError as error:
                raise HTTPException(503, "Install speech extras") from error
            except Exception as error:
                raise HTTPException(422, "Unable to transcribe audio") from error

    @app.post("/speech/synthesize", dependencies=protected)
    async def synthesize(body: SpeechText):
        with tempfile.TemporaryDirectory(prefix="xr-tts-", dir=settings.data_dir) as folder:
            path = Path(folder).resolve()/"speech.wav"
            try:
                result = await asyncio.to_thread(subprocess.run,
                    [sys.executable, "-m", "ai_xr.tts_worker"],
                    input=json.dumps({"text":body.text,"path":str(path)}), text=True,
                    capture_output=True, timeout=60)
                if result.returncode != 0 or not path.is_file():
                    raise ValueError("System speech engine unavailable")
                audio = path.read_bytes()
            except Exception as error:
                raise HTTPException(503,"System TTS unavailable; install speech extras and a system voice") from error
        return Response(content=audio, media_type="audio/wav")

    return app


app = create_app()
