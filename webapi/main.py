"""FastAPI entry point for the React verification workspace."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from webapi.config import public_case_options
from webapi.execution import (
    artifact_path,
    default_prompt,
    execute_task,
    list_common_valid_times,
    list_initializations,
)
from webapi.jobs import jobs
from webapi.schemas import CommonTimeRequest, PromptRequest, SelectionRequest, TaskRequest


load_dotenv()
app = FastAPI(
    title="强对流预报检验智能体 API",
    version="1.0.0-test",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ready", "service": "verification-api"}


@app.get("/api/catalog/options")
def catalog_options():
    return {"cases": public_case_options()}


@app.post("/api/catalog/initializations")
def catalog_initializations(selection: SelectionRequest):
    try:
        return {
            "initializations": list_initializations(
                selection.case_id, selection.models, selection.complete_only
            )
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/catalog/common-valid-times")
def common_times(selection: CommonTimeRequest):
    try:
        return {
            "valid_times": list_common_valid_times(
                selection.case_id,
                selection.models,
                selection.initialization_ids,
            )
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/tasks/default-prompt")
def task_default_prompt(request: PromptRequest):
    return {"prompt": default_prompt(request.case_id, request.models, request.scope)}


@app.post("/api/jobs", status_code=202)
def create_job(request: TaskRequest):
    job = jobs.submit(request.model_dump(), execute_task)
    return job.public()


@app.get("/api/jobs")
def recent_jobs():
    return {"jobs": jobs.recent()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return jobs.get(job_id).public()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    try:
        job = jobs.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc

    async def stream():
        cursor = 0
        while True:
            with job.lock:
                pending = list(job.events[cursor:])
                state = job.status
            for event in pending:
                cursor += 1
                yield f"id: {event['index']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            if state in {"complete", "failed"} and cursor >= len(job.events):
                yield f"event: end\ndata: {json.dumps({'status': state})}\n\n"
                break
            await asyncio.sleep(0.35)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/artifacts/{name}")
def job_artifact(job_id: str, name: str):
    try:
        path = artifact_path(job_id, name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc
    return FileResponse(path, filename=path.name)


FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def frontend(path: str):
    index = FRONTEND_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    return HTMLResponse(
        "<h1>Verification API is ready</h1>"
        "<p>Run <code>npm run dev</code> in <code>frontend/</code>, or build the frontend.</p>"
    )


def run():
    import uvicorn

    uvicorn.run("webapi.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
