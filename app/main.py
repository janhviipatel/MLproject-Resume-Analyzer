"""FastAPI app. Run: uvicorn app.main:app --reload"""
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from .analyzer import Engine
from .parser import extract_text, normalize

ROOT = Path(__file__).parent
state = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["engine"] = Engine()
    yield


app = FastAPI(title="ResumeLens", lifespan=lifespan)


@app.get("/api/health")
def health():
    e = state["engine"]
    return {"status": "ok", "embedder": e.embedder.name}


@app.get("/api/metrics")
def metrics():
    f = ROOT / "models" / "metrics.json"
    if not f.exists():
        raise HTTPException(status_code=404, detail="Run `python -m app.evaluate` first.")
    return json.loads(f.read_text())


@app.get("/api/sample")
def sample():
    d = ROOT / "data"
    return {"resume": (d / "sample_resume.txt").read_text(encoding="utf-8"),
            "job": (d / "sample_job.txt").read_text(encoding="utf-8")}


@app.post("/api/analyze")
async def analyze(resume_text: str = Form(""), job_text: str = Form(""),
                  resume_file: Optional[UploadFile] = File(None)):
    text = resume_text
    if resume_file is not None and resume_file.filename:
        try:
            text = extract_text(resume_file.filename, await resume_file.read())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    text = normalize(text)
    if len(text.split()) < 30:
        raise HTTPException(status_code=400, detail="Add at least a few lines of resume text, or upload a file.")
    if len(text) > 30000 or len(job_text) > 20000:
        raise HTTPException(status_code=400, detail="Input is too long.")
    return state["engine"].analyze(text, job_text)


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
