#!/usr/bin/env python3
"""YouTube Script Extractor - Web Server."""

import asyncio
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from transcriber import process_video
from formatter import LLM_MODELS, get_models_sorted, get_languages

app = FastAPI(title="YouTube Script Extractor")
templates = Jinja2Templates(directory="templates")

OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 진행 중인 작업 추적
jobs: dict[str, dict] = {}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/llm-models")
async def get_llm_models(sort: str = "price"):
    """LLM 모델 목록을 정렬하여 반환한다."""
    models = get_models_sorted(sort_by=sort)
    return {"models": models}


@app.get("/api/languages")
async def get_supported_languages():
    """번역 지원 언어 목록을 반환한다."""
    return {"languages": get_languages()}


@app.get("/api/vision-models")
async def get_vision_models(sort: str = "price"):
    """Vision 지원 LLM 모델 목록을 반환한다."""
    key = "price_rank" if sort == "price" else "quality_rank"
    models = [
        {"id": k, **v} for k, v in LLM_MODELS.items()
        if v.get("supports_vision", False)
    ]
    return {"models": sorted(models, key=lambda x: x[key])}


@app.post("/api/transcribe")
async def start_transcription(request: Request):
    body = await request.json()
    url = body.get("url", "").strip()
    mode = body.get("mode", "api")
    api_key = body.get("api_key", "") or os.environ.get("OPENAI_API_KEY", "")
    model_size = body.get("model_size", "base")
    formats = body.get("formats", ["txt", "srt"])
    output_dir = body.get("output_dir", "").strip() or OUTPUT_DIR

    # LLM 옵션 (전역 기본값)
    md_llm = body.get("md_llm", "")
    md_api_key = body.get("md_api_key", "")
    md_ollama_model = body.get("md_ollama_model", "llama3.2")
    translate_lang = body.get("translate_lang", "")

    # 단계별 LLM 오버라이드
    format_llm = body.get("format_llm", "")
    format_api_key = body.get("format_api_key", "")
    translate_llm = body.get("translate_llm", "")
    translate_api_key = body.get("translate_api_key", "")
    keyframe_llm = body.get("keyframe_llm", "")
    keyframe_api_key = body.get("keyframe_api_key", "")

    # 키프레임 옵션
    enable_keyframes = body.get("enable_keyframes", False)
    keyframe_method = body.get("keyframe_method", "scene")
    keyframe_interval = body.get("keyframe_interval", 30)

    if not url:
        return {"error": "YouTube URL을 입력해주세요."}

    if mode == "api" and not api_key:
        return {"error": "API 모드에서는 OpenAI API 키가 필요합니다."}

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "started", "progress": [], "result": None, "error": None}

    asyncio.create_task(
        _run_transcription(
            job_id, url, mode, api_key, model_size, formats, output_dir,
            md_llm=md_llm, md_api_key=md_api_key, md_ollama_model=md_ollama_model,
            translate_lang=translate_lang,
            format_llm=format_llm, format_api_key=format_api_key,
            translate_llm=translate_llm, translate_api_key=translate_api_key,
            keyframe_llm=keyframe_llm, keyframe_api_key=keyframe_api_key,
            enable_keyframes=enable_keyframes,
            keyframe_method=keyframe_method, keyframe_interval=keyframe_interval,
        )
    )

    return {"job_id": job_id}


async def _run_transcription(
    job_id: str,
    url: str,
    mode: str,
    api_key: str,
    model_size: str,
    formats: list[str],
    output_dir: str = OUTPUT_DIR,
    md_llm: str = "",
    md_api_key: str = "",
    md_ollama_model: str = "llama3.2",
    translate_lang: str = "",
    format_llm: str = "",
    format_api_key: str = "",
    translate_llm: str = "",
    translate_api_key: str = "",
    keyframe_llm: str = "",
    keyframe_api_key: str = "",
    enable_keyframes: bool = False,
    keyframe_method: str = "scene",
    keyframe_interval: int = 30,
):
    def on_progress(msg: str):
        jobs[job_id]["progress"].append(msg)

    try:
        result = await asyncio.to_thread(
            process_video,
            url=url,
            mode=mode,
            api_key=api_key,
            model_size=model_size,
            output_dir=output_dir,
            formats=formats,
            on_progress=on_progress,
            md_llm=md_llm or None,
            md_api_key=md_api_key or None,
            md_ollama_model=md_ollama_model,
            translate_lang=translate_lang or None,
            format_llm=format_llm or None,
            format_api_key=format_api_key or None,
            translate_llm=translate_llm or None,
            translate_api_key=translate_api_key or None,
            keyframe_llm=keyframe_llm or None,
            keyframe_api_key=keyframe_api_key or None,
            enable_keyframes=enable_keyframes,
            keyframe_method=keyframe_method,
            keyframe_interval=keyframe_interval,
        )
        jobs[job_id]["status"] = "completed"
        # 절대 경로로 변환하여 저장 위치를 정확히 표시
        abs_output = os.path.abspath(output_dir)
        jobs[job_id]["result"] = {
            "title": result["title"],
            "language": result["language"],
            "text": result["text"],
            "files": {
                fmt: os.path.basename(path) for fmt, path in result["files"].items()
            },
            "output_dir": abs_output,
            "output_dir_raw": output_dir,
        }
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"error": "작업을 찾을 수 없습니다."}
    return job


@app.get("/api/stream/{job_id}")
async def stream_status(job_id: str):
    async def event_generator():
        seen = 0
        while True:
            job = jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'type': 'error', 'message': '작업을 찾을 수 없습니다.'})}\n\n"
                break

            # 새 진행 메시지 전송
            while seen < len(job["progress"]):
                msg = job["progress"][seen]
                if isinstance(msg, dict):
                    yield f"data: {json.dumps({'type': 'progress', **msg})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'progress', 'message': msg})}\n\n"
                seen += 1

            if job["status"] == "completed":
                yield f"data: {json.dumps({'type': 'completed', 'result': job['result']})}\n\n"
                break
            elif job["status"] == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': job['error']})}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/download/{filename}")
async def download_file(filename: str, dir: str = ""):
    base_dir = dir if dir else OUTPUT_DIR
    file_path = os.path.join(base_dir, filename)
    if not os.path.exists(file_path):
        return {"error": "파일을 찾을 수 없습니다."}
    return FileResponse(file_path, filename=filename)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    print("YouTube Script Extractor 웹 서버")
    print(f"  http://localhost:{port}")
    print()
    uvicorn.run(app, host="0.0.0.0", port=port)
