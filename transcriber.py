"""YouTube 영상에서 음성을 추출하고 텍스트로 변환하는 핵심 모듈."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from typing import Callable, List, Optional
from pathlib import Path

from pytubefix import YouTube
from pydub import AudioSegment


def download_audio(
    url: str,
    output_dir: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """YouTube URL에서 오디오를 다운로드한다."""
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    def _notify(percent: int, detail: str):
        if on_progress:
            on_progress({"step": "download", "percent": percent, "detail": detail})

    _notify(0, "영상 정보를 가져오는 중...")

    # pytubefix 다운로드 진행 콜백
    def _download_cb(stream, chunk, bytes_remaining):
        total = stream.filesize
        downloaded = total - bytes_remaining
        pct = min(int(downloaded / total * 80), 80)  # 0~80%: 다운로드
        dl_mb = downloaded / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        _notify(pct, f"다운로드 중... {dl_mb:.1f}MB / {total_mb:.1f}MB")

    yt = YouTube(url, on_progress_callback=_download_cb)
    title = yt.title
    duration = yt.length

    _notify(5, f"'{title}' 오디오 스트림 선택 중...")

    audio_stream = yt.streams.filter(only_audio=True).order_by("abr").desc().first()
    if not audio_stream:
        raise RuntimeError("오디오 스트림을 찾을 수 없습니다.")

    # 다운로드 (m4a/webm)
    downloaded = audio_stream.download(output_path=output_dir)

    # mp3로 변환 (80~100%)
    _notify(85, "MP3로 변환 중...")
    mp3_path = os.path.join(output_dir, Path(downloaded).stem + ".mp3")
    audio = AudioSegment.from_file(downloaded)
    audio.export(mp3_path, format="mp3", bitrate="192k")

    if downloaded != mp3_path and os.path.exists(downloaded):
        os.remove(downloaded)

    file_mb = os.path.getsize(mp3_path) / (1024 * 1024)
    _notify(100, f"다운로드 완료 ({file_mb:.1f}MB, {duration}초)")

    return {"audio_path": mp3_path, "title": title, "duration": duration}


def download_video(
    url: str,
    output_dir: Optional[str] = None,
    on_progress: Optional[Callable] = None,
    max_resolution: int = 720,
) -> str:
    """키프레임 추출을 위해 YouTube 영상을 다운로드한다 (720p 이하)."""
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    def _notify(percent: int, detail: str):
        if on_progress:
            on_progress({"step": "download", "percent": percent, "detail": detail})

    _notify(50, "키프레임용 영상 다운로드 중...")

    def _download_cb(stream, chunk, bytes_remaining):
        total = stream.filesize
        downloaded = total - bytes_remaining
        pct = min(50 + int(downloaded / total * 45), 95)
        dl_mb = downloaded / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        _notify(pct, f"영상 다운로드 중... {dl_mb:.1f}MB / {total_mb:.1f}MB")

    yt = YouTube(url, on_progress_callback=_download_cb)

    # progressive 스트림 (오디오+비디오 합본, 작은 파일)
    stream = (
        yt.streams.filter(progressive=True, file_extension="mp4")
        .filter(res=f"{max_resolution}p")
        .first()
    )
    if not stream:
        # 해상도 제한 없이 가장 낮은 progressive 시도
        stream = (
            yt.streams.filter(progressive=True, file_extension="mp4")
            .order_by("resolution")
            .first()
        )
    if not stream:
        raise RuntimeError("영상 스트림을 찾을 수 없습니다.")

    video_path = stream.download(output_path=output_dir, filename_prefix="video_")
    _notify(98, "영상 다운로드 완료")
    return video_path


def extract_keyframes(
    video_path: str,
    output_dir: Optional[str] = None,
    method: str = "scene",
    interval_seconds: int = 30,
    max_frames: int = 50,
    on_progress: Optional[Callable] = None,
) -> List[dict]:
    """영상에서 키프레임을 추출한다 (OpenCV).

    method="scene": 히스토그램 비교로 장면 전환 감지.
    method="interval": N초 간격으로 추출.
    반환: [{"path": str, "timestamp": "MM:SS"}, ...]
    """
    import cv2

    def _notify(percent: int, detail: str):
        if on_progress:
            on_progress({"step": "keyframe_extract", "percent": percent, "detail": detail})

    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상 파일을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = total_frames / fps

    _notify(5, f"영상 분석 시작 ({total_seconds:.0f}초, {fps:.0f}fps)")

    keyframes = []
    prev_hist = None
    frame_idx = 0
    threshold = 0.6  # 장면 전환 감지 임계값

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_seconds = frame_idx / fps
        pct = min(int(5 + (frame_idx / total_frames) * 85), 90)

        should_save = False

        if method == "scene":
            # HSV 히스토그램 비교로 장면 전환 감지
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

            if prev_hist is not None:
                score = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if score < threshold:
                    should_save = True
            else:
                # 첫 프레임은 항상 저장
                should_save = True

            prev_hist = hist
        else:
            # interval 모드: N초 간격
            if frame_idx == 0 or (frame_idx % int(fps * interval_seconds) == 0):
                should_save = True

        if should_save and len(keyframes) < max_frames:
            mins = int(current_seconds // 60)
            secs = int(current_seconds % 60)
            timestamp = f"{mins:02d}:{secs:02d}"
            frame_path = os.path.join(output_dir, f"keyframe_{len(keyframes) + 1:03d}.jpg")
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            keyframes.append({"path": frame_path, "timestamp": timestamp})

            if len(keyframes) % 5 == 0:
                _notify(pct, f"{len(keyframes)}개 키프레임 추출됨 ({timestamp})")

        if len(keyframes) >= max_frames:
            break

        frame_idx += 1

    cap.release()
    _notify(100, f"키프레임 추출 완료: {len(keyframes)}개")
    return keyframes


def _split_audio(audio_path: str, max_size_mb: int = 24) -> List[str]:
    """오디오 파일을 Whisper API 제한(25MB)에 맞게 분할한다."""
    file_size = os.path.getsize(audio_path)
    max_bytes = max_size_mb * 1024 * 1024

    if file_size <= max_bytes:
        return [audio_path]

    audio = AudioSegment.from_mp3(audio_path)
    total_ms = len(audio)

    num_chunks = (file_size // max_bytes) + 1
    chunk_ms = total_ms // num_chunks

    chunks = []
    base = Path(audio_path)
    for i in range(num_chunks):
        start = i * chunk_ms
        end = min((i + 1) * chunk_ms, total_ms)
        chunk = audio[start:end]
        chunk_path = str(base.parent / f"{base.stem}_part{i}{base.suffix}")
        chunk.export(chunk_path, format="mp3")
        chunks.append(chunk_path)

    return chunks


def transcribe_api(
    audio_path: str,
    api_key: str,
    on_progress: Optional[Callable] = None,
) -> dict:
    """OpenAI Whisper API로 음성을 텍스트로 변환한다."""
    from openai import OpenAI

    def _notify(percent: int, detail: str):
        if on_progress:
            on_progress({"step": "transcribe", "percent": percent, "detail": detail})

    _notify(0, "오디오 파일 분석 중...")

    client = OpenAI(api_key=api_key)
    chunks = _split_audio(audio_path)
    total_chunks = len(chunks)
    all_segments = []
    full_text_parts = []
    detected_language = None
    time_offset = 0.0

    if total_chunks > 1:
        _notify(5, f"파일이 커서 {total_chunks}개로 분할하여 처리합니다")

    for i, chunk_path in enumerate(chunks):
        chunk_pct_start = int(i / total_chunks * 90)
        chunk_pct_end = int((i + 1) / total_chunks * 90)

        if total_chunks > 1:
            _notify(chunk_pct_start, f"청크 {i + 1}/{total_chunks} Whisper API 전송 중...")
        else:
            _notify(10, "Whisper API로 전송 중...")

        with open(chunk_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        if detected_language is None:
            detected_language = response.language

        full_text_parts.append(response.text)

        if total_chunks > 1:
            _notify(chunk_pct_end, f"청크 {i + 1}/{total_chunks} 완료")
        else:
            _notify(85, "응답 처리 중...")

        if response.segments:
            for seg in response.segments:
                _get = (lambda k: seg[k]) if isinstance(seg, dict) else (lambda k: getattr(seg, k))
                all_segments.append(
                    {
                        "start": _get("start") + time_offset,
                        "end": _get("end") + time_offset,
                        "text": _get("text"),
                    }
                )

        if chunks[-1] != chunk_path:
            audio = AudioSegment.from_mp3(chunk_path)
            time_offset += len(audio) / 1000.0

    # 분할된 임시 파일 정리
    for chunk_path in chunks:
        if chunk_path != audio_path and os.path.exists(chunk_path):
            os.remove(chunk_path)

    _notify(100, f"변환 완료 (언어: {detected_language}, 세그먼트: {len(all_segments)}개)")

    return {
        "text": " ".join(full_text_parts),
        "segments": all_segments,
        "language": detected_language or "unknown",
    }


def transcribe_local(
    audio_path: str,
    model_size: str = "base",
    on_progress: Optional[Callable] = None,
) -> dict:
    """로컬 Whisper 모델로 음성을 텍스트로 변환한다."""
    import whisper

    def _notify(percent: int, detail: str):
        if on_progress:
            on_progress({"step": "transcribe", "percent": percent, "detail": detail})

    _notify(0, f"Whisper {model_size} 모델 로딩 중...")
    model = whisper.load_model(model_size)

    _notify(20, "음성 분석 중... (영상 길이에 따라 시간이 걸립니다)")
    result = model.transcribe(audio_path)

    _notify(90, "결과 정리 중...")
    segments = []
    for seg in result.get("segments", []):
        segments.append(
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
        )

    lang = result.get("language", "unknown")
    _notify(100, f"변환 완료 (언어: {lang}, 세그먼트: {len(segments)}개)")

    return {
        "text": result["text"],
        "segments": segments,
        "language": lang,
    }


def _format_srt_time(seconds: float) -> str:
    """초를 SRT 타임스탬프 형식으로 변환한다."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def save_txt(result: dict, output_path: str) -> str:
    """텍스트 파일로 저장한다."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["text"].strip())
    return output_path


def save_srt(result: dict, output_path: str) -> str:
    """SRT 자막 파일로 저장한다."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result["segments"], 1):
            f.write(f"{i}\n")
            f.write(
                f"{_format_srt_time(seg['start'])} --> {_format_srt_time(seg['end'])}\n"
            )
            f.write(f"{seg['text'].strip()}\n\n")
    return output_path


def _sanitize_filename(title: str) -> str:
    """파일명에 사용할 수 없는 문자를 제거한다."""
    sanitized = re.sub(r'[<>:"/\\|?*]', "", title)
    sanitized = sanitized.strip(". ")
    return sanitized[:100] if sanitized else "untitled"


def process_video(
    url: str,
    mode: str = "api",
    api_key: Optional[str] = None,
    model_size: str = "base",
    output_dir: str = "./output",
    formats: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
    # LLM 옵션 (전역 기본값, 하위 호환)
    md_llm: Optional[str] = None,
    md_api_key: Optional[str] = None,
    md_ollama_model: str = "llama3.2",
    # 번역 옵션
    translate_lang: Optional[str] = None,
    # 단계별 LLM 오버라이드
    format_llm: Optional[str] = None,
    format_api_key: Optional[str] = None,
    translate_llm: Optional[str] = None,
    translate_api_key: Optional[str] = None,
    keyframe_llm: Optional[str] = None,
    keyframe_api_key: Optional[str] = None,
    # 키프레임 옵션
    enable_keyframes: bool = False,
    keyframe_method: str = "scene",
    keyframe_interval: int = 30,
    keyframe_max_frames: int = 50,
) -> dict:
    """전체 파이프라인: 다운로드 → (키프레임) → 변환 → (AI 정리) → (번역) → 저장."""
    from formatter import make_llm_config, format_as_markdown, translate_text, analyze_keyframes

    if formats is None:
        formats = ["txt", "srt"]

    os.makedirs(output_dir, exist_ok=True)

    def progress(data):
        if on_progress:
            on_progress(data)

    # 단계별 LLM 설정 빌드
    llm_cfg = make_llm_config(
        global_llm=md_llm, global_api_key=md_api_key,
        global_ollama_model=md_ollama_model,
        format_llm=format_llm, format_api_key=format_api_key,
        translate_llm=translate_llm, translate_api_key=translate_api_key,
        keyframe_llm=keyframe_llm, keyframe_api_key=keyframe_api_key,
    )

    # 1. 오디오 다운로드
    audio_info = download_audio(url, on_progress=on_progress)
    audio_path = audio_info["audio_path"]
    title = audio_info["title"]

    video_path = None
    keyframe_dir = None

    try:
        # 2. 키프레임용 영상 다운로드 (선택 시)
        if enable_keyframes:
            video_path = download_video(url, on_progress=on_progress)

        # 3. 키프레임 추출 (선택 시)
        keyframe_descriptions = None
        if enable_keyframes and video_path:
            keyframe_dir = tempfile.mkdtemp()
            keyframes = extract_keyframes(
                video_path, output_dir=keyframe_dir,
                method=keyframe_method,
                interval_seconds=keyframe_interval,
                max_frames=keyframe_max_frames,
                on_progress=on_progress,
            )

            # 4. 키프레임 Vision LLM 분석
            if keyframes:
                kf_cfg = llm_cfg["keyframe"]
                keyframe_descriptions = analyze_keyframes(
                    keyframe_paths=keyframes,
                    llm_provider=kf_cfg["llm"],
                    api_key=kf_cfg["api_key"],
                    on_progress=on_progress,
                )

        # 5. 음성 → 텍스트 변환
        if mode == "api":
            if not api_key:
                raise ValueError("API 모드에서는 OpenAI API 키가 필요합니다.")
            result = transcribe_api(audio_path, api_key, on_progress=on_progress)
        else:
            result = transcribe_local(audio_path, model_size, on_progress=on_progress)

        # 6. AI 마크다운 정리 (MD 선택 시)
        md_content = None
        fmt_cfg = llm_cfg["format"]
        if "md" in formats and fmt_cfg["llm"]:
            md_content = format_as_markdown(
                text=result["text"],
                title=title,
                llm_provider=fmt_cfg["llm"],
                api_key=fmt_cfg["api_key"],
                ollama_model=fmt_cfg["ollama_model"],
                on_progress=on_progress,
                keyframe_descriptions=keyframe_descriptions,
            )

        # 7. 번역 (선택 시)
        translated_txt = None
        translated_md = None
        tr_cfg = llm_cfg["translate"]
        if translate_lang and tr_cfg["llm"]:
            translated_txt = translate_text(
                text=result["text"],
                target_lang=translate_lang,
                llm_provider=tr_cfg["llm"],
                api_key=tr_cfg["api_key"],
                ollama_model=tr_cfg["ollama_model"],
                on_progress=on_progress,
            )

            if md_content:
                translated_md = translate_text(
                    text=md_content,
                    target_lang=translate_lang,
                    llm_provider=tr_cfg["llm"],
                    api_key=tr_cfg["api_key"],
                    ollama_model=tr_cfg["ollama_model"],
                    on_progress=on_progress,
                )

        # 8. 파일 저장
        progress({"step": "save", "percent": 0, "detail": "파일 저장 중..."})
        safe_title = _sanitize_filename(title)
        saved_files = {}

        if "txt" in formats:
            txt_path = os.path.join(output_dir, f"{safe_title}.txt")
            save_txt(result, txt_path)
            saved_files["txt"] = txt_path

        if "srt" in formats:
            srt_path = os.path.join(output_dir, f"{safe_title}.srt")
            save_srt(result, srt_path)
            saved_files["srt"] = srt_path

        if "md" in formats and md_content:
            md_path = os.path.join(output_dir, f"{safe_title}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            saved_files["md"] = md_path

        # 번역 파일 저장
        if translated_txt:
            lang_suffix = translate_lang.replace("-", "").lower()
            tr_txt_path = os.path.join(output_dir, f"{safe_title}_{lang_suffix}.txt")
            with open(tr_txt_path, "w", encoding="utf-8") as f:
                f.write(translated_txt)
            saved_files[f"txt_{lang_suffix}"] = tr_txt_path

        if translated_md:
            lang_suffix = translate_lang.replace("-", "").lower()
            tr_md_path = os.path.join(output_dir, f"{safe_title}_{lang_suffix}.md")
            with open(tr_md_path, "w", encoding="utf-8") as f:
                f.write(translated_md)
            saved_files[f"md_{lang_suffix}"] = tr_md_path

        fmt_str = ", ".join(f.upper() for f in saved_files.keys())
        progress({"step": "save", "percent": 100, "detail": f"{fmt_str} 파일 저장 완료"})

        return {
            "title": title,
            "language": result["language"],
            "text": result["text"],
            "segments": result["segments"],
            "files": saved_files,
        }
    finally:
        # 임시 파일 정리
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        if keyframe_dir and os.path.exists(keyframe_dir):
            shutil.rmtree(keyframe_dir, ignore_errors=True)
