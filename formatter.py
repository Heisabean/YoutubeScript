"""LLM을 사용하여 Whisper 추출 텍스트를 마크다운 정리/번역하는 모듈."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from typing import Callable, List, Optional

# ── LLM 정보 (가격순 정렬) ──────────────────────────────────
LLM_MODELS = {
    "ollama": {
        "name": "Ollama (로컬, 무료)",
        "model": "llama3.2",
        "version": "llama3.2",
        "price_rank": 0,
        "quality_rank": 5,
        "needs_key": False,
        "env_key": None,
        "supports_vision": False,
        "description": "로컬 실행, 무료, 인터넷 불필요",
    },
    "gemini-flash-lite": {
        "name": "Gemini 2.5 Flash-Lite",
        "model": "gemini-2.5-flash-lite",
        "version": "2.5-flash-lite",
        "price_rank": 1,
        "quality_rank": 4,
        "needs_key": True,
        "env_key": "GOOGLE_API_KEY",
        "supports_vision": True,
        "description": "⭐ 최저가, $0.10/1M토큰, 빠르고 안정적",
    },
    "gemini-flash": {
        "name": "Gemini 3.1 Flash-Lite (Preview)",
        "model": "gemini-3.1-flash-lite-preview",
        "version": "3.1-flash-lite",
        "price_rank": 2,
        "quality_rank": 3,
        "needs_key": True,
        "env_key": "GOOGLE_API_KEY",
        "supports_vision": True,
        "description": "최신 모델, $0.25/1M토큰, 향상된 성능",
    },
    "gpt-4o-mini": {
        "name": "GPT-4o-mini (OpenAI)",
        "model": "gpt-4o-mini",
        "version": "4o-mini",
        "price_rank": 3,
        "quality_rank": 2,
        "needs_key": True,
        "env_key": "OPENAI_API_KEY",
        "supports_vision": True,
        "description": "빠르고 저렴, $0.15/1M토큰",
    },
    "claude-sonnet": {
        "name": "Claude Sonnet 4.6 (Anthropic)",
        "model": "claude-sonnet-4-6",
        "version": "sonnet-4.6",
        "price_rank": 4,
        "quality_rank": 1,
        "needs_key": True,
        "env_key": "ANTHROPIC_API_KEY",
        "supports_vision": True,
        "description": "Anthropic, 빠르고 고품질 ($3/1M토큰)",
    },
    "claude-opus": {
        "name": "Claude Opus 4.6 (Anthropic)",
        "model": "claude-opus-4-6",
        "version": "opus-4.6",
        "price_rank": 5,
        "quality_rank": 0,
        "needs_key": True,
        "env_key": "ANTHROPIC_API_KEY",
        "supports_vision": True,
        "description": "Anthropic 최고 모델, 구조화 능력 최고 ($5/1M토큰)",
    },
}

# ── 번역 지원 언어 ────────────────────────────────────────
LANGUAGES = {
    "ko": "한국어",
    "en": "English",
    "ja": "日本語",
    "zh-CN": "中文(简体)",
    "zh-TW": "中文(繁體)",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "pt": "Português",
    "ru": "Русский",
    "vi": "Tiếng Việt",
    "th": "ภาษาไทย",
    "ar": "العربية",
    "hi": "हिन्दी",
}


def get_models_sorted(sort_by: str = "price") -> list:
    """정렬된 LLM 모델 리스트를 반환한다."""
    key = "price_rank" if sort_by == "price" else "quality_rank"
    return sorted(
        [{"id": k, **v} for k, v in LLM_MODELS.items()],
        key=lambda x: x[key],
    )


def get_languages() -> list:
    """지원되는 번역 언어 목록을 반환한다."""
    return [{"code": k, "name": v} for k, v in LANGUAGES.items()]


def make_llm_config(
    global_llm: Optional[str] = None,
    global_api_key: Optional[str] = None,
    global_ollama_model: str = "llama3.2",
    format_llm: Optional[str] = None,
    format_api_key: Optional[str] = None,
    translate_llm: Optional[str] = None,
    translate_api_key: Optional[str] = None,
    keyframe_llm: Optional[str] = None,
    keyframe_api_key: Optional[str] = None,
) -> dict:
    """단계별 LLM 설정 딕셔너리를 생성한다.

    per-step 값이 없으면 global 값으로 fallback.
    반환: {"format": {...}, "translate": {...}, "keyframe": {...}}
    """
    def _resolve(step_llm, step_key):
        llm = step_llm or global_llm
        api_key = step_key or global_api_key
        return {
            "llm": llm,
            "api_key": api_key,
            "ollama_model": global_ollama_model,
        }

    return {
        "format": _resolve(format_llm, format_api_key),
        "translate": _resolve(translate_llm, translate_api_key),
        "keyframe": _resolve(
            keyframe_llm or "gemini-flash-lite",
            keyframe_api_key,
        ),
    }


# ── 프롬프트 ──────────────────────────────────────────────

FORMAT_SYSTEM_PROMPT = """You are an expert document formatter. Your task is to transform raw speech-to-text transcriptions into clean, well-structured Markdown documents.

Rules:
- Detect the content's language and write the output in the SAME language
- Add a clear title as # heading
- Write a brief 2-3 sentence summary at the top
- Divide content into logical sections with ## headings
- Clean up filler words, repetitions, and stutters
- Fix obvious grammar/punctuation errors
- Keep the original meaning and tone intact
- Use bullet points or numbered lists where appropriate
- Add --- horizontal rules between major sections
- Do NOT add information that wasn't in the original text
- Output ONLY the formatted Markdown, no explanations"""

FORMAT_USER_TEMPLATE = """Here is a raw speech-to-text transcription from a YouTube video titled "{title}".
Please format it into a clean, readable Markdown document.

---
{text}
---"""

FORMAT_SYSTEM_PROMPT_WITH_KEYFRAMES = """You are an expert document formatter. Your task is to transform raw speech-to-text transcriptions into clean, well-structured Markdown documents, enhanced with visual context from video keyframes.

Rules:
- Detect the content's language and write the output in the SAME language
- Add a clear title as # heading
- Write a brief 2-3 sentence summary at the top
- Divide content into logical sections with ## headings
- Clean up filler words, repetitions, and stutters
- Fix obvious grammar/punctuation errors
- Keep the original meaning and tone intact
- Use bullet points or numbered lists where appropriate
- Add --- horizontal rules between major sections
- KEYFRAME CONTEXT: You are also given timestamped descriptions of visual keyframes.
  Insert relevant visual descriptions as blockquotes (> 🖼 [timestamp] description) at
  appropriate positions in the transcript where they add context.
- Only include keyframe descriptions that add meaningful value (skip redundant ones)
- Do NOT add information that wasn't in the original text or keyframes
- Output ONLY the formatted Markdown, no explanations"""

FORMAT_USER_TEMPLATE_WITH_KEYFRAMES = """Here is a raw speech-to-text transcription from a YouTube video titled "{title}".
Please format it into a clean, readable Markdown document.

---
TRANSCRIPT:
{text}
---

VISUAL KEYFRAME DESCRIPTIONS (timestamped):
{keyframe_descriptions}
---"""

KEYFRAME_ANALYSIS_SYSTEM_PROMPT = """You are a visual content analyst. Analyze the provided video keyframes and describe what is shown.

Rules:
- Describe each frame concisely (1-3 sentences)
- Include any visible text (OCR) exactly as shown
- Note visual elements: diagrams, charts, code, slides, people, scenes
- Focus on informational content, not aesthetic quality
- For each frame, output one line in the format: [MM:SS] description
- Keep descriptions factual and relevant to the video content
- Output ONLY the descriptions, no extra commentary"""

KEYFRAME_ANALYSIS_USER_TEMPLATE = """Analyze these keyframes from a video. For each image, describe what is shown and extract any visible text.
The timestamps for each frame are provided as labels."""

TRANSLATE_SYSTEM_PROMPT = """You are a professional translator. Translate the given text accurately into {target_lang}.

Rules:
- Maintain the original meaning, tone, and nuance
- If the text uses Markdown formatting, preserve the Markdown structure
- Translate naturally and idiomatically, not word-by-word
- Keep proper nouns, brand names, and technical terms appropriately
- Do NOT add explanations, notes, or commentary
- Output ONLY the translated text"""

TRANSLATE_USER_TEMPLATE = """Translate the following text into {target_lang}:

---
{text}
---"""


# ── LLM 호출 함수들 (범용) ────────────────────────────────

def _call_openai(system_prompt: str, user_prompt: str, api_key: str) -> str:
    """OpenAI GPT-4o-mini 호출."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


def _call_gemini(system_prompt: str, user_prompt: str, api_key: str,
                  model: str = "gemini-2.5-flash-lite") -> str:
    """Google Gemini 호출."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gmodel = genai.GenerativeModel(model)

    prompt = system_prompt + "\n\n" + user_prompt
    response = gmodel.generate_content(prompt)
    return response.text


def _call_ollama(system_prompt: str, user_prompt: str, model_name: str = "llama3.2") -> str:
    """Ollama 로컬 모델 호출."""
    import urllib.request

    payload = json.dumps({
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["message"]["content"]
    except Exception as e:
        if "Connection refused" in str(e):
            raise RuntimeError(
                "Ollama가 실행 중이 아닙니다. "
                "'ollama serve' 명령으로 먼저 시작해주세요. "
                "(설치: brew install ollama && ollama pull llama3.2)"
            )
        raise


def _call_claude(system_prompt: str, user_prompt: str, api_key: str, model: str = "claude-sonnet-4-6") -> str:
    """Anthropic Claude 호출."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.content[0].text


# ── Vision LLM 호출 함수들 ───────────────────────────────

def _call_gemini_vision(
    system_prompt: str,
    user_prompt: str,
    images: List[dict],
    api_key: str,
    model: str = "gemini-2.5-flash-lite",
) -> str:
    """Google Gemini Vision 호출."""
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=api_key)
    gmodel = genai.GenerativeModel(model)

    parts = [system_prompt + "\n\n" + user_prompt]
    for img_info in images:
        img = Image.open(img_info["path"])
        parts.append(img)
        parts.append(f"[Timestamp: {img_info['timestamp']}]")

    response = gmodel.generate_content(parts)
    return response.text


def _call_openai_vision(
    system_prompt: str,
    user_prompt: str,
    images: List[dict],
    api_key: str,
) -> str:
    """OpenAI GPT-4o-mini Vision 호출."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    content = [{"type": "text", "text": user_prompt}]
    for img_info in images:
        with open(img_info["path"], "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
        content.append({"type": "text", "text": f"[Timestamp: {img_info['timestamp']}]"})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


def _call_claude_vision(
    system_prompt: str,
    user_prompt: str,
    images: List[dict],
    api_key: str,
    model: str = "claude-sonnet-4-6",
) -> str:
    """Anthropic Claude Vision 호출."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    content = []
    for img_info in images:
        with open(img_info["path"], "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
        content.append({"type": "text", "text": f"[Timestamp: {img_info['timestamp']}]"})
    content.append({"type": "text", "text": user_prompt})

    response = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text


def _call_vision_llm(
    system_prompt: str,
    user_prompt: str,
    images: List[dict],
    llm_provider: str,
    api_key: Optional[str] = None,
) -> str:
    """Vision LLM 호출 디스패처 (Rate limit 자동 재시도 포함)."""
    def _do_call():
        if llm_provider in ("gemini-flash", "gemini-flash-lite"):
            key = api_key or os.environ.get("GOOGLE_API_KEY", "")
            if not key:
                raise ValueError("Google API 키가 필요합니다. (GOOGLE_API_KEY)")
            model_id = LLM_MODELS[llm_provider]["model"]
            return _call_gemini_vision(system_prompt, user_prompt, images, key, model=model_id)

        elif llm_provider == "gpt-4o-mini":
            key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if not key:
                raise ValueError("OpenAI API 키가 필요합니다.")
            return _call_openai_vision(system_prompt, user_prompt, images, key)

        elif llm_provider in ("claude-sonnet", "claude-opus"):
            key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                raise ValueError("Anthropic API 키가 필요합니다. (ANTHROPIC_API_KEY)")
            model_id = LLM_MODELS[llm_provider]["model"]
            return _call_claude_vision(system_prompt, user_prompt, images, key, model=model_id)

        elif llm_provider == "ollama":
            raise ValueError("Ollama는 Vision(이미지 분석)을 지원하지 않습니다.")

        else:
            raise ValueError(f"지원하지 않는 Vision LLM: {llm_provider}")

    return _retry_on_rate_limit(_do_call)


# ── Rate Limit 재시도 로직 ────────────────────────────────

def _is_rate_limit_error(error: Exception) -> bool:
    """429 Rate Limit 에러인지 확인한다."""
    err_str = str(error).lower()
    err_type = type(error).__name__
    return (
        "rate_limit" in err_str
        or "rate limit" in err_str
        or "429" in err_str
        or "resource_exhausted" in err_str
        or "quota" in err_str
        or err_type == "RateLimitError"
    )


def _parse_retry_after(error: Exception) -> float:
    """에러 메시지에서 대기 시간(초)을 추출한다."""
    err_str = str(error)
    # "Please try again in 2.129s" 같은 패턴
    match = re.search(r"try again in (\d+\.?\d*)s", err_str)
    if match:
        return float(match.group(1))
    # "Retry-After: 5" 헤더 패턴
    match = re.search(r"retry.?after:?\s*(\d+)", err_str, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0


def _retry_on_rate_limit(func, *args, max_retries: int = 3, **kwargs):
    """Rate limit 에러 시 exponential backoff으로 재시도한다."""
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if not _is_rate_limit_error(e) or attempt >= max_retries:
                raise
            # 에러에서 대기 시간 추출, 없으면 exponential backoff
            wait = _parse_retry_after(e)
            if wait <= 0:
                wait = (2 ** attempt) * 2  # 2초, 4초, 8초
            wait = min(wait + 0.5, 60)  # 여유 0.5초 추가, 최대 60초
            print(f"⏳ Rate limit 초과, {wait:.1f}초 후 재시도... ({attempt + 1}/{max_retries})")
            time.sleep(wait)


# ── 텍스트 LLM 호출 디스패처 ─────────────────────────────

def _call_llm(
    system_prompt: str,
    user_prompt: str,
    llm_provider: str,
    api_key: Optional[str] = None,
    ollama_model: str = "llama3.2",
) -> str:
    """텍스트 LLM 호출 디스패처 (Rate limit 자동 재시도 포함)."""
    def _do_call():
        if llm_provider == "gpt-4o-mini":
            key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if not key:
                raise ValueError("OpenAI API 키가 필요합니다.")
            return _call_openai(system_prompt, user_prompt, key)

        elif llm_provider in ("gemini-flash", "gemini-flash-lite"):
            key = api_key or os.environ.get("GOOGLE_API_KEY", "")
            if not key:
                raise ValueError("Google API 키가 필요합니다. (GOOGLE_API_KEY)")
            model_id = LLM_MODELS[llm_provider]["model"]
            return _call_gemini(system_prompt, user_prompt, key, model=model_id)

        elif llm_provider == "ollama":
            return _call_ollama(system_prompt, user_prompt, ollama_model)

        elif llm_provider in ("claude-sonnet", "claude-opus"):
            key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                raise ValueError("Anthropic API 키가 필요합니다. (ANTHROPIC_API_KEY)")
            model_id = LLM_MODELS[llm_provider]["model"]
            return _call_claude(system_prompt, user_prompt, key, model=model_id)

        else:
            raise ValueError(f"지원하지 않는 LLM: {llm_provider}")

    return _retry_on_rate_limit(_do_call)


# ── 키프레임 분석 ─────────────────────────────────────────

def analyze_keyframes(
    keyframe_paths: List[dict],
    llm_provider: str = "gemini-flash-lite",
    api_key: Optional[str] = None,
    on_progress: Optional[Callable] = None,
    batch_size: int = 10,
) -> str:
    """Vision LLM으로 키프레임 이미지를 분석한다.

    keyframe_paths: [{"path": str, "timestamp": str}, ...]
    반환: 타임스탬프별 설명 텍스트
    """
    def _notify(percent: int, detail: str):
        if on_progress:
            on_progress({"step": "keyframe_analysis", "percent": percent, "detail": detail})

    model_info = LLM_MODELS.get(llm_provider, {})
    model_name = model_info.get("name", llm_provider)

    total = len(keyframe_paths)
    _notify(5, f"{total}개 키프레임을 {model_name}으로 분석 준비 중...")

    all_descriptions = []
    batches = [keyframe_paths[i:i + batch_size] for i in range(0, total, batch_size)]

    for idx, batch in enumerate(batches):
        pct = int(10 + (idx / len(batches)) * 80)
        _notify(pct, f"배치 {idx + 1}/{len(batches)} 분석 중 ({len(batch)}프레임)...")

        try:
            result = _call_vision_llm(
                system_prompt=KEYFRAME_ANALYSIS_SYSTEM_PROMPT,
                user_prompt=KEYFRAME_ANALYSIS_USER_TEMPLATE,
                images=batch,
                llm_provider=llm_provider,
                api_key=api_key,
            )
            all_descriptions.append(result.strip())
        except Exception as e:
            _notify(pct, f"배치 {idx + 1} 분석 오류: {str(e)}")
            raise

    _notify(100, f"{total}개 키프레임 분석 완료")
    return "\n".join(all_descriptions)


# ── 메인 함수들 ───────────────────────────────────────────

def _truncate_text(text: str, max_chars: int = 100000) -> tuple:
    """텍스트가 너무 길면 잘라낸다. (text, truncated) 반환."""
    if len(text) > max_chars:
        return text[:max_chars], True
    return text, False


def format_as_markdown(
    text: str,
    title: str,
    llm_provider: str = "gemini-flash-lite",
    api_key: Optional[str] = None,
    ollama_model: str = "llama3.2",
    on_progress: Optional[Callable] = None,
    keyframe_descriptions: Optional[str] = None,
) -> str:
    """Whisper 추출 텍스트를 LLM으로 마크다운으로 정리한다.

    keyframe_descriptions가 주어지면 키프레임 설명을 MD에 통합한다.
    """
    def _notify(percent: int, detail: str):
        if on_progress:
            on_progress({"step": "format", "percent": percent, "detail": detail})

    model_info = LLM_MODELS.get(llm_provider, {})
    model_name = model_info.get("name", llm_provider)

    has_keyframes = bool(keyframe_descriptions and keyframe_descriptions.strip())
    extra = " + 키프레임 컨텍스트" if has_keyframes else ""
    _notify(10, f"{model_name}에 텍스트{extra} 전송 중...")

    text, truncated = _truncate_text(text)
    if truncated:
        _notify(15, f"텍스트가 길어서 앞부분만 정리합니다 ({len(text)}자)")

    _notify(30, f"{model_name} 처리 중...")

    # 키프레임 설명이 있으면 통합 프롬프트 사용
    if has_keyframes:
        sys_prompt = FORMAT_SYSTEM_PROMPT_WITH_KEYFRAMES
        usr_prompt = FORMAT_USER_TEMPLATE_WITH_KEYFRAMES.format(
            title=title, text=text, keyframe_descriptions=keyframe_descriptions,
        )
    else:
        sys_prompt = FORMAT_SYSTEM_PROMPT
        usr_prompt = FORMAT_USER_TEMPLATE.format(title=title, text=text)

    try:
        result = _call_llm(
            system_prompt=sys_prompt,
            user_prompt=usr_prompt,
            llm_provider=llm_provider,
            api_key=api_key,
            ollama_model=ollama_model,
        )
    except Exception as e:
        _notify(0, f"LLM 오류: {str(e)}")
        raise

    if truncated:
        result += "\n\n---\n> ⚠️ 원본 텍스트가 길어서 일부만 정리되었습니다.\n"

    _notify(100, f"{model_name} 정리 완료")
    return result


def translate_text(
    text: str,
    target_lang: str,
    llm_provider: str = "gemini-flash-lite",
    api_key: Optional[str] = None,
    ollama_model: str = "llama3.2",
    on_progress: Optional[Callable] = None,
) -> str:
    """텍스트를 지정 언어로 번역한다."""
    def _notify(percent: int, detail: str):
        if on_progress:
            on_progress({"step": "translate", "percent": percent, "detail": detail})

    lang_name = LANGUAGES.get(target_lang, target_lang)
    model_info = LLM_MODELS.get(llm_provider, {})
    model_name = model_info.get("name", llm_provider)

    _notify(10, f"{lang_name}로 번역 준비 중...")

    text, truncated = _truncate_text(text)
    if truncated:
        _notify(15, f"텍스트가 길어서 앞부분만 번역합니다 ({len(text)}자)")

    _notify(30, f"{model_name}으로 {lang_name} 번역 중...")

    try:
        result = _call_llm(
            system_prompt=TRANSLATE_SYSTEM_PROMPT.format(target_lang=lang_name),
            user_prompt=TRANSLATE_USER_TEMPLATE.format(target_lang=lang_name, text=text),
            llm_provider=llm_provider,
            api_key=api_key,
            ollama_model=ollama_model,
        )
    except Exception as e:
        _notify(0, f"번역 오류: {str(e)}")
        raise

    if truncated:
        result += f"\n\n---\n> ⚠️ 원본 텍스트가 길어서 일부만 번역되었습니다.\n"

    _notify(100, f"{lang_name} 번역 완료")
    return result
