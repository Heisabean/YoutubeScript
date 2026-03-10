---
title: YouTube Script Extractor
emoji: 🎬
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
---

# 🎬 YouTube Script Extractor

YouTube 영상에서 스크립트를 추출하고, AI로 깔끔하게 정리하는 도구입니다.

## ✨ 주요 기능

- **자동 스크립트 추출** — YouTube URL만 입력하면 음성을 텍스트로 변환
- **AI 마크다운 정리** — LLM이 스크립트를 구조화된 마크다운으로 정리
- **다국어 번역** — 14개 언어 지원 (한국어, 영어, 일본어 등)
- **키프레임 분석 (CV)** — 영상의 핵심 장면을 추출하고 Vision AI로 분석
- **Smart Mix** — 파이프라인 단계별로 다른 LLM을 지정 가능
- **웹 UI + CLI** — 브라우저 기반 UI와 커맨드라인 모두 지원

## 🤖 지원 LLM

| 모델 | 가격 | Vision | 비고 |
|------|------|--------|------|
| Ollama (로컬) | 무료 | ❌ | 인터넷 불필요 |
| Gemini 2.5 Flash-Lite | $0.10/1M토큰 | ✅ | ⭐ 최저가 |
| Gemini 3.1 Flash-Lite | $0.25/1M토큰 | ✅ | 최신 모델 |
| GPT-4o-mini | $0.15/1M토큰 | ✅ | OpenAI |
| Claude Sonnet 4.6 | $3/1M토큰 | ✅ | 고품질 |
| Claude Opus 4.6 | $5/1M토큰 | ✅ | 최고 품질 |

## 🚀 빠른 시작

### 1. 설치

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/youtube-script-extractor
cd youtube-script-extractor

# ffmpeg 설치 (macOS)
brew install ffmpeg

# Python 의존성 설치
pip install -r requirements.txt
```

### 2. API 키 설정

```bash
cp .env.example .env
# .env 파일을 열어 사용할 LLM의 API 키를 입력하세요
```

### 3. 실행

```bash
# 웹 UI (추천)
chmod +x run
./run

# 또는 직접 실행
python3 web.py
```

브라우저에서 `http://localhost:8000` 접속

### CLI 모드

```bash
./run "https://www.youtube.com/watch?v=VIDEO_ID"
./run "https://www.youtube.com/watch?v=VIDEO_ID" --mode local --format srt
```

## ⚙️ 설정

`.env` 파일에서 설정합니다:

```env
# 필수: Whisper API용
OPENAI_API_KEY=sk-proj-...

# 선택: 사용하는 LLM만 설정
GOOGLE_API_KEY=...        # Gemini
ANTHROPIC_API_KEY=...     # Claude

# 기본 설정
DEFAULT_MODE=api          # api 또는 local
DEFAULT_MODEL=base        # Whisper 모델 (tiny/base/small/medium/large)
PORT=8000
```

## 📁 프로젝트 구조

```
├── run                 # 실행 스크립트 (설치 + 실행)
├── web.py              # FastAPI 웹 서버
├── cli.py              # CLI 인터페이스
├── transcriber.py      # 음성 추출 + STT + 키프레임 추출
├── formatter.py        # LLM 호출 + 마크다운 정리 + 번역
├── templates/
│   └── index.html      # 웹 UI
├── requirements.txt    # Python 의존성
├── .env.example        # 환경 변수 템플릿
└── output/             # 추출 결과물 (git 제외)
```

## 🔧 Smart Mix (단계별 LLM 선택)

웹 UI의 ⚙ 버튼으로 파이프라인 단계별 LLM을 설정할 수 있습니다:

- **MD 정리** — 스크립트를 마크다운으로 구조화 (저렴한 모델 OK)
- **번역** — 다국어 번역 (저렴한 모델 OK)
- **키프레임 분석** — Vision AI로 화면 분석 (Vision 지원 모델 필요)

## 📋 요구사항

- Python 3.9+
- ffmpeg
- API 키 (사용하려는 LLM에 따라)

## 📄 License

MIT
