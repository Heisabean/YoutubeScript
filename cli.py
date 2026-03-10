#!/usr/bin/env python3
"""YouTube Script Extractor - CLI."""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from transcriber import process_video


def main():
    parser = argparse.ArgumentParser(
        description="YouTube 영상에서 음성인식으로 스크립트를 추출합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # Whisper API 사용 (빠르고 정확)
  python cli.py "https://www.youtube.com/watch?v=..." --mode api

  # 로컬 Whisper 사용 (무료)
  python cli.py "https://www.youtube.com/watch?v=..." --mode local --model base

  # SRT 자막만 생성
  python cli.py "https://www.youtube.com/watch?v=..." --format srt

  # 출력 디렉토리 지정
  python cli.py "https://www.youtube.com/watch?v=..." --output ./my_scripts
        """,
    )

    parser.add_argument("url", help="YouTube 영상 URL")
    parser.add_argument(
        "--mode",
        choices=["api", "local"],
        default=os.environ.get("DEFAULT_MODE", "api"),
        help="변환 모드 (기본: api)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API 키 (미지정 시 OPENAI_API_KEY 환경변수 사용)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DEFAULT_MODEL", "base"),
        choices=["tiny", "base", "small", "medium", "large"],
        help="로컬 Whisper 모델 크기 (기본: base)",
    )
    parser.add_argument(
        "--format",
        default="txt,srt",
        help="출력 형식, 쉼표로 구분 (기본: txt,srt)",
    )
    parser.add_argument(
        "--output",
        default="./output",
        help="출력 디렉토리 (기본: ./output)",
    )

    args = parser.parse_args()

    # API 키 처리
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if args.mode == "api" and not api_key:
        print("오류: API 모드에서는 OpenAI API 키가 필요합니다.")
        print("  --api-key 인자를 사용하거나 OPENAI_API_KEY 환경변수를 설정하세요.")
        sys.exit(1)

    formats = [f.strip() for f in args.format.split(",")]

    def on_progress(msg: str):
        print(f"  → {msg}")

    print(f"YouTube Script Extractor")
    print(f"  URL: {args.url}")
    print(f"  모드: {'Whisper API' if args.mode == 'api' else f'로컬 Whisper ({args.model})'}")
    print(f"  출력: {', '.join(formats)}")
    print()

    try:
        result = process_video(
            url=args.url,
            mode=args.mode,
            api_key=api_key,
            model_size=args.model,
            output_dir=args.output,
            formats=formats,
            on_progress=on_progress,
        )

        print()
        print(f"제목: {result['title']}")
        print(f"감지된 언어: {result['language']}")
        print(f"생성된 파일:")
        for fmt, path in result["files"].items():
            print(f"  [{fmt.upper()}] {path}")

    except KeyboardInterrupt:
        print("\n중단되었습니다.")
        sys.exit(130)
    except Exception as e:
        print(f"\n오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
