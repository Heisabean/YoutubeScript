FROM python:3.11-slim

# ffmpeg 설치 (whisper, pydub에 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/output

EXPOSE 7860

CMD ["uvicorn", "web:app", "--host", "0.0.0.0", "--port", "7860"]
