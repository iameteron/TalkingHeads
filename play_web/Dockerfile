FROM python:3.10-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAY_WEB_PERSIST_KEYS_TO_ENV=false \
    PLAY_WEB_LEADERBOARD_DIR=/app/data

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt setup.py ./
COPY oracle/ oracle/
COPY config/ config/
COPY MegaPrompt/ MegaPrompt/
COPY play_web/ play_web/

# CPU-only torch (no CUDA) — VM has no GPU; saves ~5 GB in the image.
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && grep -vE '^(torch|#|$)' requirements.txt > /tmp/requirements-cpu.txt \
    && pip install -r /tmp/requirements-cpu.txt \
    && pip install -e .

RUN chmod +x /app/play_web/docker/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/play_web/docker/entrypoint.sh"]
