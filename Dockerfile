FROM python:3.12.8-slim-bookworm

RUN apt-get update -y \
    && apt-get install -y \
    pulseaudio \
    alsa-utils \
    ffmpeg \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR app/

CMD ["python", "app/app.py"]