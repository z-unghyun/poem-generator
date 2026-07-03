FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PORT=7860
ENV MODEL_ID=z-unghyun/poem-generator-merged
ENV BASE_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
ENV ADAPTER_ID=z-unghyun/poem-generator-lora-adapter
ENV USE_ADAPTER=false
ENV ENABLE_ADAPTER_FALLBACK=true
ENV NEWLINE_STOP=3
ENV MAX_NEW_TOKENS=60

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY app.py /app/app.py

EXPOSE 7860

CMD ["python", "app.py"]
