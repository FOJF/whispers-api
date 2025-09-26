# EC2 CPU 최적화 Dockerfile
FROM ubuntu:22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# 시스템 패키지 업데이트 및 필수 패키지 설치 (EC2 최적화)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    ffmpeg \
    git \
    wget \
    curl \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python 심볼릭 링크 생성
RUN ln -s /usr/bin/python3 /usr/bin/python

# 작업 디렉토리 설정
WORKDIR /app

# Python 의존성 설치 (EC2 최적화)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY app/ ./app/

# 모델 캐시 디렉토리 생성 (EC2 최적화)
RUN mkdir -p /app/models
RUN mkdir -p /tmp/whisperx_uploads
RUN chmod 755 /app/models
RUN chmod 755 /tmp/whisperx_uploads

# 포트 노출
EXPOSE 8000

# EC2 환경을 위한 헬스체크
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# EC2 최적화된 애플리케이션 실행
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
