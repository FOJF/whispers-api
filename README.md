# WhisperX + pyannote.audio 화자분리 전사 API 서버

WhisperX와 pyannote.audio를 활용한 고성능 화자분리 전사 REST API 서버입니다.

## 🚀 주요 기능

- **고속 음성 인식**: WhisperX로 70배 실시간 속도 전사
- **화자분리**: pyannote.audio로 화자별 구분
- **정확한 타임스탬프**: 단어/문장 레벨 타임스탬프
- **다국어 지원**: 자동 언어 감지 및 다국어 전사
- **GPU 가속**: CUDA 지원으로 빠른 처리
- **Docker 지원**: 간편한 배포 및 실행

## 📋 요구사항

### 시스템 요구사항
- Python 3.10+
- CUDA 11.8+ (GPU 사용 시)
- Docker & Docker Compose (권장)

### 하드웨어 요구사항
- **GPU**: NVIDIA GPU (8GB+ VRAM 권장)
- **CPU**: 8코어 이상 (CPU 전용 모드)
- **RAM**: 16GB 이상
- **저장공간**: 20GB 이상 (모델 다운로드용)

## 🛠️ 설치 및 실행

### 1. 저장소 클론
```bash
git clone <repository-url>
cd whisperx-api
```

### 2. 환경 변수 설정
```bash
cp env.example .env
# .env 파일을 편집하여 HuggingFace 토큰 설정
```

### 3. HuggingFace 토큰 설정
1. [HuggingFace](https://huggingface.co/) 계정 생성
2. [Access Tokens](https://huggingface.co/settings/tokens)에서 토큰 생성
3. `.env` 파일에 토큰 설정:
```bash
HUGGINGFACE_TOKEN=your_token_here
```

### 4. Docker로 실행 (권장)

#### GPU 버전
```bash
# GPU 지원 Docker Compose 실행
docker-compose up -d
```

#### CPU 전용 버전
```bash
# CPU 전용 Docker Compose 실행
docker-compose -f docker-compose.cpu.yml up -d
```

### 5. 로컬 개발 환경 실행

#### 가상환경 생성
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

#### 의존성 설치
```bash
pip install -r requirements.txt
```

#### 애플리케이션 실행
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📖 API 사용법

### 기본 엔드포인트

#### 1. 헬스 체크
```bash
GET /health
```

#### 2. 오디오 전사
```bash
POST /v1/audio/transcribe
Content-Type: multipart/form-data

# 파일 업로드
curl -X POST http://localhost:8000/v1/audio/transcribe \
  -F "file=@audio.mp3"
```

#### 3. 모델 정보 조회
```bash
GET /v1/models/info
```

### 응답 형식

#### 성공 응답
```json
{
  "status": "success",
  "transcription": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.5,
      "end": 3.2,
      "text": "안녕하세요, 오늘 회의를 시작하겠습니다.",
      "confidence": 0.95
    },
    {
      "speaker": "SPEAKER_01",
      "start": 3.5,
      "end": 6.1,
      "text": "네, 좋습니다. 준비가 다 되었습니다.",
      "confidence": 0.92
    }
  ],
  "processing_time": 15.2,
  "language": "ko",
  "total_duration": 120.5,
  "speakers_count": 2
}
```

#### 에러 응답
```json
{
  "status": "error",
  "error": "파일 크기가 너무 큽니다",
  "detail": "최대 500MB까지 지원됩니다",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 🔧 설정 옵션

### 환경 변수

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `HOST` | `0.0.0.0` | 서버 호스트 |
| `PORT` | `8000` | 서버 포트 |
| `WHISPER_MODEL` | `large-v2` | Whisper 모델 |
| `DEVICE` | `cuda` | 디바이스 (cuda/cpu) |
| `COMPUTE_TYPE` | `float16` | 계산 타입 |
| `LANGUAGE` | `ko` | 언어 설정 |
| `MAX_FILE_SIZE` | `500` | 최대 파일 크기 (MB) |
| `HUGGINGFACE_TOKEN` | - | HuggingFace 토큰 |

### 지원 파일 형식
- **오디오**: MP3, WAV, FLAC, AAC, OGG, WMA
- **비디오**: MP4, AVI, MOV, MKV

## 🚀 성능 최적화

### GPU 사용 시
```bash
# CUDA 설정
export CUDA_VISIBLE_DEVICES=0
export DEVICE=cuda
export COMPUTE_TYPE=float16
```

### CPU 사용 시
```bash
# CPU 최적화 설정
export DEVICE=cpu
export COMPUTE_TYPE=int8
```

### 메모리 최적화
```bash
# 모델 캐시 디렉토리 설정
export MODEL_CACHE_DIR=/app/models
```

## 🔗 Spring Boot 연동

Spring Boot 애플리케이션과의 연동 예시는 `examples/spring-boot/` 디렉토리를 참조하세요.

### 주요 클래스
- `WhisperXClient`: API 클라이언트
- `TranscriptionController`: REST 컨트롤러
- `WhisperXConfig`: 설정 클래스

### 사용 예시
```java
@Autowired
private WhisperXClient whisperXClient;

public TranscriptionResponse transcribe(MultipartFile file) {
    return whisperXClient.transcribeAudio(file);
}
```

## 🐛 문제 해결

### 일반적인 문제

#### 1. HuggingFace 토큰 오류
```
Error: HuggingFace token is required
```
**해결방법**: `.env` 파일에 올바른 토큰 설정

#### 2. CUDA 메모리 부족
```
Error: CUDA out of memory
```
**해결방법**: 
- 더 작은 모델 사용 (`medium` 또는 `small`)
- `COMPUTE_TYPE=int8` 설정
- CPU 모드 사용

#### 3. 모델 다운로드 실패
```
Error: Model download failed
```
**해결방법**:
- 인터넷 연결 확인
- HuggingFace 토큰 확인
- 수동으로 모델 다운로드

### 로그 확인
```bash
# Docker 로그 확인
docker-compose logs -f whisperx-api

# 로컬 실행 시 로그 레벨 설정
export LOG_LEVEL=DEBUG
```

## 📊 모니터링

### 헬스 체크
```bash
curl http://localhost:8000/health
```

### 모델 상태 확인
```bash
curl http://localhost:8000/v1/models/info
```

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🙏 감사의 말

- [WhisperX](https://github.com/m-bain/whisperX) - 고속 음성 인식
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - 화자분리
- [FastAPI](https://fastapi.tiangolo.com/) - 웹 프레임워크
- [HuggingFace](https://huggingface.co/) - 모델 호스팅

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해 주세요.

---

**주의**: 이 서버는 대용량 파일 처리와 GPU 연산을 수행하므로 충분한 하드웨어 리소스가 필요합니다.
