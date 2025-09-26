"""
설정 파일
"""
import os
from typing import Optional

class Settings:
    """애플리케이션 설정"""
    
    # API 설정
    API_V1_STR: str = "/v1"
    PROJECT_NAME: str = "WhisperX API"
    VERSION: str = "1.0.0"
    
    # 서버 설정
    HOST: str = os.environ.get("HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("PORT", "8000"))
    
    # 파일 업로드 설정
    MAX_FILE_SIZE: int = int(os.environ.get("MAX_FILE_SIZE", "500")) * 1024 * 1024  # 500MB
    UPLOAD_DIR: str = os.environ.get("UPLOAD_DIR", "/tmp/whisperx_uploads")
    ALLOWED_EXTENSIONS: set = {
        ".mp3", ".wav", ".mp4", ".m4a", ".flac", ".aac", 
        ".ogg", ".wma", ".avi", ".mov", ".mkv"
    }
    
    # WhisperX 설정
    WHISPER_MODEL: str = os.environ.get("WHISPER_MODEL", "medium")
    DEVICE: str = os.environ.get("DEVICE", "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    COMPUTE_TYPE: str = os.environ.get("COMPUTE_TYPE", "float16" if os.environ.get("CUDA_VISIBLE_DEVICES") else "int8")
    
    # pyannote 설정
    HUGGINGFACE_TOKEN: Optional[str] = os.environ.get("HUGGINGFACE_TOKEN")
    DIARIZATION_MODEL: str = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
    
    # 언어 설정
    LANGUAGE: Optional[str] = os.environ.get("LANGUAGE", "ko")  # None이면 자동 감지
    
    # 로깅 설정
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
    
    # 모델 캐시 설정
    MODEL_CACHE_DIR: str = os.environ.get("MODEL_CACHE_DIR", "/app/models")
    
    def __init__(self):
        # 업로드 디렉토리 생성
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.MODEL_CACHE_DIR, exist_ok=True)

settings = Settings()
