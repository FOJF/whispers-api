"""
WhisperX + pyannote.audio 화자분리 전사 API 서버
"""
import os
import logging
import tempfile
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.config import settings
from app.models import (
    TranscriptionResponse, 
    SimpleTranscriptionResponse,
    ErrorResponse, 
    HealthResponse,
    FileInfo
)
from app.services.transcription import TranscriptionService
from app.services.file_handler import FileHandler

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="WhisperX와 pyannote.audio를 활용한 화자분리 전사 API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서비스 인스턴스
transcription_service = None
file_handler = FileHandler()

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 초기화"""
    global transcription_service
    try:
        logger.info("WhisperX API 서버 시작 중...")
        logger.info(f"디바이스: {settings.DEVICE}")
        logger.info(f"Whisper 모델: {settings.WHISPER_MODEL}")
        
        # 전사 서비스 초기화
        transcription_service = TranscriptionService()
        await transcription_service.initialize()
        
        logger.info("모든 서비스가 성공적으로 초기화되었습니다.")
    except Exception as e:
        logger.error(f"서비스 초기화 실패: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 정리"""
    logger.info("WhisperX API 서버 종료 중...")
    if transcription_service:
        await transcription_service.cleanup()

@app.get("/", response_model=HealthResponse)
async def root():
    """루트 엔드포인트"""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        models_loaded=transcription_service is not None,
        device=settings.DEVICE
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크 엔드포인트"""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        models_loaded=transcription_service is not None,
        device=settings.DEVICE
    )

@app.post("/v1/audio/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="오디오/비디오 파일")
):
    """
    오디오/비디오 파일을 업로드하여 화자분리 전사를 수행합니다.
    
    Args:
        file: 업로드된 오디오/비디오 파일
        
    Returns:
        TranscriptionResponse: 화자별 전사 결과
    """
    if not transcription_service:
        raise HTTPException(
            status_code=503, 
            detail="전사 서비스가 초기화되지 않았습니다."
        )
    
    try:
        # 파일 유효성 검사
        if not file_handler.is_valid_file(file):
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # 파일 크기 검사
        if file.size and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"파일 크기가 너무 큽니다. 최대 크기: {settings.MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        logger.info(f"파일 업로드 시작: {file.filename}")
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # 전사 수행
            result = await transcription_service.transcribe_with_diarization(
                audio_path=temp_file_path,
                language=settings.LANGUAGE
            )
            
            # 임시 파일 정리 (백그라운드)
            background_tasks.add_task(os.unlink, temp_file_path)
            
            logger.info(f"전사 완료: {file.filename}")
            return result
            
        except Exception as e:
            # 에러 발생 시 임시 파일 정리
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"전사 처리 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"전사 처리 중 오류가 발생했습니다: {str(e)}"
        )

@app.post("/v1/audio/transcribe-async")
async def transcribe_audio_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="오디오/비디오 파일")
):
    """
    비동기 전사 처리 (큰 파일용)
    """
    # TODO: 비동기 처리 구현
    raise HTTPException(
        status_code=501,
        detail="비동기 전사 기능은 아직 구현되지 않았습니다."
    )

@app.post("/v1/audio/transcribe-simple", response_model=SimpleTranscriptionResponse)
async def transcribe_audio_simple(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="오디오/비디오 파일")
):
    """
    오디오/비디오 파일을 업로드하여 단순 전사를 수행합니다. (화자분리 없음)
    
    Args:
        file: 업로드된 오디오/비디오 파일
        
    Returns:
        SimpleTranscriptionResponse: 전사 결과 (화자분리 없음)
    """
    if not transcription_service:
        raise HTTPException(
            status_code=503, 
            detail="전사 서비스가 초기화되지 않았습니다."
        )
    
    try:
        # 파일 유효성 검사
        if not file_handler.is_valid_file(file):
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # 파일 크기 검사
        if file.size and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"파일 크기가 너무 큽니다. 최대 크기: {settings.MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        logger.info(f"단순 전사 파일 업로드 시작: {file.filename}")
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # 단순 전사 수행
            result = await transcription_service.transcribe_simple(
                audio_path=temp_file_path,
                language=settings.LANGUAGE
            )
            
            # 임시 파일 정리 (백그라운드)
            background_tasks.add_task(os.unlink, temp_file_path)
            
            logger.info(f"단순 전사 완료: {file.filename}")
            return result
            
        except Exception as e:
            # 에러 발생 시 임시 파일 정리
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"단순 전사 처리 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"전사 처리 중 오류가 발생했습니다: {str(e)}"
        )

@app.get("/v1/models/info")
async def get_models_info():
    """로드된 모델 정보 조회"""
    if not transcription_service:
        raise HTTPException(
            status_code=503,
            detail="전사 서비스가 초기화되지 않았습니다."
        )
    
    return {
        "whisper_model": settings.WHISPER_MODEL,
        "diarization_model": settings.DIARIZATION_MODEL,
        "device": settings.DEVICE,
        "compute_type": settings.COMPUTE_TYPE,
        "language": settings.LANGUAGE
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )
