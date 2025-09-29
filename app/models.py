"""
Pydantic 모델 정의
"""
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime

class TranscriptionSegment(BaseModel):
    """전사 세그먼트 모델"""
    speaker: str = Field(..., description="화자 ID")
    start: float = Field(..., description="시작 시간 (초)")
    end: float = Field(..., description="종료 시간 (초)")
    text: str = Field(..., description="전사된 텍스트")
    confidence: Optional[float] = Field(None, description="신뢰도 점수")

class TranscriptionResponse(BaseModel):
    """전사 응답 모델"""
    status: str = Field(..., description="처리 상태")
    transcription: List[TranscriptionSegment] = Field(..., description="화자별 전사 결과")
    processing_time: float = Field(..., description="처리 시간 (초)")
    language: str = Field(..., description="감지된 언어")
    total_duration: float = Field(..., description="전체 오디오 길이 (초)")
    speakers_count: int = Field(..., description="감지된 화자 수")

class ErrorResponse(BaseModel):
    """에러 응답 모델"""
    status: str = Field(..., description="에러 상태")
    error: str = Field(..., description="에러 메시지")
    detail: Optional[str] = Field(None, description="상세 에러 정보")
    timestamp: datetime = Field(default_factory=datetime.now, description="에러 발생 시간")

class HealthResponse(BaseModel):
    """헬스 체크 응답 모델"""
    status: str = Field(..., description="서버 상태")
    version: str = Field(..., description="API 버전")
    timestamp: datetime = Field(default_factory=datetime.now, description="체크 시간")
    models_loaded: bool = Field(..., description="모델 로드 상태")
    device: str = Field(..., description="사용 중인 디바이스")

class FileInfo(BaseModel):
    """파일 정보 모델"""
    filename: str = Field(..., description="파일명")
    content_type: str = Field(..., description="MIME 타입")
    size: int = Field(..., description="파일 크기 (바이트)")
    duration: Optional[float] = Field(None, description="오디오 길이 (초)")

class SimpleTranscriptionSegment(BaseModel):
    """단순 전사 세그먼트 모델 (화자분리 없음)"""
    start: float = Field(..., description="시작 시간 (초)")
    end: float = Field(..., description="종료 시간 (초)")
    text: str = Field(..., description="전사된 텍스트")
    confidence: Optional[float] = Field(None, description="신뢰도 점수")

class SimpleTranscriptionResponse(BaseModel):
    """단순 전사 응답 모델 (화자분리 없음)"""
    status: str = Field(..., description="처리 상태")
    transcription: List[SimpleTranscriptionSegment] = Field(..., description="전사 결과")
    processing_time: float = Field(..., description="처리 시간 (초)")
    language: str = Field(..., description="감지된 언어")
    total_duration: float = Field(..., description="전체 오디오 길이 (초)")
    full_text: str = Field(..., description="전체 전사 텍스트")

class ProcessingStatus(BaseModel):
    """처리 상태 모델"""
    task_id: str = Field(..., description="작업 ID")
    status: str = Field(..., description="처리 상태")
    progress: float = Field(..., description="진행률 (0-100)")
    message: Optional[str] = Field(None, description="상태 메시지")
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")
    updated_at: datetime = Field(default_factory=datetime.now, description="업데이트 시간")
