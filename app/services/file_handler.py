"""
파일 처리 유틸리티
"""
import os
import logging
from typing import Optional
from fastapi import UploadFile
import librosa
import soundfile as sf

from app.config import settings

logger = logging.getLogger(__name__)

class FileHandler:
    """파일 처리 클래스"""
    
    def __init__(self):
        self.allowed_extensions = settings.ALLOWED_EXTENSIONS
        self.max_file_size = settings.MAX_FILE_SIZE
    
    def is_valid_file(self, file: UploadFile) -> bool:
        """
        파일 유효성 검사
        
        Args:
            file: 업로드된 파일
            
        Returns:
            bool: 유효한 파일 여부
        """
        if not file.filename:
            return False
        
        # 확장자 검사
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in self.allowed_extensions:
            logger.warning(f"지원하지 않는 파일 확장자: {file_ext}")
            return False
        
        return True
    
    def get_file_info(self, file: UploadFile) -> dict:
        """
        파일 정보 추출
        
        Args:
            file: 업로드된 파일
            
        Returns:
            dict: 파일 정보
        """
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": file.size
        }
    
    def convert_audio_format(self, input_path: str, output_path: str) -> str:
        """
        오디오 파일을 WAV 형식으로 변환
        
        Args:
            input_path: 입력 파일 경로
            output_path: 출력 파일 경로
            
        Returns:
            str: 변환된 파일 경로
        """
        try:
            # librosa로 오디오 로드 (자동으로 16kHz로 리샘플링)
            audio, sr = librosa.load(input_path, sr=16000, mono=True)
            
            # WAV 파일로 저장
            sf.write(output_path, audio, sr)
            
            logger.info(f"오디오 변환 완료: {input_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"오디오 변환 실패: {str(e)}")
            raise
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        오디오 파일의 길이를 초 단위로 반환
        
        Args:
            audio_path: 오디오 파일 경로
            
        Returns:
            float: 오디오 길이 (초)
        """
        try:
            # librosa로 오디오 길이 계산
            duration = librosa.get_duration(path=audio_path)
            return duration
        except Exception as e:
            logger.error(f"오디오 길이 계산 실패: {str(e)}")
            return 0.0
    
    def validate_audio_file(self, audio_path: str) -> bool:
        """
        오디오 파일 유효성 검사
        
        Args:
            audio_path: 오디오 파일 경로
            
        Returns:
            bool: 유효한 오디오 파일 여부
        """
        try:
            # librosa로 오디오 파일 로드 시도
            audio, sr = librosa.load(audio_path, sr=None)
            
            # 기본적인 검사
            if len(audio) == 0:
                logger.warning("빈 오디오 파일")
                return False
            
            if sr < 8000:  # 최소 8kHz
                logger.warning(f"샘플링 레이트가 너무 낮음: {sr}Hz")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"오디오 파일 검증 실패: {str(e)}")
            return False
    
    def cleanup_temp_files(self, *file_paths: str):
        """
        임시 파일들 정리
        
        Args:
            *file_paths: 정리할 파일 경로들
        """
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    logger.debug(f"임시 파일 삭제: {file_path}")
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {file_path}, {str(e)}")
    
    def get_safe_filename(self, filename: str) -> str:
        """
        안전한 파일명 생성
        
        Args:
            filename: 원본 파일명
            
        Returns:
            str: 안전한 파일명
        """
        import re
        import uuid
        
        # 특수문자 제거 및 공백을 언더스코어로 변경
        safe_name = re.sub(r'[^\w\-_\.]', '_', filename)
        
        # UUID 추가로 중복 방지
        name, ext = os.path.splitext(safe_name)
        return f"{name}_{uuid.uuid4().hex[:8]}{ext}"
