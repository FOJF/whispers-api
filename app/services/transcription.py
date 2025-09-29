"""
WhisperX + pyannote.audio 화자분리 전사 서비스
"""
import os
import time
import logging
import tempfile
from typing import List, Optional, Dict, Any
import torch
import whisperx
from pyannote.audio import Pipeline
import librosa
import soundfile as sf

from app.config import settings
from app.models import TranscriptionResponse, TranscriptionSegment, SimpleTranscriptionResponse, SimpleTranscriptionSegment
from app.services.file_handler import FileHandler

logger = logging.getLogger(__name__)

class TranscriptionService:
    """전사 서비스 클래스"""
    
    def __init__(self):
        self.whisper_model = None
        self.diarization_pipeline = None
        self.align_model = None
        self.align_metadata = None
        self.file_handler = FileHandler()
        self.is_initialized = False
    
    async def initialize(self):
        """서비스 초기화"""
        try:
            logger.info("전사 서비스 초기화 시작...")
            
            # HuggingFace 토큰 설정
            if settings.HUGGINGFACE_TOKEN:
                os.environ["HF_TOKEN"] = settings.HUGGINGFACE_TOKEN
            
            # WhisperX 모델 로드
            await self._load_whisper_model()
            
            # pyannote.audio 파이프라인 로드
            await self._load_diarization_pipeline()
            
            # 정렬 모델 로드
            await self._load_alignment_model()
            
            self.is_initialized = True
            logger.info("전사 서비스 초기화 완료")
            
        except Exception as e:
            logger.error(f"전사 서비스 초기화 실패: {str(e)}")
            raise
    
    async def _load_whisper_model(self):
        """WhisperX 모델 로드"""
        try:
            logger.info(f"WhisperX 모델 로드 중: {settings.WHISPER_MODEL}")
            
            self.whisper_model = whisperx.load_model(
                settings.WHISPER_MODEL,
                device=settings.DEVICE,
                compute_type=settings.COMPUTE_TYPE,
                language=settings.LANGUAGE
            )
            
            logger.info("WhisperX 모델 로드 완료")
            
        except Exception as e:
            logger.error(f"WhisperX 모델 로드 실패: {str(e)}")
            raise
    
    async def _load_diarization_pipeline(self):
        """pyannote.audio 화자분리 파이프라인 로드"""
        try:
            logger.info(f"화자분리 파이프라인 로드 중: {settings.DIARIZATION_MODEL}")
            
            # HuggingFace 토큰이 필요한 경우
            if settings.HUGGINGFACE_TOKEN:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    settings.DIARIZATION_MODEL,
                    use_auth_token=settings.HUGGINGFACE_TOKEN
                )
            else:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    settings.DIARIZATION_MODEL
                )
            
            # GPU 사용 시
            if settings.DEVICE == "cuda" and torch.cuda.is_available():
                self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cuda"))
            
            logger.info("화자분리 파이프라인 로드 완료")
            
        except Exception as e:
            logger.error(f"화자분리 파이프라인 로드 실패: {str(e)}")
            raise
    
    async def _load_alignment_model(self):
        """정렬 모델 로드"""
        try:
            logger.info("정렬 모델 로드 중...")
            
            # 언어별 정렬 모델 설정
            if settings.LANGUAGE == "ko":
                align_model, align_metadata = whisperx.load_align_model(
                    language_code=settings.LANGUAGE,
                    device=settings.DEVICE
                )
            else:
                # 영어 또는 기타 언어
                align_model, align_metadata = whisperx.load_align_model(
                    language_code=settings.LANGUAGE or "en",
                    device=settings.DEVICE
                )
            
            self.align_model = align_model
            self.align_metadata = align_metadata
            
            logger.info("정렬 모델 로드 완료")
            
        except Exception as e:
            logger.error(f"정렬 모델 로드 실패: {str(e)}")
            raise
    
    async def transcribe_with_diarization(
        self, 
        audio_path: str, 
        language: Optional[str] = None
    ) -> TranscriptionResponse:
        """
        화자분리 전사 수행
        
        Args:
            audio_path: 오디오 파일 경로
            language: 언어 코드 (None이면 자동 감지)
            
        Returns:
            TranscriptionResponse: 전사 결과
        """
        if not self.is_initialized:
            raise RuntimeError("전사 서비스가 초기화되지 않았습니다.")
        
        start_time = time.time()
        
        try:
            logger.info(f"전사 시작: {audio_path}")
            
            # 1. 오디오 전처리
            processed_audio_path = await self._preprocess_audio(audio_path)
            
            # 2. WhisperX로 전사
            transcription_result = await self._transcribe_audio(processed_audio_path, language)
            
            # 3. 화자분리
            diarization_result = await self._perform_diarization(processed_audio_path)
            
            # 4. 전사와 화자분리 결과 결합
            aligned_segments = await self._align_transcription_with_diarization(
                transcription_result, diarization_result
            )
            
            # 5. 결과 정리
            processing_time = time.time() - start_time
            speakers_count = len(set(segment["speaker"] for segment in aligned_segments))
            
            # 임시 파일 정리
            if processed_audio_path != audio_path:
                self.file_handler.cleanup_temp_files(processed_audio_path)
            
            logger.info(f"전사 완료: {processing_time:.2f}초, 화자 수: {speakers_count}")
            
            return TranscriptionResponse(
                status="success",
                transcription=[
                    TranscriptionSegment(
                        speaker=segment["speaker"],
                        start=segment["start"],
                        end=segment["end"],
                        text=segment["text"],
                        confidence=segment.get("confidence")
                    )
                    for segment in aligned_segments
                ],
                processing_time=processing_time,
                language=language or "auto",
                total_duration=transcription_result.get("duration", 0.0),
                speakers_count=speakers_count
            )
            
        except Exception as e:
            logger.error(f"전사 처리 실패: {str(e)}")
            raise
    
    async def _preprocess_audio(self, audio_path: str) -> str:
        """오디오 전처리"""
        try:
            # librosa로 오디오 로드 및 전처리
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # 임시 파일로 저장
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(temp_file.name, audio, sr)
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"오디오 전처리 실패: {str(e)}")
            raise
    
    async def _transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """WhisperX로 오디오 전사"""
        try:
            logger.info("WhisperX 전사 수행 중...")
            
            # WhisperX 전사
            result = self.whisper_model.transcribe(audio_path, language=language)
            
            # 정렬 수행
            if self.align_model and self.align_metadata:
                result = whisperx.align(
                    result["segments"], 
                    self.align_model, 
                    self.align_metadata, 
                    audio_path, 
                    settings.DEVICE
                )
            
            logger.info("WhisperX 전사 완료")
            return result
            
        except Exception as e:
            logger.error(f"WhisperX 전사 실패: {str(e)}")
            raise
    
    async def _perform_diarization(self, audio_path: str) -> Dict[str, Any]:
        """pyannote.audio로 화자분리"""
        try:
            logger.info("화자분리 수행 중...")
            
            # 화자분리 파이프라인 확인
            if self.diarization_pipeline is None:
                logger.warning("화자분리 파이프라인이 로드되지 않았습니다. 기본 화자로 설정합니다.")
                return {"segments": []}
            
            # 화자분리 수행
            diarization = self.diarization_pipeline(audio_path)
            
            # 결과를 리스트로 변환
            diarization_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                diarization_segments.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end
                })
            
            logger.info(f"화자분리 완료: {len(diarization_segments)}개 세그먼트")
            return {"segments": diarization_segments}
            
        except Exception as e:
            logger.error(f"화자분리 실패: {str(e)}")
            raise
    
    async def _align_transcription_with_diarization(
        self, 
        transcription_result: Dict[str, Any], 
        diarization_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """전사 결과와 화자분리 결과 정렬"""
        try:
            logger.info("전사-화자분리 정렬 수행 중...")
            
            transcription_segments = transcription_result.get("segments", [])
            diarization_segments = diarization_result.get("segments", [])
            
            # 화자분리 결과를 시간순으로 정렬
            diarization_segments.sort(key=lambda x: x["start"])
            
            aligned_segments = []
            
            for trans_segment in transcription_segments:
                trans_start = trans_segment["start"]
                trans_end = trans_segment["end"]
                trans_text = trans_segment["text"].strip()
                
                if not trans_text:
                    continue
                
                # 해당 시간대의 화자 찾기
                speaker = "SPEAKER_00"  # 기본값
                max_overlap = 0
                
                for diar_segment in diarization_segments:
                    diar_start = diar_segment["start"]
                    diar_end = diar_segment["end"]
                    
                    # 시간 겹침 계산
                    overlap_start = max(trans_start, diar_start)
                    overlap_end = min(trans_end, diar_end)
                    overlap_duration = max(0, overlap_end - overlap_start)
                    
                    if overlap_duration > max_overlap:
                        max_overlap = overlap_duration
                        speaker = diar_segment["speaker"]
                
                aligned_segments.append({
                    "speaker": speaker,
                    "start": trans_start,
                    "end": trans_end,
                    "text": trans_text,
                    "confidence": trans_segment.get("confidence")
                })
            
            logger.info(f"정렬 완료: {len(aligned_segments)}개 세그먼트")
            return aligned_segments
            
        except Exception as e:
            logger.error(f"전사-화자분리 정렬 실패: {str(e)}")
            raise
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            logger.info("전사 서비스 리소스 정리 중...")
            
            # 모델 메모리 해제
            if self.whisper_model:
                del self.whisper_model
                self.whisper_model = None
            
            if self.diarization_pipeline:
                del self.diarization_pipeline
                self.diarization_pipeline = None
            
            if self.align_model:
                del self.align_model
                self.align_model = None
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_initialized = False
            logger.info("전사 서비스 리소스 정리 완료")
            
        except Exception as e:
            logger.error(f"리소스 정리 실패: {str(e)}")
    
    async def transcribe_simple(self, audio_path: str, language: Optional[str] = None) -> SimpleTranscriptionResponse:
        """단순 전사 (화자분리 없음)"""
        if not self.is_initialized or not self.whisper_model:
            raise RuntimeError("전사 서비스가 초기화되지 않았습니다.")
        
        start_time = time.time()
        
        try:
            logger.info(f"단순 전사 시작: {audio_path}")
            
            # 오디오 로드
            audio, sr = librosa.load(audio_path, sr=16000)
            duration = len(audio) / sr
            
            # WhisperX로 전사
            result = self.whisper_model.transcribe(audio, language=language or settings.LANGUAGE)
            
            # 세그먼트 변환
            segments = []
            full_text = ""
            
            for segment in result["segments"]:
                segment_text = segment["text"].strip()
                if segment_text:
                    segments.append(SimpleTranscriptionSegment(
                        start=segment["start"],
                        end=segment["end"],
                        text=segment_text,
                        confidence=segment.get("avg_logprob")
                    ))
                    full_text += segment_text + " "
            
            processing_time = time.time() - start_time
            
            logger.info(f"단순 전사 완료: {processing_time:.2f}초")
            
            return SimpleTranscriptionResponse(
                status="success",
                transcription=segments,
                processing_time=processing_time,
                language=result.get("language", language or settings.LANGUAGE),
                total_duration=duration,
                full_text=full_text.strip()
            )
            
        except Exception as e:
            logger.error(f"단순 전사 실패: {str(e)}")
            raise RuntimeError(f"전사 처리 중 오류가 발생했습니다: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """로드된 모델 정보 반환"""
        return {
            "whisper_model": settings.WHISPER_MODEL,
            "diarization_model": settings.DIARIZATION_MODEL,
            "device": settings.DEVICE,
            "compute_type": settings.COMPUTE_TYPE,
            "language": settings.LANGUAGE,
            "is_initialized": self.is_initialized
        }
