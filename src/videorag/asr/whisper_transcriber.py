"""Automatic Speech Recognition using OpenAI Whisper."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import whisper
from loguru import logger

from videorag.config.settings import settings
from videorag.utils.paths import get_transcript_path


@dataclass
class TranscriptSegment:
    """A single segment of transcribed speech."""

    start: float
    end: float
    text: str
    confidence: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptSegment":
        """Create from dictionary."""
        return cls(
            start=data["start"],
            end=data["end"],
            text=data["text"],
            confidence=data.get("confidence"),
        )


@dataclass
class Transcript:
    """Complete transcript for a video chunk."""

    video_id: str
    chunk_idx: int
    segments: List[TranscriptSegment]
    full_text: str
    language: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "video_id": self.video_id,
            "chunk_idx": self.chunk_idx,
            "segments": [seg.to_dict() for seg in self.segments],
            "full_text": self.full_text,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Transcript":
        """Create from dictionary."""
        return cls(
            video_id=data["video_id"],
            chunk_idx=data["chunk_idx"],
            segments=[TranscriptSegment.from_dict(seg) for seg in data["segments"]],
            full_text=data["full_text"],
            language=data.get("language"),
        )


class WhisperTranscriber:
    """Wrapper for Whisper ASR model with batching support."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize Whisper transcriber.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cpu, cuda)
        """
        self.model_name = model_name or settings.whisper_model
        self.device = device or settings.whisper_device

        logger.info(f"Loading Whisper model: {self.model_name} on {self.device}")
        self.model = whisper.load_model(self.model_name, device=self.device)
        logger.info("Whisper model loaded successfully")

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """
        Transcribe audio file using Whisper.

        Args:
            audio_path: Path to audio/video file
            language: Language code (e.g., 'en', 'es'). Auto-detect if None
            **kwargs: Additional arguments for whisper.transcribe()

        Returns:
            Whisper transcription result dictionary

        Raises:
            ValueError: If file doesn't exist
        """
        if not audio_path.exists():
            raise ValueError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing: {audio_path.name}")

        result = self.model.transcribe(
            str(audio_path),
            language=language,
            verbose=False,
            **kwargs,
        )

        logger.info(
            f"Transcription complete: {len(result.get('segments', []))} segments, "
            f"language={result.get('language', 'unknown')}"
        )

        return result

    def transcribe_to_transcript(
        self,
        audio_path: Path,
        video_id: str,
        chunk_idx: int,
        language: Optional[str] = None,
    ) -> Transcript:
        """
        Transcribe audio and return structured Transcript object.

        Args:
            audio_path: Path to audio/video file
            video_id: Unique video identifier
            chunk_idx: Chunk index
            language: Language code (optional)

        Returns:
            Transcript object
        """
        result = self.transcribe(audio_path, language=language)

        segments = [
            TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                confidence=seg.get("avg_logprob"),  # Use log probability as confidence
            )
            for seg in result.get("segments", [])
        ]

        full_text = " ".join(seg.text for seg in segments)

        transcript = Transcript(
            video_id=video_id,
            chunk_idx=chunk_idx,
            segments=segments,
            full_text=full_text,
            language=result.get("language"),
        )

        return transcript


def save_transcript(transcript: Transcript, output_path: Path) -> None:
    """
    Save transcript to JSON file.

    Args:
        transcript: Transcript object
        output_path: Path to output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved transcript to {output_path}")


def load_transcript(transcript_path: Path) -> Transcript:
    """
    Load transcript from JSON file.

    Args:
        transcript_path: Path to JSON file

    Returns:
        Transcript object
    """
    with open(transcript_path, encoding="utf-8") as f:
        data = json.load(f)
    return Transcript.from_dict(data)
