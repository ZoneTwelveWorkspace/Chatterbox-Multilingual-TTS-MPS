"""
Enhanced TTS module with intelligent text chunking and progress tracking.

This module provides advanced text-to-speech capabilities that automatically
handle long texts by chunking them intelligently and providing real-time
progress updates with tqdm integration.
"""

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import torch
from tqdm.auto import tqdm

from .progress_tracker import ProgressStage, TTSProgressTracker, get_global_tracker
from .text_chunker import Chunk, TextChunker, smart_chunk_text

try:
    import soundfile as sf

    AUDIOIO_AVAILABLE = True
except ImportError:
    try:
        import librosa

        AUDIOIO_AVAILABLE = True
    except ImportError:
        AUDIOIO_AVAILABLE = False
        logging.warning(
            "Audio I/O libraries not available. Audio concatenation may not work properly."
        )


@dataclass
class TTSGenerationConfig:
    """Configuration for TTS generation."""

    # Text processing
    max_chars: int = 300
    language_id: str = "en"

    # Audio generation parameters
    exaggeration: float = 0.5
    temperature: float = 0.8
    cfg_weight: float = 0.5
    repetition_penalty: float = 2.0
    min_p: float = 0.05
    top_p: float = 1.0

    # Reference audio for voice cloning
    reference_audio_path: Optional[str] = None

    # Progress tracking
    show_progress: bool = True
    enable_tqdm: bool = True
    progress_callback: Optional[Callable[[Dict], None]] = None

    # Concatenation settings
    concatenate_audio: bool = True
    add_silence_between_chunks: float = 0.05  # seconds

    # Audio settings
    sample_rate: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "max_chars": self.max_chars,
            "language_id": self.language_id,
            "exaggeration": self.exaggeration,
            "temperature": self.temperature,
            "cfg_weight": self.cfg_weight,
            "repetition_penalty": self.repetition_penalty,
            "min_p": self.min_p,
            "top_p": self.top_p,
            "show_progress": self.show_progress,
            "enable_tqdm": self.enable_tqdm,
            "concatenate_audio": self.concatenate_audio,
            "add_silence_between_chunks": self.add_silence_between_chunks,
            "sample_rate": self.sample_rate,
            "reference_audio_path": self.reference_audio_path,
        }


@dataclass
class GenerationResult:
    """Result of TTS generation operation."""

    audio_data: np.ndarray
    sample_rate: int
    duration: float
    chunk_count: int
    job_id: str
    metadata: Dict[str, Any]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the audio data."""
        return self.audio_data.shape

    def save(self, filepath: Union[str, Path], format: str = "wav") -> None:
        """
        Save the generated audio to a file.

        Args:
            filepath: Path to save the audio file
            format: Audio format ('wav', 'mp3', 'flac', etc.)
        """
        if not AUDIOIO_AVAILABLE:
            raise RuntimeError("Audio I/O libraries not available. Cannot save audio.")

        filepath = Path(filepath)

        if format.lower() == "wav":
            sf.write(filepath, self.audio_data.T, self.sample_rate)
        else:
            # For other formats, we might need additional dependencies
            raise ValueError(
                f"Format '{format}' not supported. Use 'wav' or install additional dependencies."
            )

    def get_info(self) -> Dict[str, Any]:
        """Get information about the generated audio."""
        return {
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.audio_data.shape[0]
            if len(self.audio_data.shape) > 1
            else 1,
            "total_samples": self.audio_data.shape[-1],
            "chunk_count": self.chunk_count,
            "job_id": self.job_id,
            "metadata": self.metadata,
        }


class EnhancedTTS:
    """
    Enhanced Text-to-Speech engine with intelligent chunking and progress tracking.

    This class extends basic TTS functionality with:
    - Automatic text chunking for long inputs
    - Real-time progress tracking with tqdm
    - Audio concatenation
    - Comprehensive error handling
    - Performance optimization
    """

    def __init__(
        self,
        model: Any,
        device: str = "cpu",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the enhanced TTS engine.

        Args:
            model: The underlying TTS model
            device: Device to run inference on
            logger: Optional logger instance
        """
        self.model = model
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        # Initialize components
        self.chunker = TextChunker()
        self.progress_tracker = get_global_tracker()

        # Thread safety
        self._lock = threading.Lock()

        # Configuration
        self._default_config = TTSGenerationConfig()

        self.logger.info(f"Enhanced TTS initialized with device: {device}")

    def generate(
        self,
        text: str,
        config: Optional[TTSGenerationConfig] = None,
        job_id: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate speech audio from text with intelligent chunking.

        Args:
            text: Input text to synthesize
            config: Generation configuration
            job_id: Optional job identifier for progress tracking

        Returns:
            GenerationResult containing audio data and metadata
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty or whitespace only.")

        # Generate job ID if not provided
        if job_id is None:
            job_id = str(uuid4())[:8]

        # Use provided config or default
        config = config or self._default_config
        self.logger.info(f"Starting TTS generation for job {job_id}")

        try:
            # Step 1: Chunk the text if necessary
            self.progress_tracker.update_stage(
                job_id, ProgressStage.CHUNKING, "Chunking text"
            )
            chunks = self._chunk_text_intelligently(text, config)

            if not chunks:
                raise ValueError("Failed to chunk text properly.")

            self.logger.info(f"Text chunked into {len(chunks)} parts")

            # Step 2: Prepare conditionals if reference audio is provided
            if config.reference_audio_path:
                self.logger.info(
                    f"Preparing conditionals from reference audio: {config.reference_audio_path}"
                )
                self._prepare_reference_conditionals(config)

            # Step 3: Generate audio for each chunk
            audio_chunks = self._generate_audio_chunks(job_id, chunks, config)

            # Step 4: Concatenate audio if configured
            final_audio, final_sample_rate = self._concatenate_audio(
                job_id, audio_chunks, config
            )

            # Step 5: Create result
            duration = final_audio.shape[-1] / final_sample_rate
            result = GenerationResult(
                audio_data=final_audio,
                sample_rate=final_sample_rate,
                duration=duration,
                chunk_count=len(chunks),
                job_id=job_id,
                metadata={
                    "original_text_length": len(text),
                    "chunk_count": len(chunks),
                    "generation_config": config.to_dict(),
                    "device": self.device,
                    "concatenated": config.concatenate_audio,
                    "reference_audio_used": config.reference_audio_path is not None,
                },
            )

            # Mark job as completed
            self.progress_tracker.complete_job(job_id, success=True)

            self.logger.info(
                f"TTS generation completed for job {job_id}: {duration:.2f}s"
            )
            return result

        except Exception as e:
            self.logger.error(f"TTS generation failed for job {job_id}: {str(e)}")
            self.progress_tracker.complete_job(job_id, success=False)
            raise

    def _prepare_reference_conditionals(self, config: TTSGenerationConfig):
        """
        Prepare reference conditionals from audio file for efficient chunk processing.

        This method prepares the conditionals once and stores them in the model,
        so they can be reused for all chunks in a generation job. This is much
        more efficient than preparing conditionals for each chunk individually.

        Args:
            config: Generation configuration containing reference_audio_path
        """
        if not config.reference_audio_path:
            return

        # Check if model has prepare_conditionals method
        if hasattr(self.model, "prepare_conditionals"):
            try:
                self.logger.info(
                    f"Preparing conditionals from reference audio: {config.reference_audio_path}"
                )
                self.model.prepare_conditionals(
                    wav_fpath=config.reference_audio_path,
                    exaggeration=config.exaggeration,
                )
                self.logger.info("Reference conditionals prepared successfully")
            except Exception as e:
                self.logger.warning(
                    f"Failed to prepare conditionals from reference audio: {str(e)}. "
                    "Continuing without reference audio conditioning."
                )
        else:
            self.logger.warning(
                "Model does not support prepare_conditionals method. "
                "Reference audio will not be used for conditioning."
            )

    def _chunk_text_intelligently(
        self, text: str, config: TTSGenerationConfig
    ) -> List[Chunk]:
        """
        Chunk text intelligently based on configuration.

        Args:
            text: Input text
            config: Generation configuration

        Returns:
            List of text chunks
        """
        # Check if chunking is needed
        if len(text) <= config.max_chars:
            return [
                Chunk(text=text.strip(), index=0, char_count=len(text), is_final=True)
            ]

        # Use the text chunker for intelligent chunking
        chunks = self.chunker.chunk_text(text, config.language_id)

        # Validate chunks
        for chunk in chunks:
            if chunk.char_count > config.max_chars:
                self.logger.warning(
                    f"Chunk exceeds max_chars: {chunk.char_count} > {config.max_chars}"
                )

        return chunks

    def _generate_audio_chunks(
        self, job_id: str, chunks: List[Chunk], config: TTSGenerationConfig
    ) -> List[np.ndarray]:
        """
        Generate audio for each text chunk.

        Args:
            job_id: Job identifier
            chunks: List of text chunks
            config: Generation configuration

        Returns:
            List of audio arrays
        """
        audio_chunks = []
        total_chunks = len(chunks)

        # Set up progress tracking
        if config.show_progress and config.enable_tqdm:
            pbar = tqdm(
                total=total_chunks,
                desc="Generating audio",
                unit="chunk",
                position=0,
                leave=True,
            )
        else:
            pbar = None

        self.progress_tracker.update_stage(
            job_id, ProgressStage.GENERATING, "Generating audio chunks"
        )

        try:
            for i, chunk in enumerate(chunks):
                # Update progress
                if pbar:
                    pbar.set_description(f"Processing chunk {i + 1}/{total_chunks}")

                self.progress_tracker.start_chunk_processing(job_id, i)

                try:
                    # Generate audio for this chunk
                    chunk_audio = self._generate_single_chunk(chunk.text, config)

                    # Convert to numpy array if needed
                    if isinstance(chunk_audio, torch.Tensor):
                        chunk_audio = chunk_audio.squeeze(0).cpu().numpy()

                    audio_chunks.append(chunk_audio)

                    # Update progress
                    self.progress_tracker.complete_chunk(job_id, i, success=True)

                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "chunk": f"{i + 1}/{total_chunks}",
                                "duration": f"{chunk_audio.shape[-1] / getattr(self.model, 'sr', 44100):.1f}s",
                            }
                        )

                except Exception as e:
                    self.logger.error(
                        f"Failed to generate audio for chunk {i}: {str(e)}"
                    )
                    self.progress_tracker.complete_chunk(job_id, i, success=False)

                    # Re-raise the exception to fail the entire job
                    raise

        finally:
            if pbar:
                pbar.close()

        self.logger.info(f"Generated {len(audio_chunks)} audio chunks")
        return audio_chunks

    def _generate_single_chunk(
        self, text: str, config: TTSGenerationConfig
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate audio for a single text chunk.

        Args:
            text: Text for this chunk
            config: Generation configuration

        Returns:
            Audio data as numpy array or torch tensor
        """
        # Use the underlying TTS model
        if hasattr(self.model, "generate"):
            return self.model.generate(
                text=text,
                language_id=config.language_id,
                audio_prompt_path=config.reference_audio_path,
                exaggeration=config.exaggeration,
                temperature=config.temperature,
                cfg_weight=config.cfg_weight,
                repetition_penalty=config.repetition_penalty,
                min_p=config.min_p,
                top_p=config.top_p,
            )
        else:
            raise RuntimeError("Model does not have a 'generate' method")

    def _concatenate_audio(
        self, job_id: str, audio_chunks: List[np.ndarray], config: TTSGenerationConfig
    ) -> Tuple[np.ndarray, int]:
        """
        Concatenate audio chunks into final output.

        Args:
            job_id: Job identifier
            audio_chunks: List of audio arrays
            config: Generation configuration

        Returns:
            Tuple of (concatenated_audio, sample_rate)
        """
        if not audio_chunks:
            raise ValueError("No audio chunks to concatenate")

        self.progress_tracker.update_stage(
            job_id, ProgressStage.CONCATENATING, "Concatenating audio chunks"
        )

        # Get sample rate from first chunk
        first_chunk = audio_chunks[0]
        if hasattr(first_chunk, "shape"):
            sample_rate = getattr(self.model, "sr", 44100)
        else:
            sample_rate = 44100  # Default fallback

        if config.concatenate_audio and len(audio_chunks) > 1:
            self.logger.info("Concatenating audio chunks")

            # Add silence between chunks if configured
            silence_duration = config.add_silence_between_chunks
            silence_samples = int(silence_duration * sample_rate)
            silence = np.zeros((silence_samples,), dtype=first_chunk.dtype)

            # Concatenate with silence between chunks
            concatenated = []
            for i, chunk in enumerate(audio_chunks):
                concatenated.append(chunk)
                if i < len(audio_chunks) - 1:  # Don't add silence after last chunk
                    concatenated.append(silence)

            final_audio = np.concatenate(concatenated, axis=-1)

        else:
            self.logger.info("Returning first chunk only (concatenation disabled)")
            final_audio = first_chunk

        # Ensure final_audio is 2D (channels, samples)
        if final_audio.ndim == 1:
            final_audio = final_audio[np.newaxis, :]
        elif final_audio.ndim > 2:
            final_audio = final_audio.reshape(final_audio.shape[0], -1)

        self.logger.info(f"Audio concatenation complete: {final_audio.shape}")
        return final_audio, sample_rate

    def generate_batch(
        self,
        texts: List[str],
        config: Optional[TTSGenerationConfig] = None,
        job_prefix: str = "batch",
    ) -> List[GenerationResult]:
        """
        Generate speech for multiple texts in batch.

        Args:
            texts: List of input texts
            config: Generation configuration
            job_prefix: Prefix for job IDs

        Returns:
            List of GenerationResult objects
        """
        results = []
        total_texts = len(texts)

        if config is None:
            config = self._default_config.copy()
            config.show_progress = True
            config.enable_tqdm = True

        # Set up batch progress bar
        if config.show_progress and config.enable_tqdm:
            pbar = tqdm(
                total=total_texts,
                desc="Batch TTS generation",
                unit="text",
                position=0,
                leave=True,
            )
        else:
            pbar = None

        try:
            for i, text in enumerate(texts):
                if pbar:
                    pbar.set_description(f"Processing text {i + 1}/{total_texts}")

                job_id = f"{job_prefix}_{i + 1}_{uuid4().hex[:8]}"

                try:
                    result = self.generate(text, config, job_id)
                    results.append(result)

                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "current": f"{i + 1}/{total_texts}",
                                "avg_duration": f"{np.mean([r.duration for r in results]):.1f}s",
                            }
                        )

                except Exception as e:
                    self.logger.error(f"Failed to process text {i + 1}: {str(e)}")
                    # Add empty result or re-raise based on requirements
                    raise

        finally:
            if pbar:
                pbar.close()

        self.logger.info(
            f"Batch generation completed: {len(results)}/{total_texts} successful"
        )
        return results

    def estimate_processing_time(
        self, text: str, config: TTSGenerationConfig
    ) -> Dict[str, float]:
        """
        Estimate processing time for a given text.

        Args:
            text: Input text
            config: Generation configuration

        Returns:
            Dictionary with time estimates
        """
        chunks = self._chunk_text_intelligently(text, config)
        chunk_count = len(chunks)

        # Rough estimates based on typical processing speeds
        # These would ideally be calibrated based on actual performance metrics
        estimates_per_chunk = {
            "chunking": 0.01,  # Text chunking is very fast
            "generation": 2.5,  # TTS generation per chunk (seconds)
            "concatenation": 0.05,  # Audio concatenation
            "overhead": 0.5,  # General overhead
        }

        total_generation_time = chunk_count * estimates_per_chunk["generation"]
        total_time = (
            estimates_per_chunk["chunking"]
            + total_generation_time
            + estimates_per_chunk["concatenation"]
            + estimates_per_chunk["overhead"]
        )

        return {
            "total_estimated_time": total_time,
            "generation_time": total_generation_time,
            "chunk_count": chunk_count,
            "text_length": len(text),
            "chunks": [chunk.char_count for chunk in chunks],
            "estimates_per_chunk": estimates_per_chunk,
        }

    @contextmanager
    def temporary_config(self, **kwargs):
        """Context manager for temporary configuration changes."""
        original_config = self._default_config
        new_config = TTSGenerationConfig(**original_config.to_dict())
        for key, value in kwargs.items():
            setattr(new_config, key, value)
        self._default_config = new_config

        try:
            yield new_config
        finally:
            self._default_config = original_config

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the underlying model."""
        info = {
            "device": self.device,
            "model_type": type(self.model).__name__,
            "supported_languages": getattr(
                self.model, "get_supported_languages", lambda: {}
            )(),
            "sample_rate": getattr(self.model, "sr", None),
            "max_chars": self._default_config.max_chars,
        }

        # Add any other relevant model attributes
        if hasattr(self.model, "__dict__"):
            model_attrs = {}
            for attr_name in ["device", "sr", "tokenizer", "t3", "s3gen", "ve"]:
                if hasattr(self.model, attr_name):
                    model_attrs[attr_name] = str(getattr(self.model, attr_name))
            info["model_attributes"] = model_attrs

        return info

    def set_progress_callback(self, callback: Callable[[Dict], None]):
        """
        Set a progress callback function.

        Args:
            callback: Function that receives progress updates
        """
        self.progress_tracker.register_progress_callback(callback)

    def validate_config(self, config: TTSGenerationConfig) -> bool:
        """
        Validate a generation configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if configuration is valid
        """
        try:
            # Check basic parameters
            if config.max_chars <= 0:
                return False

            if not 0.0 <= config.exaggeration <= 3.0:
                return False

            if not 0.0 <= config.temperature <= 10.0:
                return False

            if not 0.0 <= config.cfg_weight <= 2.0:
                return False

            if not config.language_id:
                return False

            return True

        except Exception:
            return False


# Convenience functions for easy integration
def create_enhanced_tts(
    model: Any,
    device: str = "cpu",
    logger: Optional[logging.Logger] = None,
) -> EnhancedTTS:
    """
    Create an enhanced TTS instance.

    Args:
        model: Underlying TTS model
        device: Device to run on
        logger: Optional logger

    Returns:
        EnhancedTTS instance
    """
    return EnhancedTTS(model=model, device=device, logger=logger)


def generate_speech(
    model: Any,
    text: str,
    device: str = "cpu",
    language_id: str = "en",
    max_chars: int = 300,
    **kwargs,
) -> GenerationResult:
    """
    Convenience function for simple speech generation.

    Args:
        model: TTS model
        text: Input text
        device: Device to run on
        language_id: Language code
        max_chars: Maximum characters per chunk
        **kwargs: Additional generation parameters

    Returns:
        GenerationResult with audio data
    """
    enhanced_tts = create_enhanced_tts(model, device)

    config = TTSGenerationConfig(
        language_id=language_id,
        max_chars=max_chars,
        **{k: v for k, v in kwargs.items() if k in TTSGenerationConfig.__annotations__},
    )

    return enhanced_tts.generate(text, config)


# Example usage
if __name__ == "__main__":
    # Example usage of the enhanced TTS system
    import sys

    # This would normally use a real TTS model
    class MockModel:
        def generate(self, text, **kwargs):
            # Mock implementation - returns random audio data
            sr = 44100
            duration = len(text) / 50  # Rough estimate
            samples = int(duration * sr)
            audio = np.random.randn(samples).astype(np.float32)
            return torch.from_numpy(audio).unsqueeze(0)

        @property
        def sr(self):
            return 44100

    # Test the enhanced TTS
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    model = MockModel()
    enhanced_tts = create_enhanced_tts(model, device="cpu", logger=logger)

    # Test with long text
    long_text = (
        "This is a very long text that needs to be split into multiple chunks for processing. "
        "The enhanced TTS system automatically handles this by intelligently chunking the text "
        "at natural break points such as sentence endings and punctuation marks. "
        "Each chunk is processed separately and then the audio is concatenated together seamlessly. "
        "This approach ensures that long texts can be processed efficiently while maintaining "
        "high quality output and providing progress feedback to the user."
    ) * 5  # Make it really long

    # Generate speech
    config = TTSGenerationConfig(
        language_id="en",
        max_chars=300,
        show_progress=True,
        enable_tqdm=True,
        concatenate_audio=True,
    )

    try:
        result = enhanced_tts.generate(long_text, config)
        print(f"Generated speech: {result.duration:.2f}s, {result.chunk_count} chunks")

        # Save the result
        result.save("test_output.wav")
        print("Audio saved to test_output.wav")

        # Show model info
        info = enhanced_tts.get_model_info()
        print(f"Model info: {info}")

        # Estimate processing time
        estimate = enhanced_tts.estimate_processing_time(long_text, config)
        print(f"Processing time estimate: {estimate}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
