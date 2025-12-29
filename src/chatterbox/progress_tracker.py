"""
Progress tracking module with tqdm integration for Chatterbox Multilingual TTS.

This module provides comprehensive progress tracking capabilities for audio generation,
including support for chunked text processing, batch operations, and real-time progress updates
with tqdm integration for beautiful progress bars in the console and web interfaces.
"""

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

try:
    from tqdm import TqdmWarning, tqdm
    from tqdm.auto import tqdm as tqdm_auto

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    # Create a simple fallback progress tracker
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total", 0)
            self.desc = kwargs.get("desc", "")
            self.n = 0
            self._lock = threading.Lock()

        def update(self, n=1):
            with self._lock:
                self.n += n
                if self.n % 10 == 0:  # Print every 10 updates to avoid spam
                    print(f"{self.desc}: {self.n}/{self.total}")

        def close(self):
            pass

        def set_description(self, desc):
            self.desc = desc

        def set_postfix(self, **kwargs):
            pass

        @contextmanager
        def set_postfix_str(self, **kwargs):
            yield


class ProgressStage(Enum):
    """Enumeration of different progress stages."""

    INITIALIZING = "initializing"
    CHUNKING = "chunking"
    LOADING_MODEL = "loading_model"
    GENERATING = "generating"
    POST_PROCESSING = "post_processing"
    CONCATENATING = "concatenating"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ChunkProgress:
    """Progress information for a single chunk."""

    chunk_index: int
    chunk_text: str
    chunk_length: int
    status: str = "pending"  # pending, processing, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    progress_percentage: float = 0.0

    @property
    def duration(self) -> Optional[float]:
        """Get the duration of chunk processing."""
        if self.start_time is None:
            return None
        end_time = self.end_time or time.time()
        return end_time - self.start_time

    @property
    def is_completed(self) -> bool:
        """Check if chunk processing is completed."""
        return self.status == "completed"

    @property
    def is_failed(self) -> bool:
        """Check if chunk processing failed."""
        return self.status == "failed"


@dataclass
class GenerationProgress:
    """Complete progress information for audio generation."""

    job_id: str
    original_text: str
    total_chunks: int
    language_id: str
    current_stage: ProgressStage = ProgressStage.INITIALIZING
    chunk_progress: List[ChunkProgress] = field(default_factory=list)
    overall_progress: float = 0.0
    start_time: Optional[float] = None
    estimated_end_time: Optional[float] = None
    status: str = "running"  # running, completed, failed, cancelled

    def __post_init__(self):
        """Initialize chunk progress tracking after creation."""
        if not self.chunk_progress:
            for i in range(self.total_chunks):
                self.chunk_progress.append(
                    ChunkProgress(
                        chunk_index=i,
                        chunk_text="",  # Will be filled later
                        chunk_length=0,
                    )
                )

    @property
    def completed_chunks(self) -> int:
        """Get the number of completed chunks."""
        return sum(1 for chunk in self.chunk_progress if chunk.is_completed)

    @property
    def failed_chunks(self) -> int:
        """Get the number of failed chunks."""
        return sum(1 for chunk in self.chunk_progress if chunk.is_failed)

    @property
    def elapsed_time(self) -> float:
        """Get the elapsed time since generation started."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    @property
    def estimated_time_remaining(self) -> Optional[float]:
        """Get estimated time remaining."""
        if self.estimated_end_time is None:
            return None
        return max(0, self.estimated_end_time - time.time())

    @property
    def chunks_per_second(self) -> float:
        """Get processing speed (chunks per second)."""
        elapsed = self.elapsed_time
        if elapsed <= 0:
            return 0.0
        return self.completed_chunks / elapsed


class ProgressTracker:
    """
    Main progress tracking class with tqdm integration.

    This class manages progress tracking for audio generation operations,
    providing both real-time progress updates and batch progress reporting.
    """

    def __init__(
        self, enable_tqdm: bool = True, show_eta: bool = True, show_rate: bool = True
    ):
        """
        Initialize the progress tracker.

        Args:
            enable_tqdm: Whether to use tqdm for progress bars
            show_eta: Whether to show estimated time remaining
            show_rate: Whether to show processing rate
        """
        self.enable_tqdm = enable_tqdm and TQDM_AVAILABLE
        self.show_eta = show_eta
        self.show_rate = show_rate
        self.logger = logging.getLogger(__name__)

        # Thread-safe progress tracking
        self._progress_lock = threading.Lock()
        self._active_generations: Dict[str, GenerationProgress] = {}

        # Progress callback functions
        self._progress_callbacks: List[Callable[[str, GenerationProgress], None]] = []

        # Global progress bar for the current operation
        self._global_progress_bar: Optional[tqdm] = None

    def register_progress_callback(
        self, callback: Callable[[str, GenerationProgress], None]
    ):
        """
        Register a callback function to be called when progress updates.

        Args:
            callback: Function that receives (job_id, progress) arguments
        """
        self._progress_callbacks.append(callback)

    def create_generation_job(
        self,
        job_id: str,
        original_text: str,
        total_chunks: int,
        language_id: str,
        chunk_texts: Optional[List[str]] = None,
    ) -> GenerationProgress:
        """
        Create a new generation job and start tracking its progress.

        Args:
            job_id: Unique identifier for the job
            original_text: The original text being processed
            total_chunks: Number of chunks to process
            language_id: Language code for processing
            chunk_texts: Optional list of chunk texts for detailed tracking

        Returns:
            GenerationProgress object for the new job
        """
        with self._progress_lock:
            progress = GenerationProgress(
                job_id=job_id,
                original_text=original_text,
                total_chunks=total_chunks,
                language_id=language_id,
                start_time=time.time(),
            )

            # Fill in chunk text information if provided
            if chunk_texts and len(chunk_texts) == total_chunks:
                for i, chunk_text in enumerate(chunk_texts):
                    if i < len(progress.chunk_progress):
                        progress.chunk_progress[i].chunk_text = chunk_text
                        progress.chunk_progress[i].chunk_length = len(chunk_text)

            self._active_generations[job_id] = progress
            return progress

    def update_stage(self, job_id: str, stage: ProgressStage, description: str = ""):
        """
        Update the current stage of a generation job.

        Args:
            job_id: Job identifier
            stage: New stage
            description: Optional description for the stage
        """
        with self._progress_lock:
            if job_id not in self._active_generations:
                self.logger.warning(f"Job {job_id} not found for stage update")
                return

            progress = self._active_generations[job_id]
            progress.current_stage = stage

            self.logger.info(f"Job {job_id} stage: {stage.value} - {description}")
            self._notify_callbacks(job_id, progress)

    def start_chunk_processing(self, job_id: str, chunk_index: int):
        """
        Mark the start of processing for a specific chunk.

        Args:
            job_id: Job identifier
            chunk_index: Index of the chunk being processed
        """
        with self._progress_lock:
            if job_id not in self._active_generations:
                return

            progress = self._active_generations[job_id]
            if chunk_index < len(progress.chunk_progress):
                progress.chunk_progress[chunk_index].status = "processing"
                progress.chunk_progress[chunk_index].start_time = time.time()
                progress.chunk_progress[chunk_index].progress_percentage = 0.0

    def update_chunk_progress(self, job_id: str, chunk_index: int, percentage: float):
        """
        Update progress for a specific chunk.

        Args:
            job_id: Job identifier
            chunk_index: Index of the chunk
            percentage: Progress percentage (0.0 to 100.0)
        """
        with self._progress_lock:
            if job_id not in self._active_generations:
                return

            progress = self._active_generations[job_id]
            if chunk_index < len(progress.chunk_progress):
                progress.chunk_progress[chunk_index].progress_percentage = max(
                    0.0, min(100.0, percentage)
                )

                # Update overall progress
                completed_weight = sum(
                    chunk.progress_percentage for chunk in progress.chunk_progress
                ) / max(1, len(progress.chunk_progress))
                progress.overall_progress = completed_weight

                self._notify_callbacks(job_id, progress)

    def complete_chunk(self, job_id: str, chunk_index: int, success: bool = True):
        """
        Mark a chunk as completed.

        Args:
            job_id: Job identifier
            chunk_index: Index of the chunk
            success: Whether the chunk was processed successfully
        """
        with self._progress_lock:
            if job_id not in self._active_generations:
                return

            progress = self._active_generations[job_id]
            if chunk_index < len(progress.chunk_progress):
                chunk = progress.chunk_progress[chunk_index]
                chunk.status = "completed" if success else "failed"
                chunk.end_time = time.time()
                chunk.progress_percentage = 100.0 if success else 0.0

                # Update overall progress
                progress.overall_progress = (
                    progress.completed_chunks / progress.total_chunks * 100.0
                )

                # Estimate completion time
                if (
                    progress.completed_chunks > 0
                    and progress.total_chunks > progress.completed_chunks
                ):
                    avg_time_per_chunk = (
                        progress.elapsed_time / progress.completed_chunks
                    )
                    remaining_chunks = progress.total_chunks - progress.completed_chunks
                    progress.estimated_end_time = time.time() + (
                        avg_time_per_chunk * remaining_chunks
                    )

                self._notify_callbacks(job_id, progress)

    def complete_job(self, job_id: str, success: bool = True):
        """
        Mark a job as completed.

        Args:
            job_id: Job identifier
            success: Whether the job completed successfully
        """
        with self._progress_lock:
            if job_id not in self._active_generations:
                return

            progress = self._active_generations[job_id]
            progress.status = "completed" if success else "failed"
            progress.current_stage = (
                ProgressStage.COMPLETED if success else ProgressStage.ERROR
            )
            progress.overall_progress = 100.0 if success else 0.0

            # Set end times for any incomplete chunks
            for chunk in progress.chunk_progress:
                if chunk.status == "processing":
                    chunk.status = "failed"
                    chunk.end_time = time.time()
                    chunk.progress_percentage = 0.0

            self.logger.info(f"Job {job_id} {'completed' if success else 'failed'}")
            self._notify_callbacks(job_id, progress)

    def get_progress(self, job_id: str) -> Optional[GenerationProgress]:
        """
        Get current progress for a job.

        Args:
            job_id: Job identifier

        Returns:
            GenerationProgress object or None if job not found
        """
        with self._progress_lock:
            return self._active_generations.get(job_id)

    def cleanup_completed_jobs(self, max_age_hours: float = 24.0):
        """
        Clean up old completed jobs.

        Args:
            max_age_hours: Maximum age in hours before cleanup
        """
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)

        with self._progress_lock:
            completed_jobs = [
                job_id
                for job_id, progress in self._active_generations.items()
                if progress.status in ["completed", "failed"]
                and progress.start_time
                and progress.start_time < cutoff_time
            ]

            for job_id in completed_jobs:
                del self._active_generations[job_id]
                self.logger.info(f"Cleaned up old job: {job_id}")

    def _notify_callbacks(self, job_id: str, progress: GenerationProgress):
        """Notify all registered callbacks about progress update."""
        for callback in self._progress_callbacks:
            try:
                callback(job_id, progress)
            except Exception as e:
                self.logger.error(f"Error in progress callback: {e}")

    @contextmanager
    def global_progress_bar(self, total: int, desc: str = "Processing", **kwargs):
        """Context manager for global progress bar."""
        if not self.enable_tqdm:
            yield None
            return

        try:
            # Suppress tqdm warnings if needed
            with tqdm(total=total, desc=desc, **kwargs) as pbar:
                self._global_progress_bar = pbar
                yield pbar
        finally:
            self._global_progress_bar = None


class TTSProgressTracker(ProgressTracker):
    """
    Specialized progress tracker for TTS operations with enhanced features.

    This class extends ProgressTracker with TTS-specific progress tracking
    and integration with the main application.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)

        # Audio-specific progress tracking
        self._current_audio_job: Optional[str] = None
        self._audio_callbacks: List[Callable[[Dict], None]] = []

        # Performance metrics
        self._performance_history: List[Dict] = []

    def start_audio_generation(
        self,
        job_id: str,
        text: str,
        chunk_texts: List[str],
        language_id: str,
        config: Optional[Dict] = None,
    ) -> GenerationProgress:
        """
        Start tracking audio generation with detailed chunk information.

        Args:
            job_id: Unique job identifier
            text: Original text
            chunk_texts: List of chunk texts
            language_id: Language code
            config: Optional generation configuration

        Returns:
            GenerationProgress object
        """
        progress = self.create_generation_job(
            job_id=job_id,
            original_text=text,
            total_chunks=len(chunk_texts),
            language_id=language_id,
            chunk_texts=chunk_texts,
        )

        self._current_audio_job = job_id
        self.update_stage(job_id, ProgressStage.CHUNKING, "Text chunked successfully")

        return progress

    def track_model_loading(self, job_id: str, model_name: str):
        """Track model loading progress."""
        self.update_stage(job_id, ProgressStage.LOADING_MODEL, f"Loading {model_name}")

    def track_generation_start(self, job_id: str):
        """Track the start of audio generation."""
        self.update_stage(job_id, ProgressStage.GENERATING, "Generating audio chunks")

        if self.enable_tqdm:
            progress = self.get_progress(job_id)
            if progress:
                self.logger.info(
                    f"Starting generation for {progress.total_chunks} chunks"
                )

    def track_post_processing(self, job_id: str):
        """Track post-processing stage."""
        self.update_stage(
            job_id, ProgressStage.POST_PROCESSING, "Post-processing audio"
        )

    def track_concatenation(self, job_id: str, total_chunks: int):
        """Track audio concatenation stage."""
        self.update_stage(
            job_id,
            ProgressStage.CONCATENATING,
            f"Concatenating {total_chunks} audio chunks",
        )

    def get_detailed_progress(self, job_id: str) -> Optional[Dict]:
        """
        Get detailed progress information including performance metrics.

        Args:
            job_id: Job identifier

        Returns:
            Dictionary with detailed progress information
        """
        progress = self.get_progress(job_id)
        if not progress:
            return None

        return {
            "job_id": job_id,
            "status": progress.status,
            "stage": progress.current_stage.value,
            "overall_progress": progress.overall_progress,
            "total_chunks": progress.total_chunks,
            "completed_chunks": progress.completed_chunks,
            "failed_chunks": progress.failed_chunks,
            "elapsed_time": progress.elapsed_time,
            "estimated_time_remaining": progress.estimated_time_remaining,
            "chunks_per_second": progress.chunks_per_second,
            "language": progress.language_id,
            "chunk_details": [
                {
                    "index": chunk.chunk_index,
                    "text": chunk.chunk_text[:100] + "..."
                    if len(chunk.chunk_text) > 100
                    else chunk.chunk_text,
                    "length": chunk.chunk_length,
                    "status": chunk.status,
                    "progress": chunk.progress_percentage,
                    "duration": chunk.duration,
                }
                for chunk in progress.chunk_progress
            ],
        }

    def create_web_progress_callback(self) -> Callable[[str, GenerationProgress], None]:
        """
        Create a progress callback suitable for web interfaces.

        Returns:
            Callback function for web progress updates
        """

        def web_callback(job_id: str, progress: GenerationProgress):
            # This can be integrated with Gradio or other web frameworks
            web_info = self.get_detailed_progress(job_id)
            if web_info:
                # Log progress for web interface to pick up
                self.logger.info(f"WEB_PROGRESS: {web_info}")

        return web_callback

    def log_progress_summary(self, job_id: str):
        """Log a summary of progress for debugging."""
        progress = self.get_progress(job_id)
        if not progress:
            return

        summary = f"""
        Job {job_id} Progress Summary:
        - Status: {progress.status}
        - Stage: {progress.current_stage.value}
        - Overall Progress: {progress.overall_progress:.1f}%
        - Chunks: {progress.completed_chunks}/{progress.total_chunks} completed
        - Elapsed Time: {progress.elapsed_time:.1f}s
        - Estimated Remaining: {progress.estimated_time_remaining or 0:.1f}s
        - Processing Rate: {progress.chunks_per_second:.2f} chunks/s
        """

        self.logger.info(summary)


# Global progress tracker instance
_global_tracker: Optional[TTSProgressTracker] = None


def get_global_tracker() -> TTSProgressTracker:
    """Get the global progress tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TTSProgressTracker()
    return _global_tracker


def set_global_tracker(tracker: TTSProgressTracker):
    """Set the global progress tracker instance."""
    global _global_tracker
    _global_tracker = tracker


# Convenience functions
def create_progress_tracker(**kwargs) -> TTSProgressTracker:
    """Create a new progress tracker with the given configuration."""
    return TTSProgressTracker(**kwargs)


def track_generation_progress(
    job_id: str,
    text: str,
    chunk_texts: List[str],
    language_id: str,
    config: Optional[Dict] = None,
) -> GenerationProgress:
    """
    Convenience function to start tracking generation progress.

    Args:
        job_id: Unique job identifier
        text: Original text
        chunk_texts: List of chunk texts
        language_id: Language code
        config: Optional generation configuration

    Returns:
        GenerationProgress object
    """
    tracker = get_global_tracker()
    return tracker.start_audio_generation(
        job_id, text, chunk_texts, language_id, config
    )


# Example usage and testing
if __name__ == "__main__":
    import random

    # Create a progress tracker
    tracker = TTSProgressTracker(enable_tqdm=True, show_eta=True, show_rate=True)

    # Simulate a TTS generation job
    job_id = "test_job_001"
    original_text = (
        "This is a long text that needs to be chunked for TTS processing. " * 10
    )
    chunk_texts = [
        "This is chunk number " + str(i) + " with some content." for i in range(5)
    ]

    print("Starting TTS generation simulation...")

    # Start tracking
    progress = tracker.start_audio_generation(
        job_id=job_id, text=original_text, chunk_texts=chunk_texts, language_id="en"
    )

    # Simulate processing with progress updates
    for i in range(len(chunk_texts)):
        tracker.start_chunk_processing(job_id, i)

        # Simulate chunk processing with progress updates
        for progress_val in [25, 50, 75, 100]:
            time.sleep(0.1)  # Simulate work
            tracker.update_chunk_progress(job_id, i, progress_val)

        tracker.complete_chunk(job_id, i, success=True)

    # Complete the job
    tracker.complete_job(job_id, success=True)

    # Get final progress
    final_progress = tracker.get_detailed_progress(job_id)
    print(f"Final progress: {final_progress}")
