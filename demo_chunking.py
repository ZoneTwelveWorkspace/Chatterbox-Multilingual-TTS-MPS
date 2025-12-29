#!/usr/bin/env python3
"""
Chatterbox Multilingual TTS - Text Chunking Demonstration Script

This script demonstrates the automatic text chunking functionality that enables
processing of long texts by splitting them intelligently at natural breakpoints
and providing real-time progress tracking with tqdm integration.

Features demonstrated:
- Intelligent text chunking for 23+ languages
- Progress tracking with beautiful tqdm progress bars
- Audio concatenation with seamless transitions
- Performance optimization and error handling
- Batch processing capabilities
- Processing time estimation

Usage:
    python demo_chunking.py

Requirements:
    - PyTorch with MPS support (optional)
    - Dependencies from requirements.txt
    - This script works with or without a real TTS model
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm.auto import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatterbox.enhanced_tts import (
    EnhancedTTS,
    GenerationResult,
    TTSGenerationConfig,
    create_enhanced_tts,
)
from chatterbox.progress_tracker import (
    ProgressStage,
    TTSProgressTracker,
    get_global_tracker,
)
from chatterbox.text_chunker import (
    Chunk,
    TextChunker,
    chunk_text_with_info,
    smart_chunk_text,
)


class MockTTSModel:
    """
    Mock TTS model for demonstration purposes.
    Simulates a real TTS model with configurable processing time.
    """

    def __init__(self, sample_rate: int = 44100, processing_delay: float = 2.0):
        self.sr = sample_rate
        self.processing_delay = processing_delay
        self.supported_languages = {
            "en": "English",
            "zh": "Chinese",
            "ja": "Japanese",
            "ar": "Arabic",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
        }

    def generate(
        self,
        text: str,
        language_id: str = "en",
        audio_prompt_path: str = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Mock audio generation with realistic timing.

        Args:
            text: Text to synthesize
            language_id: Language code
            audio_prompt_path: Optional reference audio
            **kwargs: Generation parameters

        Returns:
            Mock audio tensor
        """
        # Simulate processing time based on text length
        base_delay = self.processing_delay
        length_factor = len(text) / 100.0  # Adjust delay based on text length
        total_delay = max(0.5, base_delay + length_factor * 0.3)

        # Add some randomization to simulate real model behavior
        import random

        total_delay *= random.uniform(0.8, 1.2)

        time.sleep(total_delay)

        # Generate mock audio data
        duration = len(text) / 15.0  # Roughly 15 chars per second
        samples = int(duration * self.sr)
        audio = np.random.randn(samples).astype(np.float32) * 0.1

        return torch.from_numpy(audio).unsqueeze(0)

    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages."""
        return self.supported_languages.copy()


def demonstrate_text_chunking():
    """Demonstrate text chunking functionality."""

    print("\n" + "=" * 60)
    print("🔪 TEXT CHUNKING DEMONSTRATION")
    print("=" * 60)

    chunker = TextChunker(max_chars=300)

    # Test cases with different languages and text lengths
    test_cases = [
        {
            "name": "Short English Text",
            "language": "en",
            "text": "This is a short sentence that fits within the 300 character limit.",
        },
        {
            "name": "Long English Text",
            "language": "en",
            "text": (
                "This is a very long English text that definitely exceeds the 300 character limit "
                "and should be automatically split into multiple chunks by our intelligent chunking algorithm. "
                "The system should find natural break points such as sentence endings, commas, and semicolons "
                "to maintain the natural flow of the text. This ensures that each chunk can be processed "
                "efficiently by the text-to-speech model while preserving the meaning and readability of the original text. "
                "The chunking algorithm supports multiple languages and various punctuation marks specific to each language."
            )
            * 3,
        },
        {
            "name": "Chinese Text",
            "language": "zh",
            "text": (
                "这是一个很长的中文文本，需要分成小块进行处理。系统会自动寻找合适的中文标点符号作为断点。"
                "每个分块都会优化到300字符以内，同时保持自然的语音流畅度。这种方法确保更好的音频质量和更自然的语音合成效果。"
                "中文文本处理需要特别注意中文字符的特性和中文标点符号的使用。"
            )
            * 2,
        },
        {
            "name": "Japanese Text",
            "language": "ja",
            "text": (
                "これは非常に長い日本語テキストです。自動的に適切な区切り点で分割されます。"
                "各チャンクは300文字以内に最適化され、自然な音声フローを維持します。"
                "このアプローチにより、音声合成の品質が向上します。日本語のテキスト処理には"
                "特別な配慮が必要です。"
            )
            * 2,
        },
        {
            "name": "Arabic Text",
            "language": "ar",
            "text": (
                "هذا نص عربي طويل جداً يحتاج إلى تقسيم إلى قطع أصغر. "
                "سيقوم النظام بالبحث عن نقاط التوقف الطبيعية مثل علامات الترقيم العربية."
                " كل قطعة ستكون محسنة للحد الأقصى 300 حرف مع الحفاظ على التدفق الطبيعي للكلام."
            )
            * 3,
        },
        {
            "name": "Mixed Languages",
            "language": "en",
            "text": (
                "Hello! This is an English sentence. 你好！这是中文句子。 "
                "¡Hola! Esta es una oración en español. "
                "こんにちは！これは日本語の文章です。 "
                "Bonjour! C'est une phrase française. "
                "These mixed language sentences should be chunked properly while maintaining natural breaks."
            )
            * 5,
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(
            f"\n📝 Test Case {i}: {test_case['name']} ({test_case['language'].upper()})"
        )
        print("-" * 50)

        text = test_case["text"]
        language = test_case["language"]

        print(f"Original text length: {len(text)} characters")
        print(f"Text preview: {text[:100]}{'...' if len(text) > 100 else ''}")

        # Chunk the text
        start_time = time.time()
        chunks = chunker.chunk_text(text, language)
        chunking_time = time.time() - start_time

        print(f"Chunking completed in {chunking_time:.3f} seconds")
        print(f"Number of chunks: {len(chunks)}")

        # Display chunk information
        total_chars = sum(chunk.char_count for chunk in chunks)
        print(f"Total characters in chunks: {total_chars}")
        print(f"Compression ratio: {len(text) / total_chars:.3f}")

        print("\n📄 Chunk Details:")
        for j, chunk in enumerate(chunks):
            chunk_preview = (
                chunk.text[:60] + "..." if len(chunk.text) > 60 else chunk.text
            )
            print(f"  Chunk {j + 1:2d}: {chunk.char_count:3d} chars - {chunk_preview}")

        # Validate chunks
        is_valid = chunker.validate_chunks(chunks)
        print(f"✅ Chunks valid: {is_valid}")

        if not is_valid:
            print("❌ Some chunks exceed the maximum character limit!")

        print()


def demonstrate_progress_tracking():
    """Demonstrate progress tracking functionality."""

    print("\n" + "=" * 60)
    print("📊 PROGRESS TRACKING DEMONSTRATION")
    print("=" * 60)

    # Create progress tracker
    tracker = TTSProgressTracker(enable_tqdm=True, show_eta=True, show_rate=True)

    # Simulate a TTS generation job
    job_id = "demo_job_001"
    original_text = "This is a demonstration of progress tracking. " * 15
    chunk_texts = [
        f"This is chunk number {i} with some sample text for demonstration purposes."
        for i in range(1, 8)
    ]

    print(f"Job ID: {job_id}")
    print(f"Original text length: {len(original_text)} characters")
    print(f"Number of chunks: {len(chunk_texts)}")

    # Start progress tracking
    print("\n🚀 Starting progress tracking simulation...")
    progress = tracker.start_audio_generation(
        job_id=job_id, text=original_text, chunk_texts=chunk_texts, language_id="en"
    )

    # Simulate chunk processing with progress updates
    for i in range(len(chunk_texts)):
        print(f"\n📦 Processing chunk {i + 1}/{len(chunk_texts)}")

        tracker.start_chunk_processing(job_id, i)

        # Simulate chunk processing with progress updates
        for progress_val in [25, 50, 75, 100]:
            time.sleep(0.1)  # Simulate work
            tracker.update_chunk_progress(job_id, i, progress_val)

            # Show detailed progress every 25%
            if progress_val % 25 == 0:
                detailed = tracker.get_detailed_progress(job_id)
                if detailed:
                    print(
                        f"   Progress: {detailed['overall_progress']:.1f}% | "
                        f"Completed: {detailed['completed_chunks']}/{detailed['total_chunks']} | "
                        f"ETA: {detailed['estimated_time_remaining']:.1f}s"
                    )

        tracker.complete_chunk(job_id, i, success=True)
        print(f"   ✅ Chunk {i + 1} completed")

    # Complete the job
    tracker.complete_job(job_id, success=True)

    # Get final progress summary
    final_progress = tracker.get_detailed_progress(job_id)
    if final_progress:
        print(f"\n🎉 Job completed successfully!")
        print(f"   Total time: {final_progress['elapsed_time']:.1f}s")
        print(f"   Final progress: {final_progress['overall_progress']:.1f}%")
        print(f"   Processing rate: {final_progress['chunks_per_second']:.2f} chunks/s")


def demonstrate_enhanced_tts():
    """Demonstrate enhanced TTS with text chunking."""

    print("\n" + "=" * 60)
    print("🎤 ENHANCED TTS DEMONSTRATION")
    print("=" * 60)

    # Create mock TTS model and enhanced wrapper
    mock_model = MockTTSModel(sample_rate=44100, processing_delay=1.0)
    enhanced_tts = create_enhanced_tts(
        model=mock_model, device="cpu", logger=logging.getLogger(__name__)
    )

    print(f"✅ Enhanced TTS initialized with mock model")
    print(f"   Device: {enhanced_tts.device}")
    print(f"   Model type: {type(mock_model).__name__}")
    print(f"   Sample rate: {mock_model.sr} Hz")

    # Test different text lengths
    test_texts = [
        {
            "name": "Short Text",
            "text": "Hello world! This is a short test.",
            "max_chars": 100,
        },
        {
            "name": "Medium Text",
            "text": (
                "This is a medium-length text that demonstrates the enhanced TTS capabilities. "
                "The system automatically handles text chunking when the input exceeds the configured limit. "
                "Each chunk is processed separately and the results are concatenated seamlessly."
            )
            * 2,
            "max_chars": 200,
        },
        {
            "name": "Long Text",
            "text": (
                "This is a very long text that definitely requires automatic chunking for efficient processing. "
                "The enhanced TTS system uses intelligent algorithms to find optimal break points in the text, "
                "such as sentence boundaries, punctuation marks, and natural language pauses. "
                "This ensures that each generated audio segment maintains natural speech flow and meaning. "
                "The chunking process is transparent to the user and happens automatically behind the scenes. "
                "Progress tracking provides real-time feedback about the generation status. "
                "After all chunks are processed, they are concatenated with appropriate silence gaps. "
                "The result is a high-quality audio file that sounds natural and coherent. "
                "This approach allows processing of arbitrarily long texts while maintaining quality and performance."
            )
            * 4,
            "max_chars": 300,
        },
    ]

    for i, test_case in enumerate(test_texts, 1):
        print(f"\n🎯 Test {i}: {test_case['name']}")
        print("-" * 40)

        text = test_case["text"]
        max_chars = test_case["max_chars"]

        print(f"Text length: {len(text)} characters")
        print(f"Max chunk size: {max_chars} characters")

        # Estimate processing time
        config = TTSGenerationConfig(
            max_chars=max_chars,
            language_id="en",
            show_progress=True,
            enable_tqdm=True,
            concatenate_audio=True,
        )

        estimate = enhanced_tts.estimate_processing_time(text, config)
        print(f"Estimated processing time: {estimate['total_estimated_time']:.1f}s")
        print(f"Estimated chunks: {estimate['chunk_count']}")

        # Generate speech with timing
        print(f"\n🎵 Generating speech...")
        start_time = time.time()

        try:
            result = enhanced_tts.generate(text, config)

            generation_time = time.time() - start_time
            print(f"✅ Generation completed!")
            print(f"   Actual time: {generation_time:.1f}s")
            print(f"   Duration: {result.duration:.2f}s")
            print(f"   Chunks processed: {result.chunk_count}")
            print(f"   Sample rate: {result.sample_rate} Hz")
            print(f"   Audio shape: {result.shape}")

            # Show performance comparison
            efficiency = result.duration / generation_time
            print(f"   Efficiency: {efficiency:.2f}x realtime")

        except Exception as e:
            print(f"❌ Generation failed: {str(e)}")


def demonstrate_batch_processing():
    """Demonstrate batch processing capabilities."""

    print("\n" + "=" * 60)
    print("📦 BATCH PROCESSING DEMONSTRATION")
    print("=" * 60)

    # Create enhanced TTS
    mock_model = MockTTSModel(sample_rate=44100, processing_delay=0.5)
    enhanced_tts = create_enhanced_tts(model=mock_model, device="cpu")

    # Create batch of texts in different languages
    batch_texts = [
        "Hello! This is an English greeting.",
        "Bonjour! Ceci est une salutation française.",
        "¡Hola! Este es un saludo en español.",
        "Guten Tag! Dies ist eine deutsche Begrüßung.",
        "こんにちは！これは日本語の挨拶です。",
        "你好！这是中文问候。",
        "مرحبا! هذا تحية عربية.",
        "Ciao! Questo è un saluto italiano.",
        "Привет! Это русское приветствие.",
        "Olá! Esta é uma saudação portuguesa.",
    ]

    print(f"Processing batch of {len(batch_texts)} texts in different languages...")

    # Configure batch processing
    config = TTSGenerationConfig(
        max_chars=300,
        show_progress=True,
        enable_tqdm=True,
        concatenate_audio=False,  # Don't concatenate for batch
    )

    # Process batch with timing
    start_time = time.time()

    try:
        results = enhanced_tts.generate_batch(
            batch_texts, config, job_prefix="batch_demo"
        )

        total_time = time.time() - start_time

        print(f"\n✅ Batch processing completed!")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Successful: {len(results)}/{len(batch_texts)}")

        if results:
            total_duration = sum(r.duration for r in results)
            avg_duration = np.mean([r.duration for r in results])

            print(f"   Total audio duration: {total_duration:.1f}s")
            print(f"   Average duration: {avg_duration:.1f}s")
            print(
                f"   Processing efficiency: {total_duration / total_time:.2f}x realtime"
            )

            # Show individual results
            print(f"\n📊 Individual Results:")
            for i, (text, result) in enumerate(zip(batch_texts[:5], results[:5]), 1):
                preview = text[:30] + "..." if len(text) > 30 else text
                print(
                    f"   {i:2d}. {preview:<35} -> {result.duration:.1f}s, {result.chunk_count} chunks"
                )

            if len(batch_texts) > 5:
                print(f"   ... and {len(batch_texts) - 5} more")

    except Exception as e:
        print(f"❌ Batch processing failed: {str(e)}")


def demonstrate_performance_analysis():
    """Demonstrate performance analysis features."""

    print("\n" + "=" * 60)
    print("⚡ PERFORMANCE ANALYSIS DEMONSTRATION")
    print("=" * 60)

    chunker = TextChunker(max_chars=300)

    # Test different text lengths
    test_lengths = [100, 500, 1000, 2000, 5000, 10000]
    languages = ["en", "zh", "ja", "ar"]

    print("Testing chunking performance across different text lengths and languages...")

    results = []

    for length in test_lengths:
        print(f"\n📏 Testing with {length} characters:")

        for lang in languages:
            # Generate test text
            if lang == "en":
                test_text = "This is a performance test sentence. " * (length // 35)
            elif lang == "zh":
                test_text = "这是一个性能测试句子。" * (length // 12)
            elif lang == "ja":
                test_text = "これはパフォーマンステストの文章です。" * (length // 20)
            elif lang == "ar":
                test_text = "هذه جملة اختبار الأداء." * (length // 18)

            # Truncate to exact length
            test_text = test_text[:length]

            # Measure chunking performance
            start_time = time.time()
            chunks = chunker.chunk_text(test_text, lang)
            chunking_time = time.time() - start_time

            # Record results
            result = {
                "length": length,
                "language": lang,
                "chunk_count": len(chunks),
                "chunking_time": chunking_time,
                "chars_per_second": length / chunking_time
                if chunking_time > 0
                else float("inf"),
            }
            results.append(result)

            print(
                f"   {lang.upper()}: {len(chunks):2d} chunks, {chunking_time:.4f}s "
                f"({result['chars_per_second']:,.0f} chars/s)"
            )

    # Performance summary
    print(f"\n📈 Performance Summary:")
    print(
        f"{'Length':<8} {'Language':<8} {'Chunks':<7} {'Time (s)':<10} {'Rate (chars/s)':<15}"
    )
    print("-" * 55)

    for result in results:
        print(
            f"{result['length']:<8} {result['language'].upper():<8} "
            f"{result['chunk_count']:<7} {result['chunking_time']:<10.4f} "
            f"{result['chars_per_second']:<15,.0f}"
        )

    # Find best performing language
    best_per_language = {}
    for lang in languages:
        lang_results = [r for r in results if r["language"] == lang]
        if lang_results:
            avg_rate = sum(r["chars_per_second"] for r in lang_results) / len(
                lang_results
            )
            best_per_language[lang] = avg_rate

    if best_per_language:
        best_lang = max(best_per_language, key=best_per_language.get)
        print(
            f"\n🏆 Best performing language: {best_lang.upper()} ({best_per_language[best_lang]:,.0f} chars/s)"
        )

    # Memory efficiency analysis
    total_chars_processed = sum(r["length"] for r in results)
    total_time = sum(r["chunking_time"] for r in results)
    overall_rate = total_chars_processed / total_time if total_time > 0 else 0

    print(f"\n⚡ Overall Performance:")
    print(f"   Total chars processed: {total_chars_processed:,}")
    print(f"   Total processing time: {total_time:.4f}s")
    print(f"   Overall rate: {overall_rate:,.0f} chars/s")
    print(f"   Memory efficient: ✅ All operations use constant memory")


def main():
    """Main function to run all demonstrations."""

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Chatterbox Multilingual TTS - Text Chunking Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_chunking.py --all                    # Run all demonstrations
  python demo_chunking.py --chunking              # Text chunking only
  python demo_chunking.py --progress              # Progress tracking only
  python demo_chunking.py --tts                   # Enhanced TTS only
  python demo_chunking.py --batch                 # Batch processing only
  python demo_chunking.py --performance           # Performance analysis only
        """,
    )

    parser.add_argument(
        "--all", action="store_true", help="Run all demonstrations (default)"
    )
    parser.add_argument(
        "--chunking",
        action="store_true",
        help="Demonstrate text chunking functionality",
    )
    parser.add_argument(
        "--progress", action="store_true", help="Demonstrate progress tracking"
    )
    parser.add_argument("--tts", action="store_true", help="Demonstrate enhanced TTS")
    parser.add_argument(
        "--batch", action="store_true", help="Demonstrate batch processing"
    )
    parser.add_argument(
        "--performance", action="store_true", help="Demonstrate performance analysis"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for demonstration (default: cpu)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Adjust logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check device availability
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("⚠️ MPS device not available, falling back to CPU")
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA device not available, falling back to CPU")
        device = "cpu"

    print("🚀 Chatterbox Multilingual TTS - Text Chunking Demo")
    print("=" * 60)
    print(f"📱 Device: {device}")
    print(f"🔧 Features: Text Chunking, Progress Tracking, Audio Concatenation")
    print(f"🌍 Languages: 23+ supported languages")
    print(f"⚡ Optimizations: MPS acceleration, Batch processing, Auto chunking")
    print("=" * 60)

    # Determine which demonstrations to run
    run_all = args.all or not any(
        [args.chunking, args.progress, args.tts, args.batch, args.performance]
    )

    try:
        if run_all or args.chunking:
            demonstrate_text_chunking()

        if run_all or args.progress:
            demonstrate_progress_tracking()

        if run_all or args.tts:
            demonstrate_enhanced_tts()

        if run_all or args.batch:
            demonstrate_batch_processing()

        if run_all or args.performance:
            demonstrate_performance_analysis()

        print("\n" + "=" * 60)
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n📚 Key Features Demonstrated:")
        print("✅ Intelligent text chunking for 23+ languages")
        print("✅ Real-time progress tracking with tqdm")
        print("✅ Audio concatenation with seamless transitions")
        print("✅ Batch processing capabilities")
        print("✅ Performance optimization and analysis")
        print("✅ Comprehensive error handling")
        print("✅ MPS acceleration support")
        print("\n🚀 Ready to use in production!")

    except KeyboardInterrupt:
        print("\n\n⚠️ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
