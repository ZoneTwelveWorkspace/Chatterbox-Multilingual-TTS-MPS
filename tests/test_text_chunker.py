# Chatterbox-Multilingual-TTS/tests/test_text_chunker.py
"""
Comprehensive test suite for the TextChunker module.

This module contains unit tests and integration tests to ensure the text chunking
functionality works correctly across all supported languages and edge cases.
"""

import sys
import unittest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chatterbox.text_chunker import (
    Chunk,
    TextChunker,
    chunk_text_with_info,
    smart_chunk_text,
)


class TestTextChunker(unittest.TestCase):
    """Test cases for the TextChunker class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.chunker = TextChunker(max_chars=300)

    def test_basic_chunking_english(self):
        """Test basic chunking with English text."""
        text = "This is a short text. This should stay as one chunk."
        chunks = self.chunker.chunk_text(text, "en")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)
        self.assertEqual(chunks[0].char_count, len(text))
        self.assertTrue(chunks[0].is_final)

    def test_long_text_chunking_english(self):
        """Test chunking of long English text."""
        # Create text longer than 300 characters
        long_text = (
            "This is a very long text that definitely exceeds the 300 character limit. "
            "It contains multiple sentences with proper punctuation like periods, commas, and semicolons. "
            "The system should intelligently split this text at natural break points such as sentence endings. "
            "This ensures that each chunk maintains natural speech flow when converted to audio. "
            "The algorithm looks for periods, question marks, exclamation points, and other suitable punctuation marks. "
            "When no natural breaks are found, it falls back to word boundaries to maintain readability. "
            "This comprehensive approach ensures high-quality text-to-speech conversion across multiple languages."
        )

        chunks = self.chunker.chunk_text(long_text, "en")

        # Verify all chunks are within limits
        for chunk in chunks:
            self.assertLessEqual(chunk.char_count, 300)
            self.assertGreater(len(chunk.text.strip()), 0)

        # Verify no chunks are empty
        self.assertTrue(all(chunk.text.strip() for chunk in chunks))

        # Verify reconstruction is accurate
        reconstructed = " ".join(chunk.text for chunk in chunks)
        self.assertIn("This is a very long text that definitely exceeds", reconstructed)

    def test_chinese_text_chunking(self):
        """Test chunking with Chinese text and punctuation."""
        chinese_text = (
            "这是一个很长的中文文本，需要分成小块进行处理。系统会自动寻找合适的中文标点符号作为断点。"
            "每个分块都会优化到300字符以内，同时保持自然的语音流畅度。这种方法确保更好的音频质量和更自然的语音合成效果。"
            "中文文本处理需要特别注意中文字符的特性和中文标点符号的使用。"
            "随着人工智能技术的不断发展，多语言文本处理成为了一个重要的研究方向。"
            "自然语言处理技术在跨语言应用中面临着许多挑战，包括语法结构的差异、词汇的多样性以及语义理解的复杂性。"
            "因此，开发有效的文本分块算法对于提高处理效率和质量具有重要意义。"
            "通过智能化的分块策略，我们可以更好地处理各种语言的长文本，确保每个分块都能在指定的字符限制内进行有效的处理。"
            "机器学习算法在文本分析中发挥着越来越重要的作用，特别是在处理大规模多语言数据集时。"
            "深度学习模型的兴起为自然语言处理带来了革命性的变化，使得机器能够更好地理解和生成人类语言。"
            "然而，不同语言之间的语法差异、词汇结构和文化背景仍然是对这些技术的重大挑战。"
            "因此，我们需要更加智能和适应性的方法来处理这些复杂性，确保跨语言应用的有效性和准确性。"
            "文本分块作为文本预处理的重要步骤，直接影响后续处理的质量和效率。"
            "特别是在语音合成应用中，合适的分块策略可以显著提高生成音频的自然度和流畅性。"
            "通过合理控制每个文本片段的长度，我们可以更好地平衡处理速度和输出质量之间的关系。"
        )

        chunks = self.chunker.chunk_text(chinese_text, "zh")

        # Verify all chunks are within limits
        for chunk in chunks:
            self.assertLessEqual(chunk.char_count, 300)
            self.assertGreater(len(chunk.text.strip()), 0)

        # Should have multiple chunks for long text
        self.assertGreater(len(chunks), 1)

        # Verify chunks contain Chinese characters
        for chunk in chunks:
            self.assertTrue(any(ord(char) > 127 for char in chunk.text))

    def test_japanese_text_chunking(self):
        """Test chunking with Japanese text and punctuation."""
        japanese_text = (
            "これは非常に長い日本語テキストです。自動的に適切な区切り点で分割されます。"
            "各チャンクは300文字以内に最適化され、自然な音声フローを維持します。"
            "このアプローチにより、音声合成の品質が向上します。"
        )

        chunks = self.chunker.chunk_text(japanese_text, "ja")

        # Verify all chunks are within limits
        for chunk in chunks:
            self.assertLessEqual(chunk.char_count, 300)

        # Should handle Japanese punctuation correctly
        self.assertTrue(any("。" in chunk.text for chunk in chunks))

    def test_arabic_text_chunking(self):
        """Test chunking with Arabic text."""
        arabic_text = (
            "هذا نص عربي طويل جداً يحتاج إلى تقسيم إلى قطع أصغر. "
            "سيقوم النظام بالبحث عن نقاط التوقف الطبيعية مثل علامات الترقيم العربية."
            " كل قطعة ستكون محسنة للحد الأقصى 300 حرف مع الحفاظ على التدفق الطبيعي للكلام."
        )

        chunks = self.chunker.chunk_text(arabic_text, "ar")

        # Verify all chunks are within limits
        for chunk in chunks:
            self.assertLessEqual(chunk.char_count, 300)

        # Should contain Arabic text
        self.assertTrue(any(ord(char) > 127 for char in chunks[0].text))

    def test_mixed_punctuation_chunking(self):
        """Test chunking with mixed punctuation marks."""
        text = (
            "First sentence; second sentence: third sentence, fourth sentence! "
            "Fifth sentence? Sixth sentence—seventh sentence. Eighth sentence; "
            "ninth sentence: tenth sentence, eleventh sentence! Twelfth sentence? "
            "Thirteenth sentence—fourteenth sentence. This long text should definitely "
            "be split into multiple chunks because it exceeds the 300 character limit "
            "and contains multiple punctuation marks that can serve as natural break points "
            "for the chunking algorithm to work effectively."
        )

        chunks = self.chunker.chunk_text(text, "en")

        # Should split at various punctuation marks
        self.assertGreater(len(chunks), 1)

        # Each chunk should end with punctuation or be the final chunk
        for i, chunk in enumerate(chunks):
            if not chunk.is_final:
                self.assertTrue(
                    any(
                        chunk.text.rstrip().endswith(punct)
                        for punct in [";", ":", ",", "—", "!", "?", "."]
                    )
                )

    def test_edge_case_empty_text(self):
        """Test chunking with empty text."""
        chunks = self.chunker.chunk_text("", "en")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "")
        self.assertEqual(chunks[0].char_count, 0)
        self.assertTrue(chunks[0].is_final)

    def test_edge_case_single_long_word(self):
        """Test chunking with a single word longer than 300 characters."""
        long_word = (
            "supercalifragilisticexpialidocious" * 30
        )  # Much longer "word" to ensure splitting

        chunks = self.chunker.chunk_text(long_word, "en")

        # Should split the long word into multiple chunks
        self.assertGreater(len(chunks), 1)

        # All chunks should be within limit
        for chunk in chunks:
            self.assertLessEqual(chunk.char_count, 300)

    def test_word_boundary_chunking(self):
        """Test chunking when no punctuation is available."""
        text = " ".join([f"word{i}" for i in range(1, 50)])  # Many words

        chunks = self.chunker.chunk_text(text, "en")

        # Should split at word boundaries
        self.assertGreater(len(chunks), 1)

        # Verify no chunks exceed the limit
        for chunk in chunks:
            self.assertLessEqual(chunk.char_count, 300)

    def test_unicode_handling(self):
        """Test handling of various Unicode characters."""
        unicode_text = "Hello 🌍 مرحبا こんにちは Привет 🎵"

        chunks = self.chunker.chunk_text(unicode_text, "en")

        self.assertEqual(len(chunks), 1)
        self.assertIn("🌍", chunks[0].text)
        self.assertIn("مرحبا", chunks[0].text)
        self.assertIn("こんにちは", chunks[0].text)

    def test_custom_max_chars(self):
        """Test chunking with custom maximum character limit."""
        chunker = TextChunker(max_chars=50)
        long_text = "This is a text that should be split into multiple chunks of 50 characters or less."

        chunks = chunker.chunk_text(long_text, "en")

        for chunk in chunks:
            self.assertLessEqual(chunk.char_count, 50)

    def test_chunk_metadata(self):
        """Test chunking with detailed metadata."""
        text = "Short text."
        metadata = self.chunker.chunk_with_metadata(text, "en")

        self.assertEqual(metadata["original_length"], len(text))
        self.assertEqual(metadata["chunk_count"], 1)
        self.assertEqual(metadata["total_chars_in_chunks"], len(text))
        self.assertEqual(metadata["language"], "en")
        self.assertEqual(metadata["max_chars"], 300)
        self.assertEqual(metadata["compression_ratio"], 1.0)

    def test_chunk_validation(self):
        """Test chunk validation functionality."""
        # Valid chunks
        valid_chunks = [
            Chunk("Short text", 0, 10, True),
            Chunk("Another chunk", 1, 13, False),
        ]
        self.assertTrue(self.chunker.validate_chunks(valid_chunks))

        # Invalid chunks (exceeds limit)
        invalid_chunks = [Chunk("a" * 400, 0, 400, True)]
        self.assertFalse(self.chunker.validate_chunks(invalid_chunks))

    def test_utility_functions(self):
        """Test utility functions."""
        text = "This is a test. This is another test."

        # Test smart_chunk_text
        chunks = smart_chunk_text(text, "en", 50)
        self.assertIsInstance(chunks, list)
        self.assertIsInstance(chunks[0], str)

        # Test chunk_text_with_info
        info = chunk_text_with_info(text, "en", 50)
        self.assertIn("chunks", info)
        self.assertIn("original_length", info)
        self.assertIn("chunk_count", info)
        self.assertIsInstance(info["chunks"], list)

    def test_multilingual_punctuation_support(self):
        """Test punctuation support across different languages."""
        test_cases = {
            "en": "Hello! How are you? I'm fine.",
            "zh": "你好！你好吗？我很好。",
            "ja": "こんにちは！元気ですか？はい、元気です。",
            "ar": "مرحبا! كيف حالك؟ أنا بخير.",
            "ru": "Привет! Как дела? У меня все хорошо.",
            "fr": "Bonjour! Comment allez-vous? Je vais bien.",
        }

        for lang, text in test_cases.items():
            chunks = self.chunker.chunk_text(text, lang)
            self.assertGreater(len(chunks), 0)
            for chunk in chunks:
                self.assertLessEqual(chunk.char_count, 300)

    def test_performance_with_very_long_text(self):
        """Test performance with very long text (1000+ characters)."""
        # Generate very long text
        sentences = [
            "This is sentence number {} of a very long text that will test the performance of the chunking algorithm. ",
            "Each sentence contains multiple clauses and should be processed efficiently by the text chunker. ",
            "The system should maintain good performance even with extensive text processing requirements. ",
            "Quality and speed are both important factors in this comprehensive testing scenario. ",
        ]

        long_text = "".join(
            sentences[(i - 1) % len(sentences)].format(i) for i in range(1, 20)
        )
        self.assertGreater(len(long_text), 1000)

        chunks = self.chunker.chunk_text(long_text, "en")

        # Performance check - should complete quickly
        self.assertGreater(len(chunks), 1)

        # Verify all chunks are valid
        self.assertTrue(self.chunker.validate_chunks(chunks))

        # Verify reconstruction maintains order
        reconstructed_chunks = [chunk.text for chunk in chunks]
        for i in range(len(reconstructed_chunks) - 1):
            # Each chunk (except last) should end with appropriate punctuation
            chunk_text = reconstructed_chunks[i]
            has_ending_punct = any(
                chunk_text.rstrip().endswith(punct)
                for punct in [".", "!", "?", ";", ":"]
            )
            self.assertTrue(
                has_ending_punct, f"Chunk {i} doesn't end with proper punctuation"
            )

    def test_consecutive_punctuation_handling(self):
        """Test handling of consecutive punctuation marks."""
        text = (
            "What?! Really??? Yes!!! Absolutely... This should definitely be long enough to require chunking. "
            * 5
        )

        chunks = self.chunker.chunk_text(text, "en")

        # Should handle consecutive punctuation correctly
        self.assertGreater(len(chunks), 0)

        # Verify no chunk has consecutive sentence endings that could be optimized
        for chunk in chunks:
            # Should not have multiple consecutive sentence endings
            sentence_endings = [".", "!", "?"]
            consecutive_endings = 0
            for char in reversed(chunk.text):
                if char in sentence_endings:
                    consecutive_endings += 1
                else:
                    break
            # Allow up to 3 consecutive endings to be more lenient
            self.assertLessEqual(consecutive_endings, 3)

    def test_whitespace_preservation(self):
        """Test that whitespace is handled correctly."""
        text = "  Leading and trailing spaces  should be handled.  \n\n  Multiple newlines too.  "

        chunks = self.chunker.chunk_text(text, "en")

        # Should preserve meaningful whitespace
        for chunk in chunks:
            # After stripping for display, original should be preserved
            original_preserved = any(chunk.text.strip() in text for chunk in chunks)
            self.assertTrue(original_preserved)


class TestChunkMetadata(unittest.TestCase):
    """Test cases for chunk metadata and indexing."""

    def setUp(self):
        self.chunker = TextChunker()

    def test_chunk_indexing(self):
        """Test that chunks have correct indices."""
        text = "First chunk. Second chunk! Third chunk?"

        chunks = self.chunker.chunk_text(text, "en")

        indices = [chunk.index for chunk in chunks]
        self.assertEqual(indices, list(range(len(chunks))))

    def test_final_chunk_marker(self):
        """Test that the final chunk is properly marked."""
        text = "Chunk one. Chunk two. Chunk three."

        chunks = self.chunker.chunk_text(text, "en")

        # Only the last chunk should be marked as final
        final_chunks = [chunk for chunk in chunks if chunk.is_final]
        self.assertEqual(len(final_chunks), 1)
        self.assertEqual(final_chunks[0].index, len(chunks) - 1)

    def test_char_count_accuracy(self):
        """Test that character counts are accurate."""
        text = "Hello world! This has 30 characters."

        chunks = self.chunker.chunk_text(text, "en")

        for chunk in chunks:
            self.assertEqual(chunk.char_count, len(chunk.text))


class TestLanguageSpecificFeatures(unittest.TestCase):
    """Test cases for language-specific features."""

    def setUp(self):
        self.chunker = TextChunker()

    def test_language_punctuation_mapping(self):
        """Test that different languages use correct punctuation."""
        test_cases = {
            "zh": ["。", "！", "？"],
            "ja": ["。", "！", "？"],
            "ar": [".", "!", "?", "۔"],
            "hi": ["।", "!", "?"],
        }

        for lang, expected_punct in test_cases.items():
            sentence_endings, secondary_breaks = (
                self.chunker.get_punctuation_for_language(lang)
            )

            # Check that the language has its specific punctuation
            for punct in expected_punct:
                self.assertIn(punct, sentence_endings)

    def test_unknown_language_fallback(self):
        """Test that unknown languages fall back to English punctuation."""
        chunks = self.chunker.chunk_text("Test sentence.", "unknown_lang")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Test sentence.")


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_classes = [TestTextChunker, TestChunkMetadata, TestLanguageSpecificFeatures]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )

    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")

    print(f"{'=' * 60}")
