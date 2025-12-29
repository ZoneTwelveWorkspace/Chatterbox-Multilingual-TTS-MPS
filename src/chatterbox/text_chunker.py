"""
Intelligent text chunking module for Chatterbox Multilingual TTS.

This module provides smart text chunking functionality that splits long texts
into optimal chunks for audio generation while respecting the 300-character limit
and maintaining natural speech flow.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    text: str
    index: int
    char_count: int
    is_final: bool = False


class TextChunker:
    """
    Intelligent text chunker for multilingual text-to-speech.

    Features:
    - Respects 300-character limit
    - Splits at natural breakpoints (sentence boundaries)
    - Supports 23+ languages with appropriate punctuation
    - Maximizes chunk size without exceeding limits
    - Preserves text flow and meaning
    """

    def __init__(self, max_chars: int = 300):
        """
        Initialize the text chunker.

        Args:
            max_chars: Maximum characters per chunk (default: 300)
        """
        self.max_chars = max_chars

        # Define sentence-ending punctuation for different languages
        self.sentence_endings = {
            # Major languages and their punctuation
            "en": [".", "!", "?"],
            "es": [".", "!", "?"],
            "fr": [".", "!", "?"],
            "it": [".", "!", "?"],
            "pt": [".", "!", "?"],
            "de": [".", "!", "?"],
            "ru": [".", "!", "?"],
            "pl": [".", "!", "?"],
            "nl": [".", "!", "?"],
            "sv": [".", "!", "?"],
            "da": [".", "!", "?"],
            "no": [".", "!", "?"],
            "fi": [".", "!", "?"],
            "tr": [".", "!", "?"],
            "ar": [".", "!", "?", "۔"],  # Arabic sentence ending
            "he": [".", "!", "?"],
            "zh": ["。", "！", "？"],  # Chinese
            "ja": ["。", "！", "？"],  # Japanese
            "ko": [".", "!", "?"],  # Korean
            "hi": ["।", "!", "?"],  # Hindi (Devanagari period)
            "th": [".", "!", "?"],
            "vi": [".", "!", "?"],
            "el": [".", "!", "?"],  # Greek
            "sw": [".", "!", "?"],  # Swahili
            "ms": [".", "!", "?"],  # Malay
        }

        # Secondary breakpoints (within sentences)
        self.secondary_breaks = {
            "en": [";", ":", ",", "—", "–"],
            "es": [";", ":", ",", "—", "–"],
            "fr": [";", ":", ",", "—", "–"],
            "it": [";", ":", ",", "—", "–"],
            "pt": [";", ":", ",", "—", "–"],
            "de": [";", ":", ",", "—", "–"],
            "ru": [";", ":", ",", "—", "–"],
            "pl": [";", ":", ",", "—", "–"],
            "nl": [";", ":", ",", "—", "–"],
            "sv": [";", ":", ",", "—", "–"],
            "da": [";", ":", ",", "—", "–"],
            "no": [";", ":", ",", "—", "–"],
            "fi": [";", ":", ",", "—", "–"],
            "tr": [";", ":", ",", "—", "–"],
            "ar": ["؛", "،", "—", "–"],  # Arabic
            "he": [";", ":", ",", "—", "–"],
            "zh": ["；", "：", "，"],  # Chinese
            "ja": ["；", "：", "、"],  # Japanese
            "ko": [";", ":", ",", "—", "–"],
            "hi": [";", ":", ",", "—", "–"],  # Hindi
            "th": [";", ":", ",", "—", "–"],
            "vi": [";", ":", ",", "—", "–"],
            "el": [";", ":", ",", "—", "–"],
            "sw": [";", ":", ",", "—", "–"],
            "ms": [";", ":", ",", "—", "–"],
        }

    def get_punctuation_for_language(
        self, language_id: str
    ) -> Tuple[List[str], List[str]]:
        """
        Get appropriate punctuation for the specified language.

        Args:
            language_id: Language code (e.g., 'en', 'zh', 'ar')

        Returns:
            Tuple of (sentence_endings, secondary_breaks)
        """
        lang = language_id.lower() if language_id else "en"

        sentence_endings = self.sentence_endings.get(lang, self.sentence_endings["en"])
        secondary_breaks = self.secondary_breaks.get(lang, self.secondary_breaks["en"])

        return sentence_endings, secondary_breaks

    def find_break_positions(self, text: str, punctuation: List[str]) -> List[int]:
        """
        Find optimal break positions in text based on punctuation.

        Args:
            text: Input text
            punctuation: List of punctuation marks to consider

        Returns:
            List of character positions where breaks can occur
        """
        positions = []

        for punct in punctuation:
            # Find all occurrences of this punctuation
            for match in re.finditer(re.escape(punct), text):
                pos = match.end()  # Break after the punctuation
                positions.append(pos)

        # Remove duplicates and sort
        positions = sorted(set(positions))
        return positions

    def chunk_text(self, text: str, language_id: str = "en") -> List[Chunk]:
        """
        Intelligently chunk text into optimal pieces for TTS.

        Args:
            text: Input text to chunk
            language_id: Language code for appropriate punctuation handling

        Returns:
            List of Chunk objects representing the optimal chunks
        """
        if not text or len(text) <= self.max_chars:
            return [Chunk(text=text, index=0, char_count=len(text), is_final=True)]

        # Get language-specific punctuation
        sentence_endings, secondary_breaks = self.get_punctuation_for_language(
            language_id
        )
        all_punctuation = sentence_endings + secondary_breaks

        # Find all potential break positions
        break_positions = self.find_break_positions(text, all_punctuation)

        if not break_positions:
            # No natural breaks found, split at word boundaries
            return self._chunk_by_words(text)

        chunks = []
        current_start = 0
        chunk_index = 0

        while current_start < len(text):
            # Calculate remaining text length
            remaining_length = len(text) - current_start

            # Check if we need to split this chunk
            need_to_split = remaining_length > self.max_chars

            if not need_to_split:
                # Remaining text fits in one chunk
                chunk_text = text[current_start:].strip()
                if chunk_text:
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            index=chunk_index,
                            char_count=len(chunk_text),
                            is_final=True,
                        )
                    )
                break

            # Text is longer than max_chars, need to split
            # Find break positions within the allowed range
            search_end = current_start + self.max_chars

            # Find all break positions within the current chunk limit
            valid_breaks = [
                pos for pos in break_positions if current_start < pos <= search_end
            ]

            if valid_breaks:
                # Use the last break point within the limit for maximum chunk size
                break_pos = valid_breaks[-1]
                chunk_text = text[current_start:break_pos].strip()
            else:
                # No natural breaks found within limit, try to find word boundary
                chunk_text = self._find_text_within_limit(
                    text, current_start, search_end
                )

            # Ensure we always get some text to process
            if not chunk_text or len(chunk_text) > self.max_chars:
                # Final fallback: hard split at max_chars
                chunk_text = text[
                    current_start : current_start + self.max_chars
                ].strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=chunk_index,
                        char_count=len(chunk_text),
                        is_final=False,
                    )
                )
                current_start += len(chunk_text)
                # Skip whitespace after chunk if not at end
                if current_start < len(text):
                    while current_start < len(text) and text[current_start] in " \t\n":
                        current_start += 1
                chunk_index += 1
            else:
                # If we somehow got no text, advance by max_chars to avoid infinite loop
                current_start += self.max_chars
                chunk_index += 1

        # Add remaining text as final chunk
        if current_start < len(text):
            remaining_text = text[current_start:].strip()
            if remaining_text:
                chunks.append(
                    Chunk(
                        text=remaining_text,
                        index=chunk_index,
                        char_count=len(remaining_text),
                        is_final=True,
                    )
                )

        # Mark the last chunk as final
        if chunks:
            chunks[-1].is_final = True

        return chunks

    def _find_closest_break(
        self,
        text: str,
        start: int,
        original_break: int,
        max_length: int,
        punctuation: List[str],
    ) -> int:
        """
        Find the closest break point within the max length constraint.

        Args:
            text: Input text
            start: Starting position
            original_break: Original intended break position
            max_length: Maximum allowed length
            punctuation: List of punctuation marks

        Returns:
            Character position for the break
        """
        search_end = min(start + max_length, len(text))

        # Look for breaks within the allowed length
        for break_pos in range(search_end, start, -1):
            if any(
                text[break_pos - 1 : break_pos + 1].count(punct) > 0
                for punct in punctuation
            ):
                return break_pos

        # If no punctuation found, break at word boundary
        for break_pos in range(search_end, start, -1):
            if text[break_pos - 1 : break_pos] in [" ", "\t", "\n"]:
                return break_pos

        # Last resort: hard break at max length
        return min(start + max_length, len(text))

    def _find_text_within_limit(self, text: str, start: int, end: int) -> Optional[str]:
        """
        Find the best text chunk within the given limits.

        Args:
            text: Input text
            start: Starting position
            end: Ending position (exclusive)

        Returns:
            Text chunk within limits, or None if not found
        """
        if start >= len(text):
            return None

        # Limit the search to max_chars
        actual_end = min(end, start + self.max_chars)

        # Try to find punctuation-based breaks first
        for punct in [".", "!", "?", ";", ":", "，", "。", "！", "？", "；", "："]:
            for i in range(actual_end - 1, start, -1):
                if i < len(text) and text[i] == punct:
                    # Found punctuation, return text up to and including it
                    chunk_text = text[start : i + 1].strip()
                    if chunk_text and len(chunk_text) <= self.max_chars:
                        return chunk_text

        # Try word boundaries
        for i in range(actual_end - 1, start, -1):
            if text[i] in [" ", "\t", "\n"]:
                chunk_text = text[start:i].strip()
                if chunk_text and len(chunk_text) <= self.max_chars:
                    return chunk_text

        # No good break found, return text up to limit
        chunk_text = text[start:actual_end].strip()
        if chunk_text and len(chunk_text) <= self.max_chars:
            return chunk_text

        # If nothing found, return None to trigger fallback
        return None

    def _chunk_by_words(self, text: str) -> List[Chunk]:
        """
        Fallback chunking method that splits by word boundaries.
        Handles long words by splitting them when necessary.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        words = text.split()
        chunks = []
        current_chunk = ""
        chunk_index = 0
        word_index = 0

        while word_index < len(words):
            word = words[word_index]

            # Check if adding this word would exceed the limit
            test_chunk = current_chunk + (" " if current_chunk else "") + word

            if len(test_chunk) <= self.max_chars:
                current_chunk = test_chunk
                word_index += 1
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(
                        Chunk(
                            text=current_chunk,
                            index=chunk_index,
                            char_count=len(current_chunk),
                            is_final=False,
                        )
                    )
                    chunk_index += 1
                    current_chunk = ""
                else:
                    # Current chunk is empty, so this word is too long
                    # Split the word into chunks
                    remaining_word = word
                    while len(remaining_word) > self.max_chars:
                        # Add a chunk with the first max_chars characters
                        word_chunk = remaining_word[: self.max_chars]
                        chunks.append(
                            Chunk(
                                text=word_chunk,
                                index=chunk_index,
                                char_count=len(word_chunk),
                                is_final=False,
                            )
                        )
                        chunk_index += 1
                        remaining_word = remaining_word[self.max_chars :]

                    # Set current_chunk to the remaining part (if any)
                    if remaining_word:
                        current_chunk = remaining_word
                        word_index += 1
                    else:
                        # Word was exactly split on max_chars boundary
                        current_chunk = ""
                        word_index += 1

        # Add final chunk
        if current_chunk:
            chunks.append(
                Chunk(
                    text=current_chunk,
                    index=chunk_index,
                    char_count=len(current_chunk),
                    is_final=True,
                )
            )

        # Mark the last chunk as final
        if chunks:
            chunks[-1].is_final = True

        return chunks

    def chunk_with_metadata(self, text: str, language_id: str = "en") -> Dict:
        """
        Chunk text and return detailed metadata.

        Args:
            text: Input text to chunk
            language_id: Language code

        Returns:
            Dictionary with chunks and metadata
        """
        chunks = self.chunk_text(text, language_id)

        return {
            "chunks": chunks,
            "original_length": len(text),
            "chunk_count": len(chunks),
            "total_chars_in_chunks": sum(chunk.char_count for chunk in chunks),
            "language": language_id,
            "max_chars": self.max_chars,
            "compression_ratio": len(text) / sum(chunk.char_count for chunk in chunks)
            if chunks
            else 1.0,
        }

    def validate_chunks(self, chunks: List[Chunk]) -> bool:
        """
        Validate that chunks meet the requirements.

        Args:
            chunks: List of chunks to validate

        Returns:
            True if all chunks are valid
        """
        for chunk in chunks:
            if chunk.char_count > self.max_chars:
                return False
            if not chunk.text.strip():
                return False

        return True


# Utility functions for easy integration
def smart_chunk_text(
    text: str, language_id: str = "en", max_chars: int = 300
) -> List[str]:
    """
    Simple utility function to chunk text.

    Args:
        text: Input text to chunk
        language_id: Language code
        max_chars: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    chunker = TextChunker(max_chars=max_chars)
    chunks = chunker.chunk_text(text, language_id)
    return [chunk.text for chunk in chunks]


def chunk_text_with_info(
    text: str, language_id: str = "en", max_chars: int = 300
) -> Dict:
    """
    Utility function to chunk text with detailed information.

    Args:
        text: Input text to chunk
        language_id: Language code
        max_chars: Maximum characters per chunk

    Returns:
        Dictionary with chunks and metadata
    """
    chunker = TextChunker(max_chars=max_chars)
    return chunker.chunk_with_metadata(text, language_id)


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    chunker = TextChunker()

    # Test with English
    long_text = """
    This is a long text that needs to be chunked into smaller pieces for TTS processing.
    The system will automatically find natural breaking points like periods, semicolons, and commas.
    Each chunk will be optimized for the 300-character limit while maintaining natural speech flow.
    This approach ensures better audio quality and more natural-sounding speech synthesis.
    """

    chunks = chunker.chunk_text(long_text, "en")

    print(f"Original text length: {len(long_text)}")
    print(f"Number of chunks: {len(chunks)}")
    print("\nChunks:")

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1} ({chunk.char_count} chars): {chunk.text[:100]}...")

    # Test with Chinese
    chinese_text = "这是一个很长的中文文本，需要分成小块进行处理。系统会自动寻找合适的中文标点符号作为断点。每个分块都会优化到300字符以内，同时保持自然的语音流畅度。这种方法确保更好的音频质量和更自然的语音合成效果。"

    chinese_chunks = chunker.chunk_text(chinese_text, "zh")
    print(f"\nChinese text chunks: {len(chinese_chunks)}")

    for i, chunk in enumerate(chinese_chunks):
        print(f"Chinese Chunk {i + 1} ({chunk.char_count} chars): {chunk.text}")
