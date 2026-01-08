"""
Post-Processing Module for Transcript Enhancement

This module provides text post-processing capabilities for improving
transcription quality, including:
- Punctuation restoration
- Capitalisation correction
- Optional filler word removal

Uses deepmultilingual-punctuation for 90+ language support.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import punctuation model (optional dependency)
try:
    from deepmultilingualpunctuation import PunctuationModel
    PUNCTUATION_AVAILABLE = True
except ImportError:
    PUNCTUATION_AVAILABLE = False
    PunctuationModel = None


@dataclass
class PostProcessingConfig:
    """Configuration for post-processing options."""
    restore_punctuation: bool = True
    remove_fillers: bool = False
    filler_words: tuple = ("um", "uh", "er", "ah", "like", "you know", "i mean")


class TranscriptPostProcessor:
    """
    Post-processes transcription text to improve readability.
    
    Features:
    - Punctuation restoration (using deepmultilingual-punctuation)
    - Capitalisation correction
    - Optional filler word removal
    """
    
    def __init__(self, config: Optional[PostProcessingConfig] = None):
        """
        Initialize the post-processor.
        
        Args:
            config: Post-processing configuration options
        """
        self.config = config or PostProcessingConfig()
        self._punct_model = None
        self._model_loaded = False
        
    def load_model(self, progress_callback=None):
        """
        Load the punctuation model.
        
        Args:
            progress_callback: Optional callback for progress updates
        """
        if not PUNCTUATION_AVAILABLE:
            if progress_callback:
                progress_callback("⚠️ Punctuation model not installed (pip install deepmultilingualpunctuation)")
            return False
            
        if self._model_loaded:
            return True
            
        try:
            if progress_callback:
                progress_callback("Loading punctuation model...")
            self._punct_model = PunctuationModel()
            self._model_loaded = True
            if progress_callback:
                progress_callback("Punctuation model ready")
            return True
        except Exception as e:
            if progress_callback:
                progress_callback(f"⚠️ Failed to load punctuation model: {e}")
            return False
    
    def restore_punctuation(self, text: str) -> str:
        """
        Restore punctuation and capitalisation to raw text.
        
        Args:
            text: Raw transcription text (typically lowercase, no punctuation)
            
        Returns:
            Text with restored punctuation and proper capitalisation
        """
        if not text or not text.strip():
            return text
            
        # If punctuation model is available and configured, use it
        if self.config.restore_punctuation and self._model_loaded and self._punct_model:
            try:
                # The model expects text without punctuation
                # Remove any existing punctuation first to avoid doubling
                clean_text = self._clean_for_punctuation(text)
                punctuated = self._punct_model.restore_punctuation(clean_text)
                return punctuated
            except (RuntimeError, ValueError) as e:
                logger.warning("Punctuation restoration failed: %s", e)
                # Fall back to basic processing
                return self._basic_capitalisation(text)
        else:
            # Use basic capitalisation if model not available
            return self._basic_capitalisation(text)
    
    def _clean_for_punctuation(self, text: str) -> str:
        """Clean text before punctuation restoration."""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _basic_capitalisation(self, text: str) -> str:
        """
        Apply basic capitalisation rules when punctuation model is unavailable.
        
        This is a fallback that:
        - Capitalises first letter of sentences
        - Capitalises 'I' when standalone
        - Preserves existing capitalisation
        """
        if not text:
            return text
            
        # Capitalise first letter
        result = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Capitalise after sentence endings
        result = re.sub(
            r'([.!?]\s+)([a-z])',
            lambda m: m.group(1) + m.group(2).upper(),
            result
        )
        
        # Capitalise standalone 'i'
        result = re.sub(r'\bi\b', 'I', result)
        
        return result
    
    def remove_filler_words(self, text: str) -> str:
        """
        Remove common filler words from text.
        
        Args:
            text: Text that may contain filler words
            
        Returns:
            Text with filler words removed
        """
        if not self.config.remove_fillers:
            return text
            
        result = text
        for filler in self.config.filler_words:
            # Match filler words with word boundaries
            pattern = rf'\b{re.escape(filler)}\b[,]?\s*'
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        
        # Clean up any double spaces created
        result = re.sub(r'\s+', ' ', result)
        return result.strip()
    
    def process_text(self, text: str) -> str:
        """
        Apply all configured post-processing to text.
        
        Args:
            text: Raw transcription text
            
        Returns:
            Processed text with punctuation, capitalisation, etc.
        """
        if not text:
            return text
            
        result = text
        
        # Apply punctuation restoration first
        if self.config.restore_punctuation:
            result = self.restore_punctuation(result)
        
        # Remove fillers if configured
        if self.config.remove_fillers:
            result = self.remove_filler_words(result)
        
        return result
    
    def process_segments(self, segments: list) -> list:
        """
        Apply post-processing to a list of transcript segments.
        
        Args:
            segments: List of Segment objects with 'text' attribute
            
        Returns:
            Segments with processed text
        """
        for segment in segments:
            if hasattr(segment, 'text'):
                segment.text = self.process_text(segment.text)
            elif isinstance(segment, dict) and 'text' in segment:
                segment['text'] = self.process_text(segment['text'])
        
        return segments


def check_post_processing_available() -> tuple[bool, str]:
    """
    Check if post-processing capabilities are available.
    
    Returns:
        (is_available, message)
    """
    if not PUNCTUATION_AVAILABLE:
        return False, (
            "Punctuation restoration not available.\n"
            "Install with: pip install deepmultilingualpunctuation\n\n"
            "Basic capitalisation will be used instead."
        )
    
    return True, "Punctuation restoration available"


# Convenience function for simple use cases
def restore_punctuation_simple(text: str) -> str:
    """
    Simple one-shot punctuation restoration.
    
    This loads the model each time, so for batch processing,
    use TranscriptPostProcessor instead.
    
    Args:
        text: Raw text to punctuate
        
    Returns:
        Punctuated text
    """
    processor = TranscriptPostProcessor()
    if processor.load_model():
        return processor.restore_punctuation(text)
    else:
        return processor._basic_capitalisation(text)
