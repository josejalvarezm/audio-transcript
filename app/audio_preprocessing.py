"""
Audio Preprocessing Module for Transcription Quality Enhancement

This module provides optional audio preprocessing to improve
transcription accuracy, especially for noisy or low-quality recordings.

Features:
- Noise reduction (using noisereduce library)
- Volume normalisation
- Silence detection/trimming

Note: These features are OPTIONAL and OFF BY DEFAULT.
For clean studio recordings, preprocessing may not be necessary
and could potentially reduce quality.
"""

import tempfile
from pathlib import Path
from typing import Optional
import numpy as np

# Try to import optional dependencies
try:
    import noisereduce as nr
    NOISE_REDUCE_AVAILABLE = True
except ImportError:
    NOISE_REDUCE_AVAILABLE = False
    nr = None

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    sf = None


class AudioPreprocessor:
    """
    Audio preprocessing pipeline for improving transcription quality.
    
    This is particularly useful for:
    - Recordings with background noise (traffic, AC, fan, etc.)
    - Low volume recordings
    - Recordings with long silences
    
    Note: For clean recordings, preprocessing may not help and could
    potentially reduce quality. Use with caution.
    """
    
    def __init__(self, 
                 enable_noise_reduction: bool = False,
                 enable_normalisation: bool = False,
                 noise_reduction_strength: float = 0.5):
        """
        Initialize the audio preprocessor.
        
        Args:
            enable_noise_reduction: Enable noise reduction (requires noisereduce)
            enable_normalisation: Enable volume normalisation
            noise_reduction_strength: Strength of noise reduction (0.0-1.0)
        """
        self.enable_noise_reduction = enable_noise_reduction
        self.enable_normalisation = enable_normalisation
        self.noise_reduction_strength = max(0.0, min(1.0, noise_reduction_strength))
        
    @staticmethod
    def is_available() -> tuple[bool, str]:
        """
        Check if audio preprocessing is available.
        
        Returns:
            (is_available, message)
        """
        if not SOUNDFILE_AVAILABLE:
            return False, (
                "Audio preprocessing not available.\n"
                "Install with: pip install soundfile\n"
            )
        if not NOISE_REDUCE_AVAILABLE:
            return False, (
                "Noise reduction not available.\n"
                "Install with: pip install noisereduce soundfile\n"
            )
        return True, "Audio preprocessing available"
    
    def preprocess(self, audio_path: str, progress_callback=None) -> str:
        """
        Preprocess audio for optimal transcription quality.
        
        Args:
            audio_path: Path to the input audio file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to preprocessed audio file (may be same as input if no processing needed)
        """
        # If no preprocessing is enabled or dependencies missing, return original
        if not (self.enable_noise_reduction or self.enable_normalisation):
            return audio_path
            
        if not SOUNDFILE_AVAILABLE:
            if progress_callback:
                progress_callback("⚠️ soundfile not installed, skipping preprocessing")
            return audio_path
        
        try:
            if progress_callback:
                progress_callback("Loading audio for preprocessing...")
            
            # Load audio
            audio, sample_rate = sf.read(audio_path)
            
            # Convert stereo to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Apply noise reduction
            if self.enable_noise_reduction and NOISE_REDUCE_AVAILABLE:
                if progress_callback:
                    progress_callback("Reducing background noise...")
                audio = self._reduce_noise(audio, sample_rate)
            
            # Apply volume normalisation
            if self.enable_normalisation:
                if progress_callback:
                    progress_callback("Normalising volume...")
                audio = self._normalise_volume(audio)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(temp_file.name, audio, sample_rate)
            
            if progress_callback:
                progress_callback("Audio preprocessing complete")
            
            return temp_file.name
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"⚠️ Preprocessing failed: {e}")
            return audio_path
    
    def _reduce_noise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply noise reduction to audio.
        
        Uses stationary noise reduction which works well for
        consistent background noise like AC, fans, etc.
        """
        if not NOISE_REDUCE_AVAILABLE:
            return audio
            
        # Adjust prop_decrease based on strength setting
        # Higher values = more aggressive noise reduction
        prop_decrease = 0.5 + (self.noise_reduction_strength * 0.4)
        
        reduced = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=True,
            prop_decrease=prop_decrease
        )
        
        return reduced
    
    def _normalise_volume(self, audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """
        Normalise audio volume to target peak level.
        
        Args:
            audio: Audio samples
            target_peak: Target peak amplitude (0.0-1.0)
            
        Returns:
            Normalised audio
        """
        max_amplitude = np.max(np.abs(audio))
        
        if max_amplitude > 0:
            # Scale to target peak
            audio = audio * (target_peak / max_amplitude)
        
        return audio


def check_preprocessing_available() -> tuple[bool, str]:
    """
    Check if audio preprocessing capabilities are available.
    
    Returns:
        (is_available, message)
    """
    return AudioPreprocessor.is_available()
