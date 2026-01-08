"""
Speaker Diarization and Transcript Structuring Module

This module provides speaker diarization using pyannote.audio and 
combines it with Whisper transcription for speaker-labeled transcripts.
"""

import os
# Disable pyannote OpenTelemetry telemetry to prevent network overhead
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OTEL_TRACES_EXPORTER"] = "none"
os.environ["OTEL_METRICS_EXPORTER"] = "none"
os.environ["OTEL_LOGS_EXPORTER"] = "none"
import re
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torchaudio

logger = logging.getLogger(__name__)

# Check for pyannote availability
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    Pipeline = None


@dataclass
class Segment:
    """A transcript segment with speaker and timing information."""
    start: float
    end: float
    speaker: str
    text: str
    confidence: float = 1.0  # Speaker assignment confidence (0-1)
    whisper_confidence: float = 0.0  # Whisper's avg_logprob (higher = more confident, 0 = not available)
    
    def format_timestamp(self, time_val: float) -> str:
        """Convert seconds to MM:SS.mmm format."""
        minutes = int(time_val // 60)
        seconds = time_val % 60
        return f"{minutes:02d}:{seconds:06.3f}"
    
    def to_string(self, show_confidence: bool = False) -> str:
        """Format segment as timestamped speaker line."""
        start_ts = self.format_timestamp(self.start)
        end_ts = self.format_timestamp(self.end)
        base = f"[{start_ts} → {end_ts}] **{self.speaker}:** {self.text}"
        
        # Add confidence warning for low-confidence segments
        if show_confidence and self.confidence < 0.5:
            base += " ⚠️"
        return base
    
    @property
    def is_low_confidence(self) -> bool:
        """Check if this segment has low confidence."""
        return self.confidence < 0.5 or self.whisper_confidence < -1.0


class DiarizationPipeline:
    """
    Speaker diarization pipeline using pyannote.audio.
    
    Requires a HuggingFace token with access to pyannote models.
    Get access at: https://huggingface.co/pyannote/speaker-diarization-3.1
    """
    
    def __init__(self, hf_token: Optional[str] = None, device: str = "auto"):
        """
        Initialize the diarization pipeline.
        
        Args:
            hf_token: HuggingFace API token with pyannote access
            device: "cuda", "cpu", or "auto"
        """
        if not PYANNOTE_AVAILABLE:
            raise ImportError(
                "pyannote.audio is not installed. Install it with:\n"
                "pip install pyannote.audio"
            )
        
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError(
                "HuggingFace token required for pyannote models.\n"
                "1. Get token at: https://huggingface.co/settings/tokens\n"
                "2. Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "3. Set HF_TOKEN environment variable or pass hf_token parameter"
            )
        
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.pipeline = None
        self._loaded = False
    
    def load(self, progress_callback=None):
        """Load the diarization pipeline."""
        logger.info("DiarizationPipeline.load() started")
        if progress_callback:
            progress_callback("Loading speaker diarization model...")
        
        logger.info("Downloading/loading pyannote/speaker-diarization-3.1...")
        load_start = time.time()
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=self.hf_token  # Updated API - 'token' replaces deprecated 'use_auth_token'
        )
        logger.info(f"Pipeline loaded in {time.time() - load_start:.1f}s")
        
        logger.info(f"Moving pipeline to device: {self.device}")
        self.pipeline.to(self.device)
        self._loaded = True
        logger.info("DiarizationPipeline.load() completed")
        
        if progress_callback:
            progress_callback("Diarization model ready")
    
    def diarize(self, audio_path: str, num_speakers: Optional[int] = None,
                min_speakers: Optional[int] = None, max_speakers: Optional[int] = None,
                progress_callback=None) -> list[tuple[float, float, str]]:
        """
        Perform speaker diarization on an audio file.
        
        Args:
            audio_path: Path to the audio file
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            progress_callback: Function to call with status updates
            
        Returns:
            List of (start_time, end_time, speaker_label) tuples
        """
        if not self._loaded:
            self.load(progress_callback)
        
        if progress_callback:
            progress_callback("Running speaker diarization...")
        
        # Build parameters
        params = {}
        if num_speakers is not None:
            params["num_speakers"] = num_speakers
        if min_speakers is not None:
            params["min_speakers"] = min_speakers
        if max_speakers is not None:
            params["max_speakers"] = max_speakers
        
        # Load audio as waveform to bypass torchcodec issues
        # pyannote expects {"waveform": tensor, "sample_rate": int}
        logger.info(f"Loading audio for diarization: {audio_path}")
        try:
            load_start = time.time()
            waveform, sample_rate = torchaudio.load(audio_path)
            logger.info(f"Audio loaded: shape={waveform.shape}, sample_rate={sample_rate}, time={time.time() - load_start:.2f}s")
            
            # Resample to 16kHz if needed (pyannote expects 16kHz)
            if sample_rate != 16000:
                logger.info(f"Resampling from {sample_rate}Hz to 16000Hz...")
                resample_start = time.time()
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
                logger.info(f"Resampling completed in {time.time() - resample_start:.2f}s")
            
            audio_input = {"waveform": waveform, "sample_rate": sample_rate}
            logger.info(f"Audio prepared as waveform dict")
        except Exception as e:
            # Fallback to file path if waveform loading fails
            logger.warning(f"Could not load audio as waveform ({e}), using file path instead")
            audio_input = audio_path
        
        # Run diarization
        logger.info(f"Running pyannote pipeline with params: {params}")
        logger.info(">>> DIARIZATION PROCESSING START (this may take a while) <<<")
        diarization_start = time.time()
        
        # Start heartbeat thread to show progress
        import threading
        heartbeat_active = True
        def heartbeat():
            while heartbeat_active:
                elapsed = time.time() - diarization_start
                logger.info(f"    ... diarization still running ({elapsed:.0f}s elapsed)")
                time.sleep(30)  # Log every 30 seconds
        
        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        
        try:
            diarization_output = self.pipeline(audio_input, **params)
        finally:
            heartbeat_active = False
        
        logger.info(f">>> DIARIZATION PROCESSING COMPLETE in {time.time() - diarization_start:.1f}s <<<")
        
        # pyannote 3.1+ returns DiarizeOutput dataclass, extract the Annotation
        # For transcription, use exclusive_speaker_diarization (no overlapping speech)
        logger.info(f"Extracting diarization results (output type: {type(diarization_output).__name__})")
        if hasattr(diarization_output, 'exclusive_speaker_diarization'):
            logger.info("Using exclusive_speaker_diarization")
            diarization = diarization_output.exclusive_speaker_diarization
        elif hasattr(diarization_output, 'speaker_diarization'):
            logger.info("Using speaker_diarization")
            diarization = diarization_output.speaker_diarization
        else:
            # Older pyannote versions return Annotation directly
            logger.info("Using Annotation directly (older pyannote version)")
            diarization = diarization_output
        
        # Extract segments
        logger.info("Extracting speaker segments...")
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
        
        unique_speakers = len(set(s[2] for s in segments))
        logger.info(f"Extracted {len(segments)} segments from {unique_speakers} speakers")
        
        if progress_callback:
            progress_callback(f"Found {unique_speakers} speakers")
        
        return segments


class TranscriptStructurer:
    """
    Combines Whisper transcription with speaker diarization
    to produce speaker-labeled transcripts.
    """
    
    def __init__(self, speaker_names: Optional[dict[str, str]] = None):
        """
        Initialize the transcript structurer.
        
        Args:
            speaker_names: Optional mapping of speaker IDs to names
                          e.g., {"SPEAKER_00": "Irina", "SPEAKER_01": "Guard"}
        """
        self.speaker_names = speaker_names or {}
        self.notes = []
    
    def assign_speakers(self, whisper_segments: list[dict], 
                       diarization_segments: list[tuple[float, float, str]]) -> list[Segment]:
        """
        Assign speakers to Whisper transcript segments.
        
        Args:
            whisper_segments: Segments from Whisper transcription
            diarization_segments: Speaker segments from diarization
            
        Returns:
            List of Segment objects with speaker assignments and confidence scores
        """
        self.notes = []
        result = []
        
        for ws in whisper_segments:
            start = ws.get("start", 0)
            end = ws.get("end", 0)
            text = ws.get("text", "").strip()
            whisper_confidence = ws.get("avg_logprob", 0)  # Extract Whisper confidence
            
            if not text:
                continue
            
            # Find the speaker with most overlap and get confidence
            speaker, speaker_confidence = self._find_speaker_with_confidence(start, end, diarization_segments)
            
            # Map to human-readable name if available
            display_speaker = self.speaker_names.get(speaker, speaker)
            
            result.append(Segment(
                start=start,
                end=end,
                speaker=display_speaker,
                text=text,
                confidence=speaker_confidence,
                whisper_confidence=whisper_confidence
            ))
        
        return result
    
    def _find_speaker_with_confidence(self, start: float, end: float, 
                     diarization_segments: list[tuple[float, float, str]]) -> tuple[str, float]:
        """Find the speaker with the most overlap for a time range.
        
        Returns:
            Tuple of (speaker_label, confidence_score)
        """
        best_speaker = "Speaker ?"
        best_overlap = 0
        
        for seg_start, seg_end, speaker in diarization_segments:
            # Calculate overlap
            overlap_start = max(start, seg_start)
            overlap_end = min(end, seg_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        
        # Calculate confidence as overlap ratio
        segment_duration = end - start
        if segment_duration > 0:
            confidence = best_overlap / segment_duration
        else:
            confidence = 0.0
        
        # Check if overlap is significant
        if confidence < 0.3:
            self.notes.append(f"Low confidence speaker assignment at {start:.1f}s-{end:.1f}s (confidence: {confidence:.0%})")
            return "Speaker ?", confidence
        
        return best_speaker, confidence
    
    def _find_speaker(self, start: float, end: float, 
                     diarization_segments: list[tuple[float, float, str]]) -> str:
        """Find the speaker with the most overlap for a time range (legacy method)."""
        speaker, _ = self._find_speaker_with_confidence(start, end, diarization_segments)
        return speaker
    
    def format_transcript(self, segments: list[Segment], 
                         include_notes: bool = True,
                         show_confidence_warnings: bool = True) -> str:
        """
        Format segments into a structured transcript.
        
        Args:
            segments: List of Segment objects
            include_notes: Whether to include the Notes section
            show_confidence_warnings: Whether to show ⚠️ on low-confidence segments
            
        Returns:
            Formatted transcript string
        """
        lines = ["## Transcript (Speaker-Assigned)", ""]
        
        low_confidence_count = 0
        for seg in segments:
            lines.append(seg.to_string(show_confidence=show_confidence_warnings))
            if seg.is_low_confidence:
                low_confidence_count += 1
        
        if include_notes:
            lines.extend(["", "## Notes", ""])
            
            # Add diarization logic notes
            speakers = set(seg.speaker for seg in segments)
            lines.append(f"- Identified {len(speakers)} distinct speaker(s): {', '.join(sorted(speakers))}")
            
            # Add confidence summary
            if segments:
                avg_confidence = sum(seg.confidence for seg in segments) / len(segments)
                lines.append(f"- Average speaker confidence: {avg_confidence:.0%}")
                if low_confidence_count > 0:
                    lines.append(f"- ⚠️ {low_confidence_count} segment(s) with low confidence (marked with ⚠️)")
            
            # Add any warnings
            for note in self.notes:
                lines.append(f"- ⚠️ {note}")
            
            if not self.notes and low_confidence_count == 0:
                lines.append("- ✅ All segments have high confidence")
        
        return "\n".join(lines)
    
    def merge_consecutive_speaker_segments(self, segments: list[Segment], 
                                           gap_threshold: float = 1.0) -> list[Segment]:
        """
        Merge consecutive segments from the same speaker.
        
        Args:
            segments: List of Segment objects
            gap_threshold: Maximum gap (seconds) to merge across
            
        Returns:
            Merged list of Segment objects
        """
        if not segments:
            return []
        
        merged = [segments[0]]
        
        for seg in segments[1:]:
            last = merged[-1]
            gap = seg.start - last.end
            
            if seg.speaker == last.speaker and gap <= gap_threshold:
                # Merge with previous segment, averaging confidence scores
                avg_confidence = (last.confidence + seg.confidence) / 2
                avg_whisper_conf = (last.whisper_confidence + seg.whisper_confidence) / 2
                merged[-1] = Segment(
                    start=last.start,
                    end=seg.end,
                    speaker=last.speaker,
                    text=f"{last.text} {seg.text}",
                    confidence=avg_confidence,
                    whisper_confidence=avg_whisper_conf
                )
            else:
                merged.append(seg)
        
        return merged


def structure_whisper_only(whisper_result: dict, 
                          num_speakers: int = 2,
                          speaker_names: Optional[dict[str, str]] = None,
                          post_processor=None) -> str:
    """
    Structure a Whisper transcript without diarization.
    Uses simple heuristics to guess speaker changes.
    
    This is a fallback when pyannote is not available.
    
    Args:
        whisper_result: Result from whisper.transcribe()
        num_speakers: Expected number of speakers
        speaker_names: Optional speaker name mapping
        post_processor: Optional TranscriptPostProcessor for punctuation restoration
        
    Returns:
        Formatted transcript string
    """
    segments = whisper_result.get("segments", [])
    
    if not segments:
        return "No segments found in transcript."
    
    # Simple heuristic: alternate speakers on significant pauses
    # or when there's a clear turn-taking pattern
    
    structurer = TranscriptStructurer(speaker_names)
    result_segments = []
    
    current_speaker_idx = 0
    speaker_labels = [f"Speaker {chr(65 + i)}" for i in range(num_speakers)]
    
    prev_end = 0
    for seg in segments:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "").strip()
        
        if not text:
            continue
        
        # Heuristic: switch speaker on pauses > 1 second
        gap = start - prev_end
        if gap > 1.0 and prev_end > 0:
            current_speaker_idx = (current_speaker_idx + 1) % num_speakers
        
        # Heuristic: questions often indicate speaker change
        if prev_end > 0 and result_segments:
            last_text = result_segments[-1].text
            if last_text.rstrip().endswith("?"):
                current_speaker_idx = (current_speaker_idx + 1) % num_speakers
        
        speaker = speaker_labels[current_speaker_idx]
        if speaker_names:
            speaker = speaker_names.get(speaker, speaker)
        
        result_segments.append(Segment(
            start=start,
            end=end,
            speaker=speaker,
            text=text
        ))
        
        prev_end = end
    
    # Merge consecutive same-speaker segments
    result_segments = structurer.merge_consecutive_speaker_segments(result_segments)
    
    # Apply punctuation restoration if post_processor is provided
    if post_processor:
        result_segments = post_processor.process_segments(result_segments)
    
    # Format output
    lines = ["## Transcript (Speaker-Assigned)", ""]
    for seg in result_segments:
        lines.append(seg.to_string())
    
    notes = [
        "",
        "## Notes",
        "",
        f"- Estimated {num_speakers} speakers using heuristic analysis",
        "- ⚠️ Speaker assignment based on pause detection and turn-taking patterns",
        "- ⚠️ For accurate diarization, install pyannote.audio and provide HF_TOKEN",
        "- Speaker changes detected at pauses > 1 second and after questions"
    ]
    
    if post_processor:
        notes.insert(4, "- ✅ **Punctuation Restored** (using deepmultilingual-punctuation)")
    
    lines.extend(notes)
    
    return "\n".join(lines)


def check_diarization_available() -> tuple[bool, str]:
    """
    Check if full diarization is available.
    
    Returns:
        (is_available, message)
    """
    if not PYANNOTE_AVAILABLE:
        return False, (
            "pyannote.audio not installed.\n"
            "Install with: pip install pyannote.audio\n\n"
            "Heuristic speaker detection will be used instead."
        )
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        return False, (
            "HF_TOKEN not set.\n"
            "1. Get token: https://huggingface.co/settings/tokens\n"
            "2. Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "3. Set HF_TOKEN in .env file\n\n"
            "Heuristic speaker detection will be used instead."
        )
    
    return True, "Full speaker diarization available"
