"""Audio Transcript Desktop Application - Main Entry Point.

Provides a Tkinter GUI for audio transcription using Whisper with optional
speaker diarization and punctuation restoration.
"""
import os
import sys
import warnings

# =============================================================================
# EARLY SETUP: Configure environment BEFORE importing torch/audio libraries
# This prevents FFmpeg/torchcodec warnings that cannot be suppressed later
# =============================================================================
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"  # Use legacy backend
os.environ["OTEL_SDK_DISABLED"] = "true"  # Disable OpenTelemetry

# Configure logging BEFORE any library imports
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

# Silence noisy library loggers
logging.getLogger("torio").setLevel(logging.ERROR)
logging.getLogger("torio._extension.utils").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Filter specific unavoidable warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pipelines")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.utils")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*degrees of freedom.*")

logger = logging.getLogger(__name__)

# =============================================================================
# STANDARD IMPORTS (after environment setup)
# =============================================================================
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk, simpledialog
import time
from datetime import datetime

from dotenv import load_dotenv
import whisper
import torch

# Enable TF32 for better GPU performance (suppresses pyannote warning)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Import diarization module (use relative import for package, absolute for direct run)
try:
    from app.diarization import (
        check_diarization_available,
        structure_whisper_only,
        DiarizationPipeline,
        TranscriptStructurer,
        PYANNOTE_AVAILABLE
    )
    from app.post_processing import (
        TranscriptPostProcessor,
        PostProcessingConfig,
        check_post_processing_available,
        PUNCTUATION_AVAILABLE
    )
except ModuleNotFoundError:
    from diarization import (
        check_diarization_available,
        structure_whisper_only,
        DiarizationPipeline,
        TranscriptStructurer,
        PYANNOTE_AVAILABLE
    )
    from post_processing import (
        TranscriptPostProcessor,
        PostProcessingConfig,
        check_post_processing_available,
        PUNCTUATION_AVAILABLE
    )

load_dotenv()

# Log availability status at startup (logger, not print)
logger.debug("PYANNOTE_AVAILABLE = %s", PYANNOTE_AVAILABLE)
logger.debug("PUNCTUATION_AVAILABLE = %s", PUNCTUATION_AVAILABLE)
logger.debug("HF_TOKEN set = %s", bool(os.getenv('HF_TOKEN')) and os.getenv('HF_TOKEN') != 'your_token_here')

# =============================================================================
# CONSTANTS (extracted from magic numbers)
# =============================================================================
HF_TOKEN_PLACEHOLDER = "your_token_here"

# Speed factors for ETA estimation (model_name -> realtime multiplier)
MODEL_SPEED_FACTORS = {
    "tiny": 30,
    "base": 20,
    "small": 10,
    "medium": 5,
    "large": 2,
    "large-v2": 2,
    "large-v3": 2,
    "large-v3-turbo": 3,
}
DEFAULT_SPEED_FACTOR = 5
GPU_SPEED_MULTIPLIER = 3
DIARIZATION_TIME_MULTIPLIER = 1.5

# Progress milestones
PROGRESS_AUDIO_LOADED = 5
PROGRESS_TRANSCRIPTION_START = 10
PROGRESS_DIARIZATION_START = 80
PROGRESS_PUNCTUATION_START = 90
PROGRESS_COMPLETE = 100

APP_TITLE = "Audio Transcript - GPU Accelerated"
MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3-turbo")  # Default to Large v3 Turbo for best speed/accuracy
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "transcripts"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Language options for Whisper
LANGUAGES = {
    "Auto Detect": None,
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Polish": "pl",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Hindi": "hi",
    "Turkish": "tr",
    "Vietnamese": "vi",
    "Thai": "th", 
    "Indonesian": "id",
    "Malay": "ms",
    "Filipino": "tl",
}

# Model options - includes Large v3 variants for best accuracy
MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "large-v3-turbo"]


def get_device():
    """Detect the best available device for processing."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return "cuda", f"🚀 GPU: {gpu_name} ({vram:.1f} GB VRAM)"
    else:
        return "cpu", "⚠️ CPU Mode (Install CUDA PyTorch for GPU acceleration)"


class TranscriptionFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=20)
        self.master = master
        self.file_path = tk.StringVar(value="No file selected")
        self.status_var = tk.StringVar(value="Loading Whisper model...")
        self.progress_var = tk.DoubleVar(value=0)
        self.time_var = tk.StringVar(value="")
        self.transcript_ready = False
        self.cancel_requested = False
        self.is_transcribing = False
        self.accumulated_text = []
        
        # Get device info
        self.device, self.device_info = get_device()
        
        self._build_ui()

    def _build_ui(self):
        # Header with GPU status
        header_frame = ttk.Frame(self)
        header_frame.pack(fill="x", pady=(0, 5))
        
        header = ttk.Label(header_frame, text="🎙️ Audio Transcription", font=("Segoe UI", 16, "bold"))
        header.pack(side="left")
        
        # GPU Status indicator
        gpu_status = ttk.Label(self, text=self.device_info, font=("Segoe UI", 9))
        gpu_status.pack(anchor="w", pady=(0, 10))
        
        # Settings row (Model + Language)
        settings_frame = ttk.LabelFrame(self, text="Settings", padding=10)
        settings_frame.pack(fill="x", pady=(0, 10))
        
        # Model selection
        model_frame = ttk.Frame(settings_frame)
        model_frame.pack(fill="x", pady=2)
        ttk.Label(model_frame, text="Model:", width=12).pack(side="left")
        self.model_var = tk.StringVar(value=MODEL_NAME)
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, values=MODELS, state="readonly", width=15)
        model_combo.pack(side="left", padx=(0, 20))
        model_combo.bind("<<ComboboxSelected>>", self.on_model_change)
        
        # Language selection
        ttk.Label(model_frame, text="Language:").pack(side="left")
        self.language_var = tk.StringVar(value="Auto Detect")
        lang_combo = ttk.Combobox(model_frame, textvariable=self.language_var, values=list(LANGUAGES.keys()), state="readonly", width=15)
        lang_combo.pack(side="left", padx=5)
        
        # Auto-save checkbox
        self.auto_save_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(model_frame, text="Auto-save", variable=self.auto_save_var).pack(side="left", padx=20)

        # Diarization settings row
        diarization_frame = ttk.Frame(settings_frame)
        diarization_frame.pack(fill="x", pady=2)
        
        # Enable diarization checkbox - ENABLED BY DEFAULT
        self.diarization_var = tk.BooleanVar(value=True)
        self.diarization_check = ttk.Checkbutton(
            diarization_frame, 
            text="🗣️ Speaker Diarization", 
            variable=self.diarization_var,
            command=self._on_diarization_toggle
        )
        self.diarization_check.pack(side="left")
        
        # Number of speakers (max limit for auto-detection)
        ttk.Label(diarization_frame, text="Max Speakers:").pack(side="left", padx=(20, 5))
        self.num_speakers_var = tk.StringVar(value="2")
        speakers_spinbox = ttk.Spinbox(
            diarization_frame,
            from_=2, to=10,
            width=5,
            textvariable=self.num_speakers_var
        )
        speakers_spinbox.pack(side="left")

        # Add tooltip-like hint
        ttk.Label(
            diarization_frame,
            text="(auto-detects up to this max)",
            font=("Segoe UI", 8),
            foreground="gray"
        ).pack(side="left", padx=5)
        
        # Speaker names button - ENABLED BY DEFAULT since diarization is on
        self.speaker_names_btn = ttk.Button(
            diarization_frame, 
            text="📝 Name Speakers", 
            command=self._configure_speaker_names,
            state="normal"
        )
        self.speaker_names_btn.pack(side="left", padx=10)
        
        # Diarization status label - Show clearly what's being used
        diarization_available, diarization_msg = check_diarization_available()
        self.diarization_status = "full" if diarization_available else "heuristic"
        
        if diarization_available:
            status_text = "✅ AI Pyannote (Accurate)"
            status_color = "green"
        else:
            status_text = "⚠️ Heuristic (Basic)"
            status_color = "orange"
        
        self.diarization_label = ttk.Label(
            diarization_frame, 
            text=status_text, 
            font=("Segoe UI", 9, "bold"),
            foreground=status_color
        )
        self.diarization_label.pack(side="left", padx=10)
        
        # Store speaker name mappings
        self.speaker_names = {}

        # Post-processing settings row
        postproc_frame = ttk.Frame(settings_frame)
        postproc_frame.pack(fill="x", pady=2)
        
        # Punctuation restoration checkbox - ENABLED BY DEFAULT
        self.punctuation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            postproc_frame, 
            text="✍️ Restore Punctuation", 
            variable=self.punctuation_var
        ).pack(side="left")
        
        # Punctuation status label
        punct_available, _ = check_post_processing_available()
        if punct_available:
            punct_status = "✅ Available"
            punct_color = "green"
        else:
            punct_status = "⚠️ Not installed (pip install deepmultilingualpunctuation)"
            punct_color = "orange"
        
        ttk.Label(
            postproc_frame, 
            text=punct_status, 
            font=("Segoe UI", 9),
            foreground=punct_color
        ).pack(side="left", padx=10)

        # File selection row
        file_frame = ttk.LabelFrame(self, text="Audio File", padding=10)
        file_frame.pack(fill="x", pady=(0, 10))
        
        file_row = ttk.Frame(file_frame)
        file_row.pack(fill="x")
        ttk.Button(file_row, text="📂 Select File", command=self.select_file).pack(side="left")
        self.file_label = ttk.Label(file_row, textvariable=self.file_path, wraplength=500, foreground="gray")
        self.file_label.pack(side="left", padx=10)

        # Action buttons row
        actions_frame = ttk.Frame(self)
        actions_frame.pack(fill="x", pady=(0, 10))
        
        self.transcribe_button = ttk.Button(actions_frame, text="▶️ Transcribe", command=self.handle_transcribe, state="disabled")
        self.transcribe_button.pack(side="left")
        
        self.cancel_button = ttk.Button(actions_frame, text="⏹️ Cancel", command=self.handle_cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=5)
        
        self.save_button = ttk.Button(actions_frame, text="💾 Save As...", command=self.save_transcript, state="disabled")
        self.save_button.pack(side="left", padx=5)
        
        self.clear_button = ttk.Button(actions_frame, text="🗑️ Clear", command=self.clear_transcript)
        self.clear_button.pack(side="left", padx=5)

        # Progress section
        progress_frame = ttk.LabelFrame(self, text="Progress", padding=10)
        progress_frame.pack(fill="x", pady=(0, 10))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, mode="determinate")
        self.progress_bar.pack(fill="x", pady=(0, 5))
        
        # Status and time row
        status_row = ttk.Frame(progress_frame)
        status_row.pack(fill="x")
        ttk.Label(status_row, textvariable=self.status_var).pack(side="left")
        ttk.Label(status_row, textvariable=self.time_var, foreground="blue").pack(side="right")

        # Transcript display
        transcript_frame = ttk.LabelFrame(self, text="📝 Transcript", padding=10)
        transcript_frame.pack(fill="both", expand=True)
        
        # Text widget with scrollbar
        text_container = ttk.Frame(transcript_frame)
        text_container.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(text_container)
        scrollbar.pack(side="right", fill="y")
        
        self.text_widget = tk.Text(text_container, height=12, wrap="word", state="disabled", 
                                    font=("Consolas", 10), yscrollcommand=scrollbar.set)
        self.text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.text_widget.yview)
        
        # Word count label
        self.word_count_var = tk.StringVar(value="Words: 0 | Characters: 0")
        ttk.Label(transcript_frame, textvariable=self.word_count_var, font=("Segoe UI", 8)).pack(anchor="e")

    def on_model_change(self, event=None):
        """Handle model change - reload model."""
        new_model = self.model_var.get()
        if new_model != self.master.current_model_name:
            self.status_var.set(f"Loading {new_model} model...")
            self.transcribe_button.config(state="disabled")
            threading.Thread(target=self.master.load_model, args=(new_model,), daemon=True).start()

    def _on_diarization_toggle(self):
        """Handle diarization checkbox toggle."""
        if self.diarization_var.get():
            self.speaker_names_btn.config(state="normal")
            # Show info about diarization mode
            if self.diarization_status == "heuristic":
                messagebox.showinfo(
                    "Heuristic Mode",
                    "Speaker diarization will use heuristic detection.\n\n"
                    "For more accurate results, install pyannote.audio:\n"
                    "pip install pyannote.audio\n\n"
                    "And set HF_TOKEN in your .env file."
                )
        else:
            self.speaker_names_btn.config(state="disabled")

    def _configure_speaker_names(self):
        """Open dialog to configure speaker names."""
        num_speakers = int(self.num_speakers_var.get())
        
        # Create a simple dialog for speaker names
        dialog = tk.Toplevel(self.master)
        dialog.title("Configure Speaker Names")
        dialog.geometry("400x300")
        dialog.transient(self.master)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Enter names for each speaker:", 
                  font=("Segoe UI", 10, "bold")).pack(pady=10)
        
        entries = {}
        for i in range(num_speakers):
            frame = ttk.Frame(dialog)
            frame.pack(fill="x", padx=20, pady=5)
            
            label = f"Speaker {chr(65 + i)}"
            ttk.Label(frame, text=f"{label}:", width=12).pack(side="left")
            
            entry = ttk.Entry(frame)
            entry.pack(side="left", fill="x", expand=True)
            
            # Pre-fill with existing names
            if label in self.speaker_names:
                entry.insert(0, self.speaker_names[label])
            
            entries[label] = entry
        
        def save_names():
            for label, entry in entries.items():
                name = entry.get().strip()
                if name:
                    self.speaker_names[label] = name
                elif label in self.speaker_names:
                    del self.speaker_names[label]
            dialog.destroy()
            self.status_var.set(f"Configured {len(self.speaker_names)} speaker name(s)")
        
        ttk.Button(dialog, text="Save", command=save_names).pack(pady=20)

    def select_file(self):
        path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.m4a *.flac *.ogg *.webm *.wma *.aac"),
                ("Video files", "*.mp4 *.mkv *.avi *.mov"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.file_path.set(path)
            self.file_label.config(foreground="black")
            # Show file size
            size_mb = os.path.getsize(path) / (1024 * 1024)
            self.status_var.set(f"Selected: {Path(path).name} ({size_mb:.1f} MB)")
            # Enable transcribe button if model is ready
            if self.master.model is not None:
                self.transcribe_button.config(state="normal")

    def enable_model_usage(self, model_name):
        device_text = "GPU 🚀" if self.device == "cuda" else "CPU"
        self.status_var.set(f"✅ Model '{model_name}' ready on {device_text}. Select a file to begin.")
        self.transcribe_button.config(state="normal")

    def handle_transcribe(self):
        if not self.master.model:
            messagebox.showinfo("Please wait", "The model is still loading.")
            return
        file_path = self.file_path.get()
        if not os.path.isfile(file_path):
            messagebox.showwarning("No file", "Select an audio file before transcribing.")
            return
        
        # Reset state
        self.cancel_requested = False
        self.is_transcribing = True
        self.accumulated_text = []
        self.progress_var.set(0)
        self.time_var.set("")
        self.clear_transcript()
        
        # Update UI
        self.status_var.set("🔄 Starting transcription...")
        self.transcript_ready = False
        self.save_button.config(state="disabled")
        self.transcribe_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        
        # Get selected language
        lang_name = self.language_var.get()
        language = LANGUAGES.get(lang_name)
        
        threading.Thread(target=self.run_transcription, args=(file_path, language), daemon=True).start()

    def handle_cancel(self):
        """Request cancellation of the current transcription.
        
        Note: Cancellation only takes effect before/after transcription completes.
        Whisper's transcribe() is synchronous and cannot be interrupted mid-process.
        """
        self.cancel_requested = True
        self.status_var.set("⏹️ Cancel requested (takes effect when processing completes)")
        self.cancel_button.config(state="disabled")

    def run_transcription(self, file_path: str, language: str = None):
        """Run transcription with progress tracking and optional diarization.
        
        Note: Whisper's transcribe() is synchronous and doesn't support 
        real-time callbacks. Progress is estimated and updated after completion.
        """
        try:
            logger.info("="*60)
            logger.info(f"TRANSCRIPTION STARTED: {Path(file_path).name}")
            logger.info("="*60)
            start_time = time.time()
            use_diarization = self.diarization_var.get()
            use_punctuation = self.punctuation_var.get()
            num_speakers = int(self.num_speakers_var.get())
            logger.info(f"Settings: diarization={use_diarization}, punctuation={use_punctuation}, max_speakers={num_speakers}")
            
            # Initialize post-processor if punctuation restoration is enabled
            post_processor = None
            if use_punctuation and PUNCTUATION_AVAILABLE:
                logger.info("Loading punctuation model...")
                self.master.after(0, lambda: self.status_var.set("🔄 Loading punctuation model..."))
                post_processor = TranscriptPostProcessor()
                post_processor.load_model()
                logger.info("Punctuation model loaded")
            
            # Load audio and get duration for progress tracking
            logger.info(f"Loading audio file: {file_path}")
            self.master.after(0, lambda: self.status_var.set("🔄 Loading audio file..."))
            self.master.after(0, lambda: self.progress_var.set(PROGRESS_AUDIO_LOADED))
            
            audio = whisper.load_audio(file_path)
            audio_duration = len(audio) / whisper.audio.SAMPLE_RATE
            logger.info(f"Audio loaded: duration={audio_duration:.1f}s ({int(audio_duration//60)}m {int(audio_duration%60)}s)")
            
            # Update status with duration info and set progress to 10%
            duration_str = f"{int(audio_duration//60)}m {int(audio_duration%60)}s"
            self.master.after(0, lambda: self.status_var.set(
                f"🔄 Transcribing {duration_str} of audio... (please wait)"
            ))
            self.master.after(0, lambda: self.progress_var.set(PROGRESS_TRANSCRIPTION_START))
            
            # Estimate time based on model and device
            model_name = self.master.current_model_name or "medium"
            speed = MODEL_SPEED_FACTORS.get(model_name, DEFAULT_SPEED_FACTOR)
            if self.device == "cuda":
                speed *= GPU_SPEED_MULTIPLIER
            estimated_time = audio_duration / speed
            if use_diarization:
                estimated_time *= DIARIZATION_TIME_MULTIPLIER
            eta_str = f"~{int(estimated_time//60)}m {int(estimated_time%60)}s"
            self.master.after(0, lambda: self.time_var.set(f"Est: {eta_str}"))
            
            # Transcription options
            transcribe_options = {
                "verbose": False,
                "fp16": self.device == "cuda",  # Use FP16 on GPU for speed
            }
            
            if language:
                transcribe_options["language"] = language
            
            # Run transcription (this is synchronous - blocks until complete)
            # Whisper doesn't support real-time progress callbacks
            logger.info(f"Starting Whisper transcription with options: {transcribe_options}")
            whisper_start = time.time()
            result = self.master.model.transcribe(
                audio,
                **transcribe_options
            )
            whisper_elapsed = time.time() - whisper_start
            num_segments = len(result.get('segments', []))
            logger.info(f"Whisper completed in {whisper_elapsed:.1f}s - {num_segments} segments")
            
            # Check cancellation (only effective before/after transcription)
            if self.cancel_requested:
                logger.info("Cancellation requested, aborting")
                self.master.after(0, self.on_transcription_cancelled)
                return
            
            # Apply diarization if enabled
            if use_diarization:
                logger.info("Starting speaker diarization...")
                self.master.after(0, lambda: self.status_var.set("🗣️ Assigning speakers..."))
                self.master.after(0, lambda: self.progress_var.set(PROGRESS_DIARIZATION_START))
                
                # Check if full pyannote diarization is available
                hf_token = os.getenv("HF_TOKEN")
                use_full_diarization = PYANNOTE_AVAILABLE and hf_token and hf_token != HF_TOKEN_PLACEHOLDER
                logger.info(f"PYANNOTE_AVAILABLE={PYANNOTE_AVAILABLE}, HF_TOKEN set={bool(hf_token)}, use_full_diarization={use_full_diarization}")
                
                if use_full_diarization:
                    try:
                        logger.info("Initializing Pyannote diarization pipeline...")
                        self.master.after(0, lambda: self.status_var.set("🗣️ Running Pyannote AI diarization (accurate)..."))
                        
                        # Initialize and run pyannote diarization
                        diarization_pipeline = DiarizationPipeline(hf_token=hf_token, device=self.device)
                        logger.info("Loading Pyannote model...")
                        diarization_pipeline.load()
                        
                        # Run diarization with min/max range (auto-detect up to max)
                        logger.info(f"Running diarization on: {file_path}")
                        diarization_start = time.time()
                        diarization_segments = diarization_pipeline.diarize(
                            file_path,
                            min_speakers=1,
                            max_speakers=num_speakers
                        )
                        diarization_elapsed = time.time() - diarization_start
                        logger.info(f"Diarization completed in {diarization_elapsed:.1f}s - {len(diarization_segments)} segments")
                        
                        # Combine with Whisper transcript
                        logger.info("Assigning speakers to whisper segments...")
                        structurer = TranscriptStructurer(speaker_names=self.speaker_names)
                        segments = structurer.assign_speakers(
                            result.get("segments", []),
                            diarization_segments
                        )
                        logger.info(f"Merging consecutive segments...")
                        segments = structurer.merge_consecutive_speaker_segments(segments)
                        logger.info(f"Final segments: {len(segments)}")
                        
                        # Apply punctuation restoration if enabled
                        if post_processor:
                            logger.info("Restoring punctuation...")
                            self.master.after(0, lambda: self.status_var.set("✍️ Restoring punctuation..."))
                            self.master.after(0, lambda: self.progress_var.set(PROGRESS_PUNCTUATION_START))
                            punct_start = time.time()
                            segments = post_processor.process_segments(segments)
                            logger.info(f"Punctuation restored in {time.time() - punct_start:.1f}s")
                        
                        logger.info("Formatting final transcript...")
                        final_text = structurer.format_transcript(segments)
                        
                        # Add note that pyannote was used
                        notes_addition = "- ✅ **Used: Pyannote AI Diarization** (accurate speaker detection)"
                        if post_processor:
                            notes_addition += "\n- ✅ **Punctuation Restored** (using deepmultilingual-punctuation)"
                        final_text = final_text.replace(
                            "## Notes",
                            f"## Notes\n\n{notes_addition}"
                        )
                        
                    except Exception as e:
                        # Fall back to heuristic if pyannote fails
                        error_msg = str(e)[:80]
                        self.master.after(0, lambda: self.status_var.set(f"⚠️ Pyannote failed: {error_msg}"))
                        logger.error("Pyannote error: %s", e)
                        final_text = structure_whisper_only(
                            result,
                            num_speakers=num_speakers,
                            speaker_names=self.speaker_names if self.speaker_names else None,
                            post_processor=post_processor
                        )
                else:
                    # Use heuristic diarization
                    self.master.after(0, lambda: self.status_var.set("🗣️ Using heuristic speaker detection..."))
                    final_text = structure_whisper_only(
                        result,
                        num_speakers=num_speakers,
                        speaker_names=self.speaker_names if self.speaker_names else None,
                        post_processor=post_processor
                    )
            else:
                # Process segments and build full text (original behavior - no diarization)
                full_text_parts = []
                for i, segment in enumerate(result.get("segments", [])):
                    segment_text = segment.get("text", "").strip()
                    # Apply punctuation restoration to each segment
                    if segment_text and post_processor:
                        segment_text = post_processor.process_text(segment_text)
                    if segment_text:
                        full_text_parts.append(segment_text)
                        
                        # Update progress as we process segments (post-transcription)
                        end_time = segment.get("end", 0)
                        progress = min(100, (end_time / audio_duration) * 100)
                        elapsed = time.time() - start_time
                        
                        if progress > 0 and progress < 100:
                            eta = (elapsed / progress) * (100 - progress)
                            eta_str = f"ETA: {int(eta//60)}m {int(eta%60)}s"
                        else:
                            eta_str = ""
                        
                        # Update UI with segment
                        current_text = " ".join(full_text_parts)
                        self.master.after(0, lambda t=current_text, p=progress, e=eta_str: 
                                         self._update_live_transcript(t, p, e))
                
                # Final result
                final_text = result.get("text", "").strip()
                if not final_text and full_text_parts:
                    final_text = " ".join(full_text_parts)
            
            elapsed = time.time() - start_time
            logger.info(f"TRANSCRIPTION COMPLETE - Total time: {elapsed:.1f}s")
            logger.info("="*60)
            self.master.after(0, lambda: self.on_transcription_complete(final_text, elapsed, audio_duration))
            
        except Exception as exc:
            logger.exception(f"TRANSCRIPTION FAILED: {exc}")
            self.master.after(0, lambda: self.on_transcription_failed(str(exc)))

    def _update_progress_ui(self, progress, eta_str):
        """Update progress bar and ETA."""
        self.progress_var.set(progress)
        self.time_var.set(eta_str)
        self.status_var.set(f"🔄 Transcribing... {progress:.1f}%")

    def _update_live_transcript(self, text, progress, eta_str):
        """Update the transcript display in real-time."""
        self.progress_var.set(progress)
        self.time_var.set(eta_str)
        self.status_var.set(f"🔄 Transcribing... {progress:.1f}%")
        
        # Update text widget
        self.text_widget.config(state="normal")
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)  # Auto-scroll to bottom
        self.text_widget.config(state="disabled")
        
        # Update word count
        words = len(text.split())
        chars = len(text)
        self.word_count_var.set(f"Words: {words:,} | Characters: {chars:,}")

    def on_transcription_complete(self, text: str, elapsed: float, audio_duration: float):
        """Handle successful transcription completion."""
        self.is_transcribing = False
        self.progress_var.set(PROGRESS_COMPLETE)
        
        # Calculate stats
        speed_ratio = audio_duration / elapsed if elapsed > 0 else 0
        elapsed_str = f"{int(elapsed//60)}m {int(elapsed%60)}s"
        
        self.update_text(text)
        
        # Auto-save if enabled
        if self.auto_save_var.get():
            auto_name = Path(self.file_path.get()).stem or "transcript"
            auto_path = OUTPUT_DIR / f"{auto_name}_transcript.txt"
            auto_path.write_text(text or "", encoding="utf-8")
            save_msg = f" | Saved to {auto_path.name}"
        else:
            save_msg = ""
        
        self.save_button.config(state="normal")
        self.transcribe_button.config(state="normal")
        self.cancel_button.config(state="disabled")
        
        self.status_var.set(f"✅ Done in {elapsed_str} ({speed_ratio:.1f}x realtime){save_msg}")
        self.time_var.set("")
        self.transcript_ready = True

    def on_transcription_cancelled(self):
        """Handle transcription cancellation."""
        self.is_transcribing = False
        self.transcribe_button.config(state="normal")
        self.cancel_button.config(state="disabled")
        self.status_var.set("⏹️ Transcription cancelled. Partial results shown above.")
        self.time_var.set("")
        self.transcript_ready = True
        self.save_button.config(state="normal")

    def on_transcription_failed(self, error_message: str):
        """Handle transcription failure."""
        self.is_transcribing = False
        self.transcribe_button.config(state="normal")
        self.cancel_button.config(state="disabled")
        self.progress_var.set(0)
        self.time_var.set("")
        self.status_var.set("❌ Transcription failed.")
        messagebox.showerror("Transcription failed", error_message)

    def update_text(self, text: str):
        """Update the transcript text widget."""
        self.text_widget.config(state="normal")
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert(tk.END, text or "(No text returned)")
        self.text_widget.config(state="disabled")
        
        # Update word count
        words = len(text.split()) if text else 0
        chars = len(text) if text else 0
        self.word_count_var.set(f"Words: {words:,} | Characters: {chars:,}")

    def clear_transcript(self):
        """Clear the transcript display."""
        self.text_widget.config(state="normal")
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.config(state="disabled")
        self.word_count_var.set("Words: 0 | Characters: 0")
        self.progress_var.set(0)

    def save_transcript(self):
        """Save transcript to a user-selected file."""
        if not self.transcript_ready:
            messagebox.showwarning("Nothing to save", "Run a transcription first.")
            return
        default_name = Path(self.file_path.get()).stem + ".txt"
        target_path = filedialog.asksaveasfilename(
            defaultextension=".txt", 
            initialfile=default_name,
            filetypes=[
                ("Text files", "*.txt"),
                ("SRT Subtitles", "*.srt"),
                ("All files", "*.*")
            ]
        )
        if not target_path:
            return
        text = self.text_widget.get("1.0", tk.END).strip()
        Path(target_path).write_text(text, encoding="utf-8")
        self.status_var.set(f"💾 Saved to {target_path}")


class DesktopApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("800x700")
        self.minsize(700, 600)
        self.model = None
        self.current_model_name = None
        
        # Set app icon (optional)
        try:
            self.iconbitmap(default="")
        except:
            pass

        self.transcription_frame = TranscriptionFrame(self)
        self.transcription_frame.pack(fill="both", expand=True)

        # Load initial model
        threading.Thread(target=self.load_model, args=(MODEL_NAME,), daemon=True).start()

    def load_model(self, model_name: str = MODEL_NAME):
        """Load Whisper model with GPU support."""
        try:
            device, device_info = get_device()
            
            self.after(0, lambda: self.transcription_frame.status_var.set(
                f"⏳ Loading '{model_name}' model on {device.upper()}... (this may take a moment)"
            ))
            
            # Load model with specified device
            model = whisper.load_model(model_name, device=device)
            
            self.model = model
            self.current_model_name = model_name
            self.after(0, lambda: self.on_model_ready(model_name))
            
        except Exception as exc:
            error_msg = str(exc)
            
            # Provide helpful error messages
            if "CUDA" in error_msg or "cuda" in error_msg:
                error_msg += "\n\nTo fix: Install PyTorch with CUDA support:\npip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            elif "out of memory" in error_msg.lower():
                error_msg += "\n\nTry using a smaller model (e.g., 'small' or 'base')"
            
            self.after(0, lambda: messagebox.showerror("Model error", error_msg))
            self.after(0, lambda: self.transcription_frame.status_var.set("❌ Failed to load model"))

    def on_model_ready(self, model_name: str):
        self.transcription_frame.enable_model_usage(model_name)


def main():
    # Log GPU info at startup
    device, device_info = get_device()
    logger.info("="*60)
    logger.info("  Audio Transcript - Whisper Desktop App")
    logger.info("  %s", device_info)
    logger.info("="*60)
    
    app = DesktopApp()
    app.mainloop()


if __name__ == "__main__":
    main()
