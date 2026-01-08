# Audio Transcript

A Windows-friendly desktop application for securely transcribing audio files with Whisper. Everything runs locally so private recordings never leave your machine.

## 🚀 Quick Start (Recommended)

**One command does everything** — creates virtual environment, installs all dependencies, and launches the app:

```powershell
git clone https://github.com/josejalvarezm/audio-transcript.git
cd audio-transcript
.\setup-and-run.ps1
```

That's it! The script automatically installs Python packages, PyTorch with GPU support, and optional enhancements.

> **Note**: For speaker diarization, you'll need a free HuggingFace token — see [Speaker Diarization Setup](#speaker-diarization-setup) below.

---

## Features

- **Whisper Large v3 Turbo** (default) — Best accuracy/speed trade-off
- Support for WAV, MP3, M4A, FLAC, OGG, and video files (MP4, MKV, AVI)
- **Speaker Diarization** — Identify who said what using Pyannote 3.1 AI
- **Punctuation Restoration** — Automatically adds punctuation and capitalisation
- **Confidence Scores** — Shows speaker assignment confidence with warnings
- Automatic transcript saving to `transcripts/` folder
- Simple Tkinter UI with real-time progress

## Requirements

- **Windows 10/11**
- **Python 3.10+** (3.11 or 3.13 recommended)
- **NVIDIA GPU** with 6GB+ VRAM for best performance (CPU mode also works)
- Optional: FFmpeg on `PATH` for audio format conversion

---

## Manual Setup (Alternative)

If you prefer manual installation over the automated script:

### 1. Clone and create virtual environment

```powershell
git clone https://github.com/josejalvarezm/audio-transcript.git
cd audio-transcript

python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install PyTorch

Choose the command that matches your hardware:

```powershell
# NVIDIA GPU with CUDA 12.6 (recommended for RTX 30/40 series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# NVIDIA GPU with CUDA 12.4 (older driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CPU only (no NVIDIA GPU)
pip install torch torchvision torchaudio
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Install optional enhancements (recommended)

```powershell
# Punctuation restoration (highly recommended for readable output)
pip install deepmultilingualpunctuation

# Speaker diarization (requires HuggingFace token - see below)
pip install pyannote.audio
```

### 5. Configure environment (optional)

```powershell
Copy-Item .env.example .env
```

Edit `.env` to customize settings. See [Configuration](#configuration) for details.

### 6. Run the app

```powershell
python -m app.main

# Or use the quick-run script (after initial setup):
.\run.ps1
```

### 7. Verify installation (optional)

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import whisper; print('Whisper: OK')"
```

---

## Speaker Diarization Setup

Speaker diarization identifies *who* said *what* in multi-speaker recordings. It requires a free HuggingFace account:

1. **Create account**: Go to [huggingface.co](https://huggingface.co/join)
2. **Get token**: Visit [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → Create new token
3. **Accept model terms**: Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and click "Agree"
4. **Add token to .env**:

   ```powershell
   Copy-Item .env.example .env
   # Edit .env and add: HF_TOKEN=hf_your_token_here
   ```

> **Note**: Diarization is optional. The app works without it — transcription will just show all text without speaker labels.

## Usage

1. Launch the program
2. Click **Select audio file** to choose a recording
3. Configure options:
   - **Model**: Choose from tiny → large-v3-turbo (default)
   - **Speaker Diarization**: Enable to identify speakers
   - **Restore Punctuation**: Enable for readable output
4. Press **Transcribe** and wait for completion
5. The transcript automatically saves to `transcripts/<filename>_transcript.txt`
6. Use **Save transcript** to export a copy to any directory

## Configuration

The application reads the following environment variables (via `.env`):

| Name | Default | Description |
| ---- | ------- | ----------- |
| `WHISPER_MODEL` | `large-v3-turbo` | Whisper model variant |
| `OUTPUT_DIR` | `transcripts` | Folder where transcripts are auto-saved |
| `HF_TOKEN` | (none) | HuggingFace token for Pyannote diarization |

## Models and Performance

| Model | Relative Speed | Accuracy | GPU Memory |
| ----- | -------------- | -------- | ---------- |
| tiny | 32x | Baseline | ~1GB |
| base | 16x | +10% | ~1GB |
| small | 6x | +20% | ~2GB |
| medium | 2x | +30% | ~5GB |
| large-v3 | 1x | +45% | ~10GB |
| **large-v3-turbo** (default) | 1.5x | +42% | ~6GB |

## Optional: Audio Preprocessing

For noisy recordings, you can install optional preprocessing:

```powershell
pip install noisereduce soundfile
```

This enables noise reduction and volume normalisation.

## Optional: whisper.cpp CLI

The original `setup-whisper.ps1` script remains available if you want the high-performance whisper.cpp CLI. Run it to rebuild the C++ executable and model artifacts used outside the desktop UI.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'whisper'` | Run `pip install -r requirements.txt` |
| `torch.cuda.is_available()` returns `False` | Reinstall PyTorch with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126` |
| `CUDA out of memory` | Use a smaller model (medium, small, or base) in the dropdown |
| Transcription is slow | Ensure you're using GPU, not CPU. Check with `python -c "import torch; print(torch.cuda.is_available())"` |
| No punctuation in output | Install: `pip install deepmultilingualpunctuation` |
| Speaker diarization not working | 1) Install: `pip install pyannote.audio` 2) Set `HF_TOKEN` in `.env` 3) Accept terms at huggingface.co |
| `RuntimeError: operator torchvision::nms does not exist` | PyTorch version mismatch. Reinstall: `pip uninstall torch torchvision torchaudio -y && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126` |
| Virtual environment issues after moving folder | Delete `.venv` and run `.\setup-and-run.ps1` again |

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `setup-and-run.ps1` | **First-time setup** — creates venv, installs everything, launches app |
| `run.ps1` | **Quick launch** — activates venv and runs app (after initial setup) |
| `setup-whisper.ps1` | Optional: builds whisper.cpp CLI for command-line usage |

## License

MIT License — see [LICENSE](LICENSE) for details.
