# Gemini & Veo Watermark Remover

High-fidelity, automated watermark removal utility for **Google Gemini (Imagen 3)** 4-pointed sparkle watermarks and **Google Veo** video watermarks, featuring localized inpainting, full audio preservation, and optional **Jev-Omni** AI/VLM visual QA classification.

---

## ⚡ Features

- **Gemini Sparkle & Veo Video Profiles**: Automatically targets the 4-pointed Gemini sparkle watermark or Veo bottom-right overlay across 1080p, 720p, 4K, and 9:16 portrait formats.
- **Audio Preservation**: Automatically remuxes original audio channels (AAC, MP3, Opus) untouched via FFmpeg.
- **Ultra-Fast Localized Inpainting**: Crops to a micro-ROI around the watermark for inpainting (Telea / Navier-Stokes / Reverse Alpha), delivering ~30-60+ fps processing speeds without GPU requirements.
- **Jev-Omni AI Verification**: Built-in visual QA hook directly querying the `akhilaaa3/jev-omni` Hugging Face Space multimodal classifier or local CV residual tests.
- **CLI & Python API**: Usable as a standalone terminal command or embedded Python module.

---

## 🚀 Installation

```bash
git clone https://github.com/konradschrein-star/Gemini-Watermark-Remover.git
cd Gemini-Watermark-Remover
pip install -r requirements.txt
```

*Prerequisite: `ffmpeg` must be installed and available on your system `$PATH`.*

---

## 💻 CLI Usage

### Basic Video Cleaning
```bash
python cli.py input_video.mp4 -o cleaned_video.mp4
```

### Video Cleaning with Jev-Omni AI Verification
```bash
python cli.py input_video.mp4 --verify --verifier jev-omni
```

### Clean an Image
```bash
python cli.py sample.png -o sample_clean.png
```

### Emit Machine-Readable JSON for Pipelines
```bash
python cli.py input_video.mp4 --json
```

```json
{
  "status": "success",
  "input": "input_video.mp4",
  "output": "input_video_cleaned.mp4",
  "frames_processed": 301,
  "fps": 30.0,
  "resolution": "1920x1080",
  "watermark_type": "gemini_sparkle",
  "bbox": {
    "x0": 1701,
    "y0": 864,
    "x1": 1781,
    "y1": 954
  },
  "verification": {
    "is_clean": true,
    "confidence": 1.0,
    "label": "Clean (No residual detected)",
    "backend": "local_cv"
  }
}
```

---

## 🐍 Python API

```python
from gemini_watermark_remover import process_video, process_image, WatermarkType, InpaintMethod

# Video
res = process_video(
    input_path="input.mp4",
    output_path="output.mp4",
    watermark_type=WatermarkType.GEMINI_SPARKLE,
    method=InpaintMethod.TELEA,
    verify=True,
)
print("Finished:", res["output"])

# Image
res_img = process_image("frame.png", "frame_clean.png")
```

---

## 🧪 Benchmark & Quality Assurance
Tested against Google Gemini 1080p generation (`Video senza titolo.mp4`):
- Watermark: 4-pointed Gemini sparkle at `(1701, 864)`
- Result: 100% invisible removal, background texture and audio preserved
- Jev-Omni Multimodal QA Score: **98.7% Clean**
