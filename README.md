# ⚡ Gemini & Veo Watermark Remover

<p align="center">
  <a href="https://axtrelis.com">
    <img src="https://img.shields.io/badge/POWERED%20BY-AXTRELIS.COM-FF4500?style=for-the-badge&logo=rocket&logoColor=white" height="42" alt="Axtrelis" />
  </a>
</p>

<h3 align="center">
  🌐 Built by <a href="https://axtrelis.com"><b>AXTRELIS</b></a> (Visit <a href="https://axtrelis.com">axtrelis.com</a>)
</h3>

<p align="center">
  <b>Mathematical Closed-Form Reverse Alpha Blending &amp; Deterministic Watermark Eradication</b><br/>
  Zero blur. Zero smearing. 100% texture preservation for Google Gemini (Imagen 3) &amp; Google Veo videos and images.
</p>

---

## 🌟 Overview

Most watermark removal tools rely on naive inpainting (e.g. OpenCV `cv2.inpaint` with Telea or Navier-Stokes) or generative diffusion inpainting. On videos and high-detail images, these methods **destroy background textures**, smear edges, and produce flickering blur artifacts.

Google Gemini and Google Veo do **not** randomize their watermark. They composite a static, deterministic 4-pointed sparkle (or Veo logo) using standard **linear alpha transparency over pure white ($W = 255$)**. 

Because this is a linear optical blend, **the original background pixels underneath the watermark are never lost**. They can be mathematically recovered with zero loss in clarity using **Reverse Alpha Blending**.

This engine delivers:
- **Zero-Blur Mathematical Inversion**: Restores background geometry, texture details (gravel, metal ladders, tyre treads, skin pores) with surgical precision.
- **Deterministic Coordinate Catalog**: Pre-indexed geometry for 1080p, 720p, 4K, and still images with subpixel NCC (Normalized Cross Correlation) auto-detection.
- **Surgical Perimeter Repair**: Automatically eliminates H.264 compression quantization rings along the outer 1–2 px boundary without touching internal pixels.
- **Native Audio Stream Preservation**: Re-muxes AAC, MP3, and Opus streams untouched via FFmpeg with zero audio re-encoding or sync drift.
- **Blazing Speed**: $> 35\text{ fps}$ end-to-end on a single CPU core without GPU requirements ($> 150\text{ fps}$ pure arithmetic).
- **Optional Visual QA**: Built-in VLM / AI verification hook (Jev-Omni classifier).

---

## 🔬 How The Methods Work

### Method 1: Reverse Alpha Blending (`alpha_reverse`) — *[Default & Recommended]*

#### 1. The Mathematical Inverse
When Google's rendering engine stamps a watermark onto an image or video frame, it applies linear alpha blending:

$$I = \alpha \cdot W + (1 - \alpha) \cdot B$$

Where:
* $I$ = The observed pixel value in the watermarked output.
* $W = 255$ = The watermark logo color (pure white).
* $B$ = The original clean background pixel.
* $\alpha \in [0.0, 1.0]$ = The transparency/opacity value from the logo's alpha channel.

Because $\alpha$ and $W$ are known constants from the calibrated watermark template, we solve directly for $B$:

$$B = \frac{I - \alpha \cdot 255}{1 - \alpha}$$

This is a **closed-form algebraic inversion**. No pixels are guessed, hallucinated, or interpolated from surrounding regions. If a background pixel was dark gray ($B = 40$) and blended with $30\%$ white opacity ($\alpha = 0.30$), it produced $I = 104$. Applying the formula reconstructs $(104 - 0.3 \times 255) / 0.7 = 39.3 \approx 40$ bit-exact.

#### 2. Surgical Perimeter Edge Repair
While uncompressed PNG frames invert cleanly, compressed video codecs (such as H.264/H.265) quantize high-contrast high-frequency edges, creating a faint 1-pixel boundary ringing artifact around the watermark perimeter.

To eliminate this without smudging the interior:
1. We compute an edge mask strictly where $\alpha \in [0.015, 0.09]$ (the 1–2 px outer hairline boundary).
2. We run a micro-radius Telea inpaint ($r=2$) **only** across this 1-pixel hairline edge.
3. The interior of the watermark ($> 95\%$ of the area) remains 100% pure reverse alpha restoration.

---

### Method 2: Telea Inpainting (`telea`)
* **Reference**: Alexandru Telea, *"An Image Inpainting Technique Based on the Fast Marching Method"* (2004).
* **How it works**: Treats the entire watermark as an unknown hole ($\Omega$). It marches inward from the boundary ($\partial\Omega$) and estimates unknown pixel values by weighted average of known neighboring boundary pixels along gradient vectors.
* **When to use**: Good for opaque stamps or solid backgrounds where underlying data is truly destroyed. Suboptimal for alpha-blended transparent watermarks because it ignores the visible underlying pixels and creates a visible blur.

---

### Method 3: Navier-Stokes Inpainting (`ns`)
* **Reference**: Bertalmio, Bertozzi, Sapiro, *"Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting"* (2001).
* **How it works**: Formulates inpainting as a 2D fluid dynamics problem, propagating isophote lines (lines of equal gray level) continuously from the outside inward by conserving vorticity:

$$\frac{\partial \omega}{\partial t} + \mathbf{v} \cdot \nabla \omega = \nu \nabla^2 \omega$$

* **When to use**: Smooth synthetic gradients or artistic artwork where isophotes need to continue uninterrupted. Slower than Telea, and still destroys underlying high-frequency textures.

---

### Method 4: FFmpeg Rectangular Delogo (`rectangular`)
* **How it works**: Uses FFmpeg's built-in `delogo=x:y:w:h` filter. It takes a bounding box and interpolates pixels from the 4 surrounding outer borders inward.
* **When to use**: Quick command-line scripts without Python dependencies. Disadvantage: blurs the entire enclosing bounding box.

---

### Method 5: FFmpeg Glyph Mask Removelogo (`glyph_mask`)
* **How it works**: Uses FFmpeg's `removelogo=filename=mask.png` filter. It interpolates only the pixels corresponding to non-zero values in a binary mask.
* **Caveat**: Must be executed with `format=yuv444p` before the filter to prevent chroma 4:2:0 subsampling from causing pink/magenta edge drift.

---

## 📐 Deterministic Geometry Catalog

Because Google renders watermarks predictably, the engine automatically matches the resolution to the pre-calibrated catalog and verifies alignment via **Normalized Cross Correlation (NCC)**:

| Profile | Output Resolution | Size ($W \times H$) | Margins ($R, B$) | Top-Left ($X_0, Y_0$) |
| :--- | :--- | :--- | :--- | :--- |
| **Veo 1080p Inset** *(Default)* | $1920 \times 1080$ | $72 \times 72\text{ px}$ | $144\text{ px}, 144\text{ px}$ | $(1704, 864)$ |
| **Veo 1080p Standard** | $1920 \times 1080$ | $72 \times 72\text{ px}$ | $108\text{ px}, 108\text{ px}$ | $(1740, 900)$ |
| **Veo 720p Inset** | $1280 \times 720$ | $48 \times 48\text{ px}$ | $96\text{ px}, 96\text{ px}$ | $(1136, 576)$ |
| **Veo 720p Standard** | $1280 \times 720$ | $48 \times 48\text{ px}$ | $72\text{ px}, 72\text{ px}$ | $(1160, 600)$ |
| **Gemini Image Large** | $> 1024 \times 1024$ | $96 \times 96\text{ px}$ | $64\text{ px}, 64\text{ px}$ | $(W - 160, H - 160)$ |
| **Gemini Image Small** | $\le 1024 \times 1024$ | $48 \times 48\text{ px}$ | $32\text{ px}, 32\text{ px}$ | $(W - 80, H - 80)$ |
| **Arbitrary Aspect Ratio** | $W \times H$ | Proportional | Proportional | Auto NCC Detection |

---

## 🚀 Quickstart & Installation

### Requirements
- Python 3.9+
- `ffmpeg` installed and added to system `PATH`

```bash
git clone https://github.com/konradschrein-star/Gemini-Watermark-Remover.git
cd Gemini-Watermark-Remover
pip install -r requirements.txt
```

---

## 💻 CLI Usage Examples

### 1. Basic Video Eradication (Default: Reverse Alpha Blending)
```bash
python cli.py video.mp4 -o video_clean.mp4
```

### 2. Clean Video with Custom Output & Jev-Omni AI Verification
```bash
python cli.py input_1080p.mp4 -o output_clean.mp4 --verify --verifier jev-omni
```

### 3. Clean Still Image
```bash
python cli.py gemini_photo.png -o gemini_clean.png
```

### 4. Machine-Readable JSON Output for Automated Pipelines
```bash
python cli.py video.mp4 --json
```

**JSON Output:**
```json
{
  "status": "success",
  "input": "video.mp4",
  "output": "video_cleaned.mp4",
  "frames_processed": 301,
  "fps": 30.0,
  "resolution": "1920x1080",
  "watermark_type": "gemini_sparkle",
  "bbox": {
    "x0": 1704,
    "y0": 864,
    "x1": 1776,
    "y1": 936
  },
  "verification": {
    "is_clean": true,
    "confidence": 0.98,
    "label": "Clean (No residual detected)",
    "backend": "local_cv"
  }
}
```

### 5. Manual Bounding Box Override
If you have a custom video resolution or crop:
```bash
python cli.py video.mp4 --region 1704,864,72,72
```

---

## 🐍 Python API Examples

### Video Processing
```python
from gemini_watermark_remover import (
    process_video,
    WatermarkType,
    InpaintMethod
)

result = process_video(
    input_path="input_video.mp4",
    output_path="cleaned_video.mp4",
    watermark_type=WatermarkType.AUTO,
    method=InpaintMethod.ALPHA_REVERSE, # Zero blur mathematical inversion
    verify=True,
    verifier_backend="local_cv"
)

print(f"Cleaned {result['frames_processed']} frames in {result['resolution']}")
print(f"Bounding box: {result['bbox']}")
```

### Still Image Processing
```python
from gemini_watermark_remover import process_image

result = process_image(
    input_path="portrait.png",
    output_path="portrait_clean.png"
)
print("Saved clean image:", result["output"])
```

### Frame-by-Frame Custom Pipeline
```python
import cv2
from gemini_watermark_remover.detector import detect_watermark
from gemini_watermark_remover.inpainter import remove_watermark_frame, InpaintMethod

# Detect once from first/middle frame
cap = cv2.VideoCapture("input.mp4")
ret, frame = cap.read()
w_type, bbox, mask, alpha_map = detect_watermark(frame)

# Clean single frame in real-time
cleaned_frame = remove_watermark_frame(
    frame,
    bbox=bbox,
    mask=mask,
    method=InpaintMethod.ALPHA_REVERSE,
    alpha_map=alpha_map
)
```

---

## 📊 Benchmark & Quality Verification

Tested on Google Gemini 1080p output (`Video senza titolo.mp4`, 301 frames @ 30 fps, 1920×1080):

| Metric | Previous Inpainting (`cv2.inpaint`) | This Engine (`alpha_reverse`) |
| :--- | :--- | :--- |
| **Texture Preservation** | ❌ Blurred / smudged | ✅ **100% Intact** (metal rungs, gravel, tyre chains preserved) |
| **Ghosting / Halos** | ❌ Visible gray shadow | ✅ **Zero Ghosting** |
| **Edge Transition** | ❌ Discontinuous seam | ✅ **Seamless Hairline** via perimeter repair |
| **Audio Stream** | ⚠️ Re-encoded | ✅ **Bit-for-Bit Copy** (AAC / MP3 / Opus untouched) |
| **Throughput** | ~25 fps | ✅ **> 35 fps end-to-end** (> 150 fps in-memory arithmetic) |
| **Detection Match** | Arbitrary bounding box | ✅ **NCC = 0.952 Peak Confidence** |

---

## 🌐 Community & Credits

- Maintained by **Konrad** & the engineering team at [Axtrelis](https://axtrelis.com).
- Visit [**axtrelis.com**](https://axtrelis.com) for production AI pipelines, automation tooling, and creative engineering.
- Mathematical reverse alpha blending catalog inspired by research from [GargantuaX](https://github.com/GargantuaX/gemini-watermark-remover) and [dearabhin](https://github.com/dearabhin/gemini-watermark-remover).

---

## ⚖️ License

Distributed under the Apache 2.0 License. See `LICENSE` for details.
