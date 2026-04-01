# Simple Captioner

A minimal media captioning tool powered by **[Qwen2.5/3 VL Instruct](https://huggingface.co/Qwen/)** from Alibaba Group.

This tool uses a Gradio UI to batch process folders of **images and videos** and generate descriptive captions.

Written by [Olli S.](https://github.com/o-l-l-i)

---

![Splash image](/images/screenshot.png)

## ✨ Features

Version 1.0.2

- ✅ Uses `Qwen2.5/3 VL Instruct and Qwen3.5 4B/9B` for high-quality understanding
- ✅ Support for:
  - Qwen/Qwen3.5-4B
  - Qwen/Qwen3.5-9B
  - Qwen/Qwen3-VL-4B-Instruct
  - Qwen/Qwen3-VL-8B-Instruct
  - Qwen/Qwen2.5-VL-3B-Instruct
  - Qwen/Qwen2.5-VL-7B-Instruct
- ✅ Flash attention 2 support (with toggle)
- ✅ Quantization via BitsAndBytes (None / 8-bit / 4-bit)
- ✅ Caption multiple images or videos from a selected folder
- ✅ Sub-folder support
- ✅ Supports prompt customization
- ✅ "Summary Mode" and "One-Sentence Mode" options for different caption styles
- ✅ Can skip already-captioned images
- ✅ Image previews with real-time progress
- ✅ Abort long runs safely

---

## Requirements

- Python 3.9+
- A modern NVIDIA GPU with CUDA (tested on Ampere and newer)
- ~16GB VRAM recommended for smooth operation

---

## Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/o-l-l-i/simple-captioner.git
   cd simple-captioner

2. **Create a virtual environment (optional but recommended)**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install the dependencies**:

    ```bash
    pip install -r requirements.txt

4. **Install Torch with GPU support**:
   - You have to install GPU compatible Torch yourself, get it from here:
   - [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
   - Copy the "Run this Command" string from the page after selecting correct version.
     - i.e. if you have Cuda 12.8, select that option. (Windows, Pip, Python, CUDA 12.8.)

    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

5. **Install Triton**:

    On Windows, install [woctordho's Triton fork for Windows](https://github.com/woct0rdho/triton-windows)

    ```bash
    pip install triton
    pip install triton-windows # On Windows use this

6. **Run the app**:

    ```bash
    python app.py

7. **To run this app later**:

    - When you need to return back to use this, the virtual environment (venv) needs to be activated again.
    - Use/modify the included start up scripts.

    **Windows**:
    - run_app.bat

    ```bash
    @echo off
    call venv\Scripts\activate
    python app.py
    ```

    **Linux/macOS**:
    - run_app.sh

    ```bash
    #!/bin/bash
    source venv/bin/activate
    python app.py
    ```

    Make it executable:
    ```bash
    chmod +x run_app.sh
    ```

## Model Files

When you run the app for the first time, the model (Qwen/Qwen2.5-VL-7B-Instruct) is automatically downloaded from Hugging Face. This download is cached locally, so subsequent runs are much faster and offline-compatible.

By default, Hugging Face stores downloaded models in:


```bash
Linux/macOS: ~/.cache/huggingface/

Windows: C:\Users\<YourUsername>\.cache\huggingface\
```

You can inspect, manage, or clear this cache manually, or change the location by setting the HF_HOME environment variable:


```bash
export HF_HOME=/custom/path/to/huggingface
# On Windows: set HF_HOME=E:\huggingface_cache
```

This is useful if you're working with limited disk space or want to centralize model caches across multiple projects.

---

## Video Support Note
To enable video processing, make sure qwen-vl-utils is installed.
On Linux:

```bash
pip install qwen-vl-utils[decord]==0.0.8

```

```bash
On other platforms (Windows/macOS):
pip install qwen-vl-utils
```

This will fall back to using torchvision for video loading if decord does not work, which is slower.
For better performance, [you can try to install decord from source](https://github.com/dmlc/decord)

## Usage Notes
- Place your images in a folder (recursively scanned, subfolders are supported.)
- Text files with the same name (e.g. image1.jpg → image1.txt) are created alongside the images.
- Use the “Skip already captioned” checkbox to avoid reprocessing.
- Captions can be styled with prompt modifiers or sentence-length constraints.

---

## Customization

- Prompt handling is adjustable with toggles.
- Modify the base prompt or model behavior in generate_caption() inside the code.
- Want more control over output format? Adjust the file writing or UI code.

---

## Troubleshooting

- Make sure you’re using a CUDA-compatible GPU.
- On Windows you have to install GPU compatible Torch yourself, get it from here:
  - [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
  - Select a Torch version which matches your CUDA version.
- If VRAM usage is too high, reduce max_tokens. This is only tested on 3090 and 5090, but I did monitor the VRAM usage.

---

## Versions

- **1.0.2 - 2026-4-1**
  - Add support for Qwen 3.5 4B and 9B.
  - Updated requirements (transformers to >=5.4.0 and bitsandbytes to >=0.46.1)

- **1.0.1 - 2025-10-15**
  - Model dropdown with multiple model support.
  - Quantization (None / 8-bit / 4-bit.)
  - Attention implementation toggle (flash attention 2 supported) + auto-fallback to `eager`
  - Model is no longer loaded at import; loads via UI or on app UI start.
  - Defaults to Qwen/Qwen3-VL-8B-Instruct, this can be memory intensive, so use quantization or 4B model.
  - Improved VRAM cleanup.

- **1.0.0**
  - Initial release.
  - Qwen/Qwen2.5-VL-7B-Instruct support for image and video captioning.

---

## Early Development Notice

This project is currently in a very early phase of development. While it aims to provide useful image and video captioning capabilities, you may encounter bugs, unexpected behavior, or incomplete features.

If you run into any issues:

- Please check the console or logs for error messages.
- Try to use supported media formats as listed.
- Feel free to report problems or request features via the project’s GitHub Issues page.

---

## License & Usage Terms

Copyright (c) 2025 Olli Sorjonen

This project is source-available, but not open-source under a standard open-source license, and not freeware.
You may use and experiment with it freely, and any results you create with it are yours to use however you like.

However:

Redistribution, resale, rebranding, or claiming authorship of this code or extension is strictly prohibited without explicit written permission.

Use at your own risk. No warranties or guarantees are provided.

The only official repository for this project is: 👉 https://github.com/o-l-l-i/simple-captioner

---

## Author

Created by [@o-l-l-i](https://github.com/o-l-l-i)
