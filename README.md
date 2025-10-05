# OVSD-Handy

**OVSD-Handy** (OpenVINO Stable Diffusion Handy) is a universal wrapper for running any [Optimum Intel OpenVINO Stable Diffusion pipelines](https://huggingface.co/docs/optimum/intel/openvino/stable_diffusion) with a single handy function call.  
It automatically detects the correct pipeline class (`txt2img`, `img2img`, `inpainting`, `SDXL`, …), manages seeds, and includes a built-in NSFW content detector.

---

## ✨ Features

- 🚀 **One function, any model**: Run `txt2img`, `img2img`, `inpainting`, and `SDXL` with the same API.  
- 🔍 **Auto pipeline detection**: Detects the right pipeline type for the given model.  
- 🎲 **Smart seed handling**: Supports int, string, or random seed generation.  
- 🛡️ **NSFW guard**: Built-in unsafe prompt detection before inference.  
- ⚡ **Optimized for CPU**: Automatically compiles with OpenVINO for fast inference.  
- 🔄 **Model caching**: Reloads pipeline only if model_id changes.  

---

## 📦 Installation

From PyPI:
```bash
pip install ovsd-handy
