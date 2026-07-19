# Tissue Segmentation Pipeline (mIHC)

## Overview
This repository contains a modular, CLI-based pipeline for multi-label semantic segmentation and spatial quantification of tumor tissue microarrays (TMAs) and Whole Slide Images (WSIs). It is designed to handle gigapixel images efficiently, bridging deep learning architectures with mathematical morphology.

## Architecture
The codebase has been refactored from a monolithic script into a professional, modular framework:
* **`models/`**: Isolates the PyTorch architecture (U-Net) used for broad structural and specialized immune cell segmentation.
* **`utils/`**: Contains pure mathematical logic (`scikit-image`, `scipy`), including watershed instance segmentation and strict RGB signal deconvolution rules (e.g., CK vs MUM1 resolution).
* **`inference.py`**: The core execution engine. It implements overlapping sliding windows, aggressive garbage collection, and out-of-core memory mapping to prevent Out-Of-Memory (OOM) errors during large-scale inference.
* **`main.py`**: The CLI entry point orchestrating dynamic parameters.

## Installation
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt

Usage
The pipeline is entirely parameterized. Do not hardcode paths. Execute the pipeline via the command line:
python main.py \
  --input_dir /path/to/your/tiffs \
  --output_dir /path/to/save/results \
  --model_a_path /path/to/weights/unet_structural.pth \
  --model_b_path /path/to/weights/unet_immune.pth \
  --device cuda

Features & Engineering Choices
Memory Management: Uses pyvips.cache_set_max(0) and numpy.memmap to write predictions directly to disk, bypassing RAM limitations.
Edge Artifact Prevention: Implements contextual padding during the tiling phase.
Hierarchical Exclusion: Enforces strict biological constraints (e.g., tumor structures acting as physical barriers for certain cell types) before instance segmentation.
