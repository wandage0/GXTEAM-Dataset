# GXETAM-Dataset LoRA Training and Architectural Style Transfer Complete Workflow

This document provides a complete workflow based on the **GXETAM-Dataset**, **SD-Trainer framework**, and **ComfyUI tool**, covering LoRA model training for Guangxi ethnic traditional architectural styles and architectural style transfer. Suitable for cultural heritage digital preservation, architectural design innovation, and other scenarios. The dataset can be obtained from: https://huggingface.co/datasets/wanmuzi/GXETAM_Dataset

## 1. Prerequisites

### 1.1 Hardware Requirements

- GPU: NVIDIA graphics card, VRAM ≥ 16GB

### 1.2 Software Environment

1. Operating System: Windows 10/11 or Linux (Ubuntu 22.04+ recommended)
2. Python: 3.10 (recommended to use Anaconda to create a virtual environment)

### Environment Setup (Windows Example)

```bash
# 1. Create and activate Anaconda virtual environment
conda create -n lora_train python=3.10
conda activate lora_train

# 2. Clone SD-Trainer repository (with submodules)
git clone --recurse-submodules https://github.com/Akegarasu/lora-scripts.git
cd lora-scripts

# 3. Install dependencies (Mainland China users use install-cn.ps1)
./install-cn.ps1 # Windows
# Linux/Mac users run: bash install-cn.sh
```

## 2. Step 1: Download GXETAM-Dataset

Download the dataset from Hugging Face Hub for subsequent LoRA model training:

```bash
# Install huggingface-hub tool
pip install huggingface-hub

# Download dataset to local (replace <your_dataset_directory> with actual path)
huggingface-cli download wanmuzi/GXETAM_Dataset --local-dir <your_dataset_directory>/GXETAM_Dataset --local-dir-use-symlinks False
```

### Dataset Preprocessing

1. Extract the dataset and filter clear, stylistically typical architectural images from `main_dataset`
2. Copy the filtered images to the **training data directory**

## 3. Step 2: Label Processing via SD-Trainer GUI

SD-Trainer provides a visual UI to complete **automatic tagging** and **manual label optimization** without command line operations:

### 3.1 Launch SD-Trainer GUI

```bash
# Windows launch Chinese interface
./run_gui_cn.ps1

# Linux/Mac launch Chinese interface: bash run_gui_cn.sh
```

After launching, access `http://127.0.0.1:28000` in your browser to enter the SD-Trainer main interface (includes three core modules: "Tensorboard", "WD 1.4 Tagger", and "Tag Editor").

### 3.2 Automatic Tagging (via "WD 1.4 Tagger" module)

1. Select the "WD 1.4 Tagger" module on the left side of the interface
2. Click "Select Image Directory" and import `<your_training_data_directory>`
3. Configure tagging parameters (key parameters): Model: select `wd14-vit-v2-git` (default), minimum confidence, and output format
4. Click "Start" to automatically generate initial tags for each image

### 3.3 Manual Label Optimization (via "Tag Editor" module)

1. Switch to the "Tag Editor" module and load `<your_training_data_directory>`
2. Optimize labels in batch or individually: add professional terms, add cultural attributes, and remove irrelevant tags
3. After optimization, click "Save All Changes" to overwrite the original `.txt` files

## 4. Step 3: Train LoRA Model via SD-Trainer GUI

Configure key training parameters in the SD-Trainer interface without manually writing complex commands:

### 4.1 Enter Training Configuration Interface

Click the "Train" module on the SD-Trainer main interface and select "LoRA Training" mode.

### 4.2 Configure Core Training Parameters (Key Items Only)

| Parameter Category | Key Parameter | Configuration Suggestion |
|-------------------|---------------|-------------------------|
| **Basic Settings** | Base Model Path | `<your_base_model_directory>` |
|  | Training Data Directory | `<your_training_data_directory>` |
|  | Output Directory | `<your_model_output_directory>` |
|  | Model Name | Architectural_Style_LoRA (custom) |
| **LoRA Network Settings** | Network Dimension (network_dim) | 32 (balance effect and VRAM) |
|  | Network Alpha (network_alpha) | 32 (consistent with dimension, avoid training instability) |
| **Training Strategy** | Learning Rate (learning_rate) | 1e-4 (common for SD 1.5 models) |
|  | Max Epochs (epoch) | 10-20 (adjust based on data volume, avoid overfitting) |
|  | Batch Size | 4 (adjust based on GPU VRAM) |
|  | Optimizer | AdamW8bit (memory efficient, suitable for low VRAM devices) |
|  | Learning Rate Scheduler | cosine_with_restarts (stable convergence) |

### 4.3 Start Training

Click "Start Training" to automatically execute the training process. View loss curves and sample generation effects through the "Tensorboard" module (access `http://localhost:6006`).

### Training Completion

After training completes, the LoRA model (`.safetensors` format) will be saved to `<your_model_output_directory>`, with filename matching the "Model Name".

## 5. Step 4: Architectural Style Transfer via ComfyUI

Use the trained LoRA model to complete style transfer through visual nodes in ComfyUI:

### 5.1 Prepare ComfyUI Environment

```bash
# Clone ComfyUI repository
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install dependencies
pip install -r requirements.txt

# Place models (critical step)
# 1. Base model → ComfyUI/models/checkpoints/
# 2. Trained LoRA model → ComfyUI/models/loras/
```

### 5.2 Launch ComfyUI and Load Workflow

```bash
# Windows launch (NVIDIA GPU)
./run_nvidia_gpu.bat

# Linux/Mac launch: python main.py
```

Access `http://127.0.0.1:8188` in your browser, click "Load" to load the `Style_Transfer.json` workflow file.

### 5.3 Configure Core Nodes

1. **Load Models**: In the "CheckpointLoader" node select the base model, in the "LoraLoader" node select the trained architectural style LoRA (set weight 0.6-0.8 to control style strength)
2. **Input Image**: In the "Load Image" node upload the building image for style transfer (e.g., modern building image)

### 5.4 Execute Style Transfer

Click "Queue Prompt" in the upper right corner of the interface, ComfyUI automatically executes the workflow, and the generated results are saved to the `ComfyUI/output` directory. Right-click to save the image.

## 6. License

1. **GXETAM-Dataset**: Follows CC BY-NC-SA 4.0 license, non-commercial use requires attribution, commercial use requires contacting the dataset authors
2. **SD-Trainer and ComfyUI**: Both are open-source projects, following MIT license and GPL-3.0 license 
