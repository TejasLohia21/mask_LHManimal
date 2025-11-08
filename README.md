# Masker: High-Quality Animal Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

> A robust, pluggable engine for generating high-quality binary masks for animals in images. It uses a smart, cascaded approach with SAM (Segment Anything Model) and Detectron2 for state-of-the-art results.

This tool is designed to be easily integrated into larger computer vision pipelines, such as 3D reconstruction or pose estimation projects.

---

### Key Features

*   **High-Quality Masks:** Leverages the zero-shot power of SAM, guided by bounding boxes for precision.
*   **Robust Fallback:** Uses a COCO-pretrained Detectron2 Mask R-CNN model as a fallback, ensuring a mask is almost always generated.
*   **Pluggable Backend:** The architecture is designed to easily swap or add new segmentation backends.
*   **Simple API:** A clean `MaskerPredictor` class makes integration into other projects trivial.
*   **Command-Line Interface:** Includes a powerful CLI for batch processing entire directories of images.
*   **Reproducible Environment:** Comes with a detailed `environment.yml` to perfectly replicate the complex runtime.

### Demo

*(Suggestion: Create a side-by-side image showing an input image and the final `_mask.png` output. Place it in a folder like `assets/` and link it here.)*

| Input Image                                     | Generated Mask                                    |
| ----------------------------------------------- | ------------------------------------------------- |
| ![Input Tiger](demo/images/your_tiger_image.png) | ![Output Mask](demo/masks/your_tiger_mask.png) |

---

### Installation

Follow these steps to set up the project and its dependencies.

#### 1. Prerequisites

*   **Conda:** You must have `conda` installed to manage the environment.
*   **NVIDIA GPU & Drivers:** A CUDA-enabled GPU is required, along with appropriate NVIDIA drivers. This project is built against CUDA 11.8.

#### 2. Clone the Repository

```
git clone https://github.com/your-username/Masker.git
cd Masker
```

#### 3. Create and Activate the Conda Environment

This command uses the provided `environment.yml` file to create a new Conda environment named `masker-env` with all dependencies pinned to the correct, working versions.

```
conda env create -f environment.yml
conda activate masker-env
```

#### 4. Install the `masker` Package

Install the project in editable mode. This makes the `masker` package available to your Python environment.

```
pip install -e .
```

#### 5. Download Pretrained Models

The `Detectron2` model is downloaded automatically. You only need to download the SAM checkpoint.

```
# Create a directory for the model
mkdir -p models/sam

# Download the SAM ViT-H checkpoint (recommended for quality)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P models/sam/
```

---

### Usage

You can use `Masker` via its CLI or by importing it into your own Python scripts.

#### Command-Line Interface (CLI)

The CLI is perfect for batch-processing a folder of images.

```
python -m masker.cli \
    --in_dir /path/to/your/images \
    --out_dir /path/to/save/masks \
    --sam_ckpt models/sam/sam_vit_h_4b8939.pth
```

This will generate a `_mask.png` file for each image in the input directory.

#### Python API

Integrating `Masker` into your own project (like LHM for Animals) is straightforward.

```
from masker import MaskerConfig, MaskerPredictor
import cv2

# 1. Configure the predictor
cfg = MaskerConfig(
    sam_ckpt="models/sam/sam_vit_h_4b8939.pth",
    sam_model_type="vit_h",
    use_box_prompt=True,
    det_conf=0.5
)

# 2. Initialize the predictor (models are loaded once)
predictor = MaskerPredictor(cfg)

# 3. Load an image and get the mask
image_path = "path/to/your/animal.jpg"
image_bgr = cv2.imread(image_path)

result = predictor.predict(image_bgr)

# 4. Use the result
if result["mask"] is not None:
    binary_mask = result["mask"] # This is a (H, W) numpy array with {0, 1}
    score = result["score"]
    box = result["box"]
    
    # Save the mask as a viewable image
    predictor.save(binary_mask, "output_mask.png")
    
    print(f"Mask generated with score: {score:.3f}")
else:
    print("No mask could be generated for the image.")

```

---

### License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

### Acknowledgements

This project stands on the shoulders of giants. Our sincere thanks to the teams behind:

*   **[Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)**
*   **[Detectron2](https://github.com/facebookresearch/detectron2)**

```
