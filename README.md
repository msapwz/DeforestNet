# DeforestNet: Change Detection Model for Deforestation Dataset

DeforestNet is a change detection model specifically designed to detect deforestation using satellite imagery. The model supports various encoder and fusion method variations, providing flexibility for different types of change detection tasks. Built with PyTorch, this model is optimized for detecting changes between two time-series satellite images.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Model Variations](#model-variations)
- [Results](#results)



## Installation

1. Clone this repository:

   ```bash
   git clone https://gitlab.nrp-nautilus.io/msapwz/DeforestNet.git
   ```
Install the required dependencies:

    pip install pytorch pytorchvideo
    pip install timm
    pip install mmcv
    pip install matplotlib

## Usage
#### Training
To train the model, run the main.py script:

    python main.py --mode train --root_dir_dataset PATH_TO_DATASET --output_folder OUTPUT_FOLDER

Replace PATH_TO_DATASET and OUTPUT_FOLDER with the appropriate paths to your dataset and output folder. You can customize the training further by using the additional arguments listed below.

#### Evaluation
To evaluate a trained model, use the eval.py script:

    python eval.py --model_checkpoint PATH_TO_CHECKPOINT --root_dir_dataset PATH_TO_DATASET --output_folder OUTPUT_FOLDER

Ensure that you provide the correct path to the trained model checkpoint. Also, the checkpoints must fit the same input parameters from the training phase.

#### Model Variations
DeforestNet supports multiple model variations, allowing you to experiment with different architectures and fusion methods:

- Encoders: SegFormer, SegNext
- Fusion Methods: sum, average, concat, attention
- Attention Types: att, cta (Cross-Temporal Attention)
- Siamese Types: single, dual
- Feature Exchange Variants: Various combinations like SpatialExchange (se), ChannelExchange (ce), and learnable options.



