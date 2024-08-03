# neural-style-transfer-pytorch

# Neural Style Transfer in PyTorch using Forward Hooks

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction
This repository contains an implementation of Neural Style Transfer using PyTorch. The project leverages forward hooks to capture activations from specific layers of a pre-trained convolutional neural network (VGG16). By optimizing the content and style representations, this implementation can apply the artistic style of one image to the content of another.

## Features
- Capture intermediate activations using forward hooks.
- Compute Gram matrices for style representation.
- Minimize content and style losses to generate stylized images.
- Configurable parameters for customization.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/bumSin/neural-style-transfer-pytorch.git
    cd neural-style-transfer-pytorch
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Prepare your content and style images.
2. Run the neural style transfer script:
    ```bash
    python neural_style_transfer.py --content_img_name content.png --style_img_name style.png --epochs 300 --alpha 2e5 --beta 2e10
    ```

### Command Line Arguments
- `--content_img_name`: File name of the content image.
- `--style_img_name`: File name of the style image.
- `--init_strategy`: Strategy to initiate base image.
- `--epochs`: Number of iterations for optimization (default: 100).
- `--alpha`: Weight for content loss (default: 1e5).
- `--beta`: Weight for style loss (default: 1e10).

## Project Structure
```plaintext
neural-style-transfer-pytorch/
│
├── my_project/
│   ├── neural_style_transfer.py
│   ├── services/
│   │   ├── loss_service.py
│   ├── data/
│   │   ├── content_image
│   │   ├── style_image
│   ├── utils/
│   │   ├── ImageUtils.py
│   │   ├── ModelUtils.py
└── environment.yml
