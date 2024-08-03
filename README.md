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
This repository contains an implementation of Neural Style Transfer using PyTorch. The project leverages forward hooks to capture activations from specific layers of a pre-trained convolutional neural network (VGG19). By optimizing the content and style representations, this implementation can apply the artistic style of one image to the content of another.

## Features
- Capture intermediate activations using forward hooks.
- Compute Gram matrices for style representation.
- Minimize content and style losses to generate stylized images.
- Configurable parameters for customization.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/neural-style-transfer-pytorch.git
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
    python main.py --content-image path/to/content.jpg --style-image path/to/style.jpg --output-image path/to/output.jpg --epochs 300
    ```

### Command Line Arguments
- `--content-image`: Path to the content image.
- `--style-image`: Path to the style image.
- `--output-image`: Path to save the output image.
- `--epochs`: Number of iterations for optimization (default: 300).
- `--content-weight`: Weight for content loss (default: 1e5).
- `--style-weight`: Weight for style loss (default: 1e10).
- `--learning-rate`: Learning rate for the optimizer (default: 0.003).

## Project Structure
```plaintext
neural-style-transfer-pytorch/
│
├── my_project/
│   ├── __init__.py
│   ├── main.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── data_service.py
│   │   ├── logic_service.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── data_model.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── helper.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py
│
├── tests/
│   ├── __init__.py
│   ├── test_services/
│   │   ├── __init__.py
│   │   ├── test_data_service.py
│   │   ├── test_logic_service.py
│   ├── test_models/
│   │   ├── __init__.py
│   │   ├── test_data_model.py
│
├── requirements.txt
└── setup.py
