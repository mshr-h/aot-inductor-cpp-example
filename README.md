# aot-inductor-cpp-example

Minimal working example of Ahead-Of-Time (AOT) compiled PyTorch model deployment in C++.

## Overview
This project demonstrates how to export a PyTorch model using AOTInductor and run it from C++ using LibTorch. It is intended as a minimal, reproducible example for research, benchmarking, or deployment prototyping.

## Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (for Python environment management)
- CUDA 11.8 (for GPU support)
- wget, unzip

## Setup Instructions

1. **Download and extract LibTorch (PyTorch C++ API):**
   ```bash
   wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu118.zip
   unzip libtorch-cxx11-abi-shared-with-deps-2.7.0+cu118.zip
   ```
   This will create a `libtorch/` directory in your project folder.

2. **Create a Python virtual environment and install dependencies:**
   ```bash
   uv venv
   uv pip install cmake ninja torch torchvision
   ```

3. **Export/compile the PyTorch model:**
   ```bash
   uv run model.py --model resnet18
   ```
   This will generate `model.pt2` (AOT-compiled model).

4. **Configure the C++ build:**
   ```bash
   CMAKE_PREFIX_PATH=libtorch uv run cmake -S . -B build -G Ninja
   ```

5. **Build the C++ example:**
   ```bash
   uv run cmake --build build --config Release
   ```
   The compiled binary will be located at `build/aoti_example`.

6. **Run the example:**
   ```bash
   ./build/aoti_example
   ```
   This will load the compiled model and run inference.

## Usage
The example loads the AOT-compiled model and runs inference on a random input data.

- To use a different model, edit `model.py` and re-run step 3.
- To change the C++ inference logic or input, edit `inference.cpp` and rebuild.
