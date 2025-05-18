#!/bin/bash

# Find where the latest MSVC is installed - SKIPPED (Not relevant on Linux)

# Activate python virtual environment
cd "$(dirname "$0")" || exit

# if [ -f "./venv/bin/activate" ]; then
#     echo "Activating Linux venv"
#     source ./venv/bin/activate
# elif [ -f "./.venv/bin/activate" ]; then
#     echo "Activating Linux .venv"
#     source ./.venv/bin/activate
# else
#     echo "ERROR: No virtual environment found!"
#     exit 1
# fi

echo "INFO: Assuming you are already in the correct Python environment (e.g., Conda zonos)."

# Set environment variables
export HF_HOME="$(pwd)/huggingface"
export TORCH_HOME="$(pwd)/torch"
# export HF_ENDPOINT="https://hf-mirror.com"
export XFORMERS_FORCE_DISABLE_TRITON=1
export PHONEMIZER_ESPEAK_LIBRARY="$HOME/.local/lib/libespeak-ng.so"
export GRADIO_HOST="http://localhost:7860/"

# (Optional) export CUDA_HOME if not automatically set
if [ -z "$CUDA_HOME" ] && [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
fi

# Run Gradio app
python -m gradio_interface
