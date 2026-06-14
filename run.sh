#!/bin/bash

# Function to print usage guidelines
print_help() {
    echo "Usage: $0 <INPUT> <HUMAN3D_CHECKPOINT>"
    echo ""
    echo "Options:"
    echo "  -h, --help    Show this help message and exit"
    echo ""
    echo "Example:"
    echo "  $0 data/input/tr_scan_066.ply data/ckpts/human3d.ckp"
}

# Check if the user asked for help explicitly
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    print_help
    exit 0
fi

if [ "$#" -lt 2 ]; then
    echo "Error: Two arguments are required." >&2
    print_help
    exit 1
fi

# Run Human3D segmentation
CONDA_PROFILE_PATH="$HOME/miniconda/etc/profile.d/conda.sh"
ENV_NAME="human3d_cuda113"

# Initialize Conda for non-interactive shells
if [ -f "$CONDA_PROFILE_PATH" ]; then
    source "$CONDA_PROFILE_PATH"
else
    echo "Error: Conda profile script not found at $CONDA_PROFILE_PATH"
    exit 1
fi

echo "$ENV_NAME"
# Activate the environment
conda activate "$ENV_NAME"

echo "---------------------------------------------------"
echo "|                  SEGMENTATION                   |"
echo "---------------------------------------------------"
python lib/human3d/infer_mhbps.py segfit.data_path=$1 general.checkpoint=$2

conda deactivate

# Remove saved folder
rm -rf saved

echo "---------------------------------------------------"
echo "|                    FITTING                      |"
echo "---------------------------------------------------"

# Run fitting
uv run fit.py
