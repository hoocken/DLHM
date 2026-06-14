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
HUMAN3D="$HOME/miniconda/envs/human3d_cuda113/bin/python"
echo "---------------------------------------------------"
echo "|                  SEGMENTATION                    |"
echo "---------------------------------------------------"
$HUMAN3D lib/human3d/infer_mhbps.py segfit.data_path=$1 general.checkpoint=$2

# Remove saved folder
rm -rf saved

echo "---------------------------------------------------"
echo "|                    FITTING                      |"
echo "---------------------------------------------------"

# Run fitting
uv run fit.py
