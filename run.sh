#!/bin/bash

# Function to print usage guidelines
print_help() {
    echo "Usage: $0 <INPUT> <HUMAN3D_CHECKPOINT> <MALE | FEMALE>"
    echo ""
    echo "Options:"
    echo "  -h, --help    Show this help message and exit"
    echo ""
    echo "Example:"
    echo "  $0 data/input/tr_scan_066.ply MALE"
}

# Check if the user asked for help explicitly
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    print_help
    exit 0
fi

if [ "$#" -lt 3 ]; then
    echo "Error: Three arguments are required." >&2
    print_help
    exit 1
fi

# Set model and experiment to male or female
if [[ "$3" == "MALE" ]]; then
    SMPL_MODEL_PATH="data/models/SMPL_MALE.pkl"
    EXPERIMENT_NAME="hit_male"
elif [[ "$3" == "FEMALE" ]]; then
    SMPL_MODEL_PATH="data/models/SMPL_FEMALE.pkl"
    EXPERIMENT_NAME="hit_female"
else
    echo "Error: Only 'MALE' or 'FEMALE' is allowed." >&2
    print_help
    exit 1
fi

# Run Human3D segmentation
CONDA_PROFILE_PATH="$HOME/miniconda/etc/profile.d/conda.sh"
HUMAN3D_ENV_NAME="human3d_cuda113"
HIT_ENV_NAME="hit"

# Initialize Conda for non-interactive shells
if [ -f "$CONDA_PROFILE_PATH" ]; then
    source "$CONDA_PROFILE_PATH"
else
    echo "Error: Conda profile script not found at $CONDA_PROFILE_PATH"
    exit 1
fi

# Activate the environment
conda activate "$HUMAN3D_ENV_NAME"

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
uv run fit.py model.base_model=$SMPL_MODEL_PATH

echo "---------------------------------------------------"
echo "|                      HIT                        |"
echo "---------------------------------------------------"

# Run HIT
conda activate "$HIT_ENV_NAME"

python lib/HIT/demos/infer_smpl.py --exp_name=$EXPERIMENT_NAME --to_infer smpl_file --target_body outputs/fit/smpl_fit_params.pkl

conda deactivate
