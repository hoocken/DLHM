#!/bin/bash

# Function to print usage guidelines
print_help() {
    echo "Usage: $0 <INPUT> <male | female> [-m <MOTION>]"
    echo ""
    echo "Options:"
    echo "  -h, --help    Show this help message and exit"
    echo "  -m <MOTION>   Input motion to animate segmentations. If -n is set, then this step is skipped."
    echo "  -n            Skip HIT segmentation"    
    echo "  -s <MOTION>   Soft body animation"    
    echo ""
    echo "Example:"
    echo "  $0 data/input/tr_scan_066.ply female -m data/motion/Jog_3_poses.npz"
}

# Check if the user asked for help explicitly
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    print_help
    exit 0
fi

MOTION=""
SOFT_MOTION=""
HIT=true
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m)
            if [[ $# -lt 2 ]]; then
                echo "Error: -m requires a value." >&2
                print_help
                exit 1
            fi
            MOTION="$2"
            shift 2
            ;;
        -s)
            if [[ $# -lt 2 ]]; then
                echo "Error: -s requires a value." >&2
                print_help
                exit 1
            fi
            SOFT_MOTION="$2"
            shift 2
            ;;
        -n)
            HIT=false
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        --)
            shift
            POSITIONAL+=("$@")
            break
            ;;
        -*)
            echo "Invalid option: $1" >&2
            print_help
            exit 1
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [[ ${#POSITIONAL[@]} -lt 2 ]]; then
    echo "Error: Two arguments are required." >&2
    print_help
    exit 1
fi

INPUT="${POSITIONAL[0]}"
GENDER="${POSITIONAL[1]}"

if [[ "$GENDER" == "male" ]]; then
    EXPERIMENT_NAME="hit_male"
elif [[ "$GENDER" == "female" ]]; then
    EXPERIMENT_NAME="hit_female"
else
    echo "Error: Only 'male' or 'female' is allowed." >&2
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

# Create outputs folder
mkdir -p outputs

# Activate the environment
conda activate "$HUMAN3D_ENV_NAME"

echo "---------------------------------------------------"
echo "|                  SEGMENTATION                   |"
echo "---------------------------------------------------"
python lib/human3d/infer_mhbps.py segfit.data_path="$INPUT" general.checkpoint=data/ckpts/human3d.ckpt

conda deactivate

# Remove saved folder
rm -rf saved

echo "---------------------------------------------------"
echo "|                    FITTING                      |"
echo "---------------------------------------------------"

# Run fitting
conda activate "$HIT_ENV_NAME"
python src/fit/fit.py model.gender="$GENDER"

if [[ "$HIT" = false ]] ; then
    exit 0
fi

echo "---------------------------------------------------"
echo "|                      HIT                        |"
echo "---------------------------------------------------"

# Run HIT
python lib/HIT/demos/infer_smpl.py --exp_name="$EXPERIMENT_NAME" --to_infer smpl_file --target_body outputs/fit/smpl_fit_params.pkl

if [[ "$MOTION" != "" ]]; then
    echo "---------------------------------------------------"
    echo "|                     MOTION                      |"
    echo "---------------------------------------------------"

    # Pose HIT
    python src/animate/animate_hit.py animate.data="$MOTION"
fi

if [[ "$SOFT_MOTION" != "" ]]; then
    # Tetrahedralize obj mesh from HIT
    echo "---------------------------------------------------"
    echo "|                 TETRAHEDRALIZE                  |"
    echo "---------------------------------------------------"

    uv run src/soft_body/create_tetrahedral_mesh.py

    # Soft tissue animation
    echo "---------------------------------------------------"
    echo "|             SOFT TISSUE ANIMATION               |"
    echo "---------------------------------------------------"

    python src/soft_body/animate_smpl.py soft_body.data="$SOFT_MOTION" soft_body.exp_name="$EXPERIMENT_NAME"
fi

conda deactivate