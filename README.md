# 3D Point Cloud to Animation
## Installation

### Human3D
To use this, you must first set up the Human3D conda environment in another folder:
```sh
git clone https://github.com/human-3d/Human3D.git
cd Human3D

# Some users experienced issues on Ubuntu with an AMD CPU
# Install libopenblas-dev (issue #115, thanks WindWing)
# sudo apt-get install libopenblas-dev

export TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"

conda env create -f environment.yaml # here you may need to update some package versions

conda activate human3d_cuda113

pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html --no-build-isolation
pip3 install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps --no-build-isolation

cd third_party

# Don't forget to install CUDA Toolkit for 11.3 before installing MinkowskiEngine!
git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228

# Uncomment this line if you want to set max jobs:
# export MAX_JOBS=3
python setup.py install --force_cuda --blas=openblas

cd ../pointnet2
python setup.py install

cd ../../
pip3 install pytorch-lightning==1.7.2 --no-deps
pip3 install numpy==1.26.0
```

You may need to fix some dependency issues that arises. If you have problems regarding MinkowskiEngine compilation, downgrade gcc and g++ to version 9 and export the compiler version as an environment variable.

### HIT
For the SMPL code and HIT network, we use the repository from [HIT](https://github.com/MarilynKeller/HIT). 

To install it, go to the directory:
```sh
cd lib/HIT
```

and then do follow the steps in `lib/HIT/README.md` to fully install the library.

### Tetrahedralize
For the soft body animation, we use PyTetGen which is installed to another virtual environment, which is managed by [uv](https://docs.astral.sh/uv/). See their website for the full installation tutorial.

After installing uv, run
```sh
uv sync
```
to install all required libraries.

### SMPL Model
You can download the SMPL model from https://smpl.is.tue.mpg.de/. Create a `models` folder in `data` and put the SMPL models in `data/models/smpl`, resulting in the following hierarchy:
```
.
├── data
│   ├── models
│       ├── SMPL_MALE.pkl
│       └── SMPL_FEMALE.pkl
│   └── ...
```

## Usage
To run the fitting, you can put your ipnuts in the `data` folder. Then run
```sh
bash run.sh --help
```
for further instructions on running the script.

You can configure the parameters in `config/config.yaml` for different stages of the pipeline.

Your results will be available in the `outputs` folder with the following information:
```
outputs
├── fit # Results from fitting 
├── hit_best # Results from HIT, which contains tissue meshes
├── human3d_segs # Results from human3d segmentation       
├── motion # Results from SMPL animation 
├── motion_seg # Results from rigged tissue animation 
├── motion_soft # Results from soft tissue animation
└── tet_mesh # Results from tetrahedralization of SMPL mesh
```

The results in `motion_seg` and `motion_soft` are OBJ sequences which can be played and rendered as a video with the OBJSequence addon from Blender (https://extensions.blender.org/add-ons/stop-motion-obj2/).

## Animate SMPL
To only animate the SMPL model fitted from the fitting stage, run
```sh
bash run.sh <INPUT> <GENDER> -n
```
to only fit but skip HIT segmentations. Then set your desired motion for `pose.data` in `config/config.yaml` and run
```sh
uv run src/animate/animate_model.py
```

The output will be in `outputs/motion/motion.npz`, where you can use the SMPL Blender addon (https://github.com/Meshcapade/SMPL_blender_addon) and add this file as animation.

### Metrics
#### Fitting
To measure the fitting with FAUST, you need to put all FAUST training data into `data/input` and `data/target`. Then run
```sh
uv run src/fit/evaluate_fit.py
```

#### Soft Tissue Animation
To measure the reconstruction error of soft tissue animation, go to `config/config.yaml` and put the desired path into `motion_gt` for the ground truth scan and `data` for the motion. Then run
```sh
conda activate hit
python src/soft_body/extract_shape_from_motion.py
```
which produces the SMPL parameters from the motion file. Then run the tetrahedralizer:
```sh
uv run src/soft_body/create_tetrahedral_mesh.py
```
and then the animation.
```sh
python src/soft_body/animate_smpl.py
```

Automatically, it should print out the reconstruction error and also graph the error over time.

## Libraries
### SMPL Prior
SMPL Prior is taken from https://github.com/DavidBoja/SMPL-Fitting.

### Human3D
[Human3D](https://github.com/human-3d/Human3D) is taken from [SegFit](https://github.com/segfit/segfit), with minor modifications regarding the rotation of the point clouds to ensure correct segmentation. The checkpoints are taken from SegFit.

### HIT
[HIT](https://github.com/MarilynKeller/HIT) is taken directly from the website with modifications.