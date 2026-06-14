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

conda env create -f environment.yaml

conda activate human3d_cuda113

pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html --no-build-isolation
pip3 install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps --no-build-isolation

cd third_party

# Don't forget to install CUDA Toolkit for 11.3 before installing MinkowskiEngine!
git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --force_cuda --blas=openblas --no-build-isolation

cd ../../pointnet2
python setup.py install

cd ../../
pip3 install pytorch-lightning==1.7.2 pytorch==1.12
pip3 install numpy==1.26.0
```

You may need to fix some dependency issues that arises. If you have problems regarding MinkowskiEngine compilation, downgrade gcc and g++ to version 9 and export the compiler flags as an environment variable.

### Fitting
Afterwards, run
```sh
uv sync
```
to install all required libraries.

### SMPL Model
You can download the SMPL model from https://smpl.is.tue.mpg.de/.

## Usage
To run the fitting, put your input in `data/input/` and the SMPL models in `data/models/`. Afterwards, set the path
to the SMPL model in `config/config.yaml`. Then run
```sh
bash run.sh data/input/<input_file>.ply data/ckpts/human3d.ckpt
```

If you just want to run the fitting on a segmentation result, run
```sh
uv run fit.py
```

Your fitted model will be available in `outputs/<date>/<time>/`. The segmentation results is viewable in `outputs/human3d_segs`.

## Libraries
### SMPL
Base SMPL code for PyTorch in Python 3.11 is taken from https://github.com/Pokerlishao/SMPL-py311 with modifications. 

### SMPL Prior
SMPL Prior is taken from https://github.com/DavidBoja/SMPL-Fitting.

### Human3D
[Human3D](https://github.com/human-3d/Human3D) is taken from [SegFit](https://github.com/segfit/segfit), with minor modifications regarding the rotation of the point clouds to ensure correct segmentation. The checkpoints are taken from SegFit.