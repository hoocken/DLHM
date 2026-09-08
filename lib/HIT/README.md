# HIT
Taken from https://github.com/MarilynKeller/HIT. This README is modified from its original state.

## Installation
Setup a virtual env
```
conda create -n 'hit' python=3.8
```

#### Install packages
Check your CUDA toolkit version
```shell
nvcc --version
```
Install torch depending on CUDA toolkit version. See: https://pytorch.org/get-started/previous-versions/ . 

For 11.8:
```shell
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```


Install relevant packages
```shell
pip install -r requirements.txt
pip install  git+https://github.com/MPI-IS/mesh.git
pip install  git+https://github.com/mattloper/chumpy
pip install -e .
```

The **LEAP** package is used for its marching cube implementation and creating ground truth occupancy. Install it with:
```shell
cd hit
mkdir external
cd external 
git clone https://github.com/neuralbodies/leap.git
cd leap
python setup.py build_ext --inplace
pip install -e .
```

## Pretrained Models
To download pretrained HIT networks, go to the download tab at https://hit.is.tue.mpg.de/ and create a `pretrained` folder in this local directory, so `lib/HIT/pretrained`. Then you should have it in the following hierarchy:

```
${HIT}
├── pretrained
│   ├── hit_female
│       ├── ckpt
│       └── config.yaml
│   └── hit_male
│       └── ...
├── hit
└── ...
```
