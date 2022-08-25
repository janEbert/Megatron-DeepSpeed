# Docker image for running https://github.com/OpenGPTX/bigscience_megatron_deepspeed

Juwels setup: https://github.com/OpenGPTX/BigScience-Setup
- torch 1.11 (or does this come from deepspeedmegatron reqs?)
- cuda 11.5

DFKI setup:
- cuda 11.6 

Bigscience setup:
- torch>=1.7
- cudatoolkit=11.3


## Build

```bash
# clone repo and change dir
git clone https://github.com/OpenGPTX/bigscience_megatron_deepspeed.git
cd bigscience_megatron_deepspeed

# build
docker build -t malteos/obmd:latest -f docker/Dockerfile .

# run bash
docker run -it malteos/obmd bash

# run with gpu with `nvidia-docker` executor; specific GPUs with NV_GPU=1,2
NV_GPU=1,2  nvidia-docker run -v $PWD:/app -it malteos/obmd bash

# run original nvidia image
NV_GPU=1,2  nvidia-docker run -it nvcr.io/nvidia/pytorch:21.12-py3 bash

	

# or with `--gpus all` flag
docker run --gpus all -it malteos/obmd bash

# push
docker push malteos/obmd

# run tests
docker run -malteos/obmd pytest ./tests/test_training_debug.py

```

## TODO

```bash
python -m pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

###
git clone https://github.com/NVIDIA/apex
cd apex
git checkout cd0a1f11061068db45f12ef829ca3250389cd7ae
pip install --global-option="--cpp_ext" --global-option="--cuda_ext" --no-cache -v --disable-pip-version-check .  2>&1 | tee build.log
####

nstall torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

python -m pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

RUN TORCH_CUDA_ARCH_LIST="8.0" DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=0 DS_BUILD_UTILS=1 pip install git+https://github.com/microsoft/deepspeed.git@3da841853ca07abf3a09e7bd325a576c4e642c11 --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check
#

RUN git clone https://github.com/NVIDIA/apex /apex
RUN cd /apex && pip install --global-option="--cpp_ext" --global-option="--cuda_ext" --no-cache -v --disable-pip-version-check .


# Install git-lfs for huggingface hub cli
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs
RUN git lfs install

# Upgrade tensorboard
RUN pip uninstall tensorboard-plugin-wit tensorboard-data-server tensorboard jupyter-tensorboard nvidia-tensorboard nvidia-tensorboard-plugin-dlprof -y
RUN pip install tensorboard>=2.9.0
``