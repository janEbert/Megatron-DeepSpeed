# Docker image for running https://github.com/OpenGPTX/bigscience_megatron_deepspeed

The image reproduces the JUWELS setup (see https://github.com/OpenGPTX/BigScience-Setup) but uses Python 3.8 instead of Python 3.9.

Base image: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-12.html#rel_21.12

Software stack:
- Python 3.8.12
- NVIDIA CUDA 11.5.0
- PyTorch 1.11.0a0+b6df043 

## Build & push

```bash
# clone repo and change dir
git clone https://github.com/OpenGPTX/bigscience_megatron_deepspeed.git
cd bigscience_megatron_deepspeed

# build
docker build -t malteos/obmd:latest -f docker/Dockerfile .

# push
docker push malteos/obmd
```


## Run & test

```bash
# run bash
docker run -it malteos/obmd bash

# run with gpu with `nvidia-docker` executor; specific GPUs with NV_GPU=1,2
NV_GPU=1,2  nvidia-docker run -v $PWD:/app -it malteos/obmd bash

# run original nvidia image
NV_GPU=1,2  nvidia-docker run -it nvcr.io/nvidia/pytorch:21.12-py3 bash

# or with `--gpus all` flag
docker run --gpus all -it malteos/obmd bash

# run tests
docker run -malteos/obmd pytest ./tests/test_training_debug.py
```
