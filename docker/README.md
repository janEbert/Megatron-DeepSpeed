# Docker image for running https://github.com/OpenGPTX/bigscience_megatron_deepspeed

The image reproduces the JUWELS setup (see https://github.com/OpenGPTX/BigScience-Setup) but uses Python 3.8 instead of Python 3.9.

Base image: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-12.html#rel_21.12

Software stack (with `BASE_TAG=21.12-py3`):
- Python 3.8.12
- NVIDIA CUDA 11.5.0
- PyTorch 1.11.0a0+b6df043 

## Variants

Please set the `BASE_TAG` environment variable to select the [NVIDIA base image tag](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html) (this defines the cuda + torch version):

```bash
export BASE_TAG=21.12-py3
# default
# - CUDA 11.5.0
# - PyTorch 1.11.0a0+b6df043 

export BASE_TAG=22.08-py3
# latest
# - CUDA 11.7.1
# - PyTorch 1.13.0a0+d321be6
```


## Build & push

```bash
# clone repo and change dir
git clone https://github.com/OpenGPTX/bigscience_megatron_deepspeed.git
cd bigscience_megatron_deepspeed

# activate docker buildkit for multi-stage build
export DOCKER_BUILDKIT=1

# build
docker build -t malteos/obmd:$BASE_TAG --target main --build-arg BASE_TAG=$BASE_TAG -f docker/Dockerfile .

# push
docker push malteos/obmd:$BASE_TAG

# tag and push with git hash
COMMIT_HASH=$(git rev-parse --short HEAD)
COMMIT_TAG=${BASE_TAG}-${COMMIT_HASH}

docker tag malteos/obmd:$BASE_TAG malteos/obmd:$COMMIT_TAG
docker push malteos/obmd:$COMMIT_TAG
```


## Run & test

```bash
# run bash
docker run -it malteos/obmd:$BASE_TAG bash

# run with gpu with `nvidia-docker` executor; specific GPUs with NV_GPU=1,2
NV_GPU=1,2  nvidia-docker run -v $PWD:/app -it malteos/obmd:$BASE_TAG bash

# run original nvidia image (for testing)
NV_GPU=1,2  nvidia-docker run -it nvcr.io/nvidia/pytorch:$BASE_TAG bash

# or with `--gpus all` flag
docker run --gpus all -it malteos/obmd:$BASE_TAG bash

# run tests
docker run -it malteos/obmd:$BASE_TAG pytest ./tests/test_training_debug.py

NV_GPU=1,2  nvidia-docker run -it malteos/obmd:${BASE_TAG}-torch_1-12-1  pytest ./tests/test_training_debug.py
```


## Build image for Github actions runner

This image include extra files for Github's CICD runner and a different starting command.

```bash
# activate docker buildkit for multi-stage build
export DOCKER_BUILDKIT=1

# build
docker build -t malteos/obmd:${BASE_TAG}-runner --target actions_runner --build-arg BASE_TAG=$BASE_TAG -f docker/Dockerfile .
```


## Build PyTorch 1.12.1 with CUDA 11.5 from source

See https://github.com/pytorch/pytorch#from-source

```bash
pip uninstall -y torch

conda install astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses -y
conda install mkl mkl-include -y
conda install -c pytorch magma-cuda115 -y

git clone https://github.com/pytorch/pytorch
cd pytorch
git fetch --all --tags
git checkout tags/v1.12.1
git submodule update --init --recursive

git submodule sync
git submodule update --init --recursive --jobs 0

export PYTORCH_BUILD_VERSION=1.12.1
export TORCH_DEFAULT_VERSION=1.12.1
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```

#### Build image with custom PyTorch version

This is needed when our wanted PyTorch version is not provided by NVIDIA base images.

```bash
export BASE_TAG=21.12-py3
export BASE_IMAGE=malteos/nvidia-pytorch

# activate docker buildkit for multi-stage build
export DOCKER_BUILDKIT=1

# base image
docker build -t $BASE_IMAGE:${BASE_TAG}-torch_1-12-1 --build-arg BASE_TAG=$BASE_TAG --build-arg TORCH_VERSION=1.12.1 -f docker/custom_pytorch.Dockerfile .
docker run -it $BASE_IMAGE:${BASE_TAG}-torch_1-12-1 bash
docker push $BASE_IMAGE:${BASE_TAG}-torch_1-12-1


# obmd image with custom base image
docker build -t malteos/obmd:${BASE_TAG}-torch_1-12-1 --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg BASE_TAG=${BASE_TAG}-torch_1-12-1 -f docker/Dockerfile .
docker push malteos/obmd:${BASE_TAG}-torch_1-12-1 

```

#### Check PyTorch version 

See https://github.com/pytorch/pytorch/releases/tag/v1.12.0

```python
import torch
a = torch.tensor([1., 2., 3., 4.], dtype=torch.float32)
b = torch.tensor([2., 2., 2., 2.], dtype=torch.float64)
c = torch.tensor([3., 3., 3., 3.], dtype=torch.float64)

# 1.12 = torch.float64 (1.11 = torch.float32)
assert torch.clamp(a, b, c).dtype == torch.float64
```



## Build with Slurm compute job (DFKI infrastructure)

Start interactive compute job with Podman installed (`/dev/fuse` needs to be mounted):

```bash
srun  \
    --container-image=/netscratch/enroot/podman+enroot.sqsh \
    --container-mounts=/dev/fuse:/dev/fuse,/netscratch/$USER:/netscratch/$USER,/netscratch/$USER/podman_storage:/var/lib/containers/storage,"`pwd`":"`pwd`"     --container-workdir="`pwd`" \
    --time 0-12:00:00 --pty bash
```

Build with Podman for different base tags (`--isolation=chroot` is needed):
```bash
export BASE_TAG=21.12-py3
export BASE_TAGS="21.12-py3 22.08-py3"
export ENROOT_SQUASH_OPTIONS="-comp lz4 -Xhc -b 262144"

for BASE_TAG in $BASE_TAGS
do
    echo "Building ${BASE_TAG} ..."

    podman build -t malteos/obmd:$BASE_TAG --isolation=chroot --build-arg BASE_TAG=$BASE_TAG  -f docker/Dockerfile .
    enroot import -o /netscratch/$USER/obmd+$BASE_TAG .sqsh podman://malteos/obmd:$BASE_TAG 

    echo "Completed ${BASE_TAG}"
done

```

Export as enroot images:
```bash
export ENROOT_SQUASH_OPTIONS="-comp lz4 -Xhc -b 262144"
enroot import -o /netscratch/$USER/obmd.sqsh podman://temp
```

Push to Docker hub:
```bash
podman build . --isolation=chroot -t temp
```


### Run the image

Start GPU job with image

```bash
$ srun \
    --container-image=/netscratch/$USER/obmd.sqsh \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`" \
    --container-workdir="`pwd`" \
    --ntasks=1 --nodes=1 -p A100 --gpus=1 --mem=100G \
    --pty bash
```

