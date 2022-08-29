ARG BASE_TAG=21.12-py3

FROM  nvcr.io/nvidia/pytorch:$BASE_TAG
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-12.html#rel_21.12
# Python 3.8.12
# NVIDIA CUDA 11.5.0
# PyTorch 1.11.0a0+b6df043
ARG TORCH_VERSION
USER root
RUN mkdir /app
WORKDIR /app

# Remove old torch
RUN pip uninstall torch -y

# Manually install torch
RUN conda install astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses -y
RUN conda install mkl mkl-include -y
RUN conda install -c pytorch magma-cuda115 -y

RUN git clone https://github.com/pytorch/pytorch /pytorch
WORKDIR /pytorch
RUN git fetch --all --tags
RUN git checkout tags/v${TORCH_VERSION}
RUN git submodule update --init --recursive
RUN git submodule sync
RUN git submodule update --init --recursive --jobs 0

# Hard-coded version. See https://github.com/pytorch/pytorch/issues/20525
# export USE_CUDA=1 USE_CUDNN=1 USE_MKLDNN=1
ENV PYTORCH_BUILD_VERSION=$TORCH_VERSION
ENV TORCH_DEFAULT_VERSION=$TORCH_VERSION
#  export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
ENV CMAKE_PREFIX_PATH=/opt/conda/  

RUN python setup.py install

# back to normal workdir
WORKDIR /app

# print versions
RUN nvcc --version
RUN python --version
RUN python -c "import torch; print(torch.__version__); print(torch.version.cuda)"

CMD ["jupyter", "notebook"]
