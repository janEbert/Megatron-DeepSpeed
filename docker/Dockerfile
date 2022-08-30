ARG BASE_TAG=21.12-py3
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch

FROM $BASE_IMAGE:$BASE_TAG
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-12.html#rel_21.12
# Python 3.8.12
# NVIDIA CUDA 11.5.0
# PyTorch 1.11.0a0+b6df043 (pip install torch==1.11.0a0+b6df043)
USER root

RUN mkdir -p /app
WORKDIR /app

# Install git-lfs for huggingface hub cli
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs
RUN git lfs install

# Install deepspeed
# Before we need to install triton==1.0.0 manually.
RUN python -m pip install triton==1.0.0

# Needed to build deepspeed
RUN apt -y update
RUN apt install -y libaio-dev

# Copy DeepSpeed repo, checkout specfic commit, and install from source:
#
# Parameters:
# - TORCH_CUDA_ARCH_LIST => supported GPU types (A100 = 8.0)
#   - A100 (SM80)
#   - Quadro RTX 6000 (SM75) => 7.5?
# - DS_BUILD_OPS => DeepSpeed ops (DS_BUILD_OPS=1 => DeepSpeed will only install any ops that are compatible with your machine)
# - You can adjust -j to specify how many cpu-cores are to be used during the build. 
RUN git clone https://github.com/microsoft/deepspeed.git /deepspeed
WORKDIR /deepspeed
RUN git checkout 0f5c2012ce19c36936be562094ef3d73b230e364
RUN TORCH_CUDA_ARCH_LIST="7.5;8.0" \
    DS_BUILD_CPU_ADAM=1 \
    DS_BUILD_UTILS=1 \
    pip install -e . \
    --global-option="build_ext" \
    --global-option="-j8" \
    --no-cache -v --disable-pip-version-check
WORKDIR /app


# Install normal requirements

# Copy requirements first to use cache
COPY ./requirements/requirements.txt ./requirements/requirements.txt
COPY ./requirements/requirements_dev.txt ./requirements/requirements_dev.txt

# A hacky but working way to exclude deepspeed, apex and torch from the requirements (we install them manually or they are part of base image)
# See https://github.com/OpenGPTX/BigScience-Setup/blob/main/run_scripts/set_up.bash#L69-L71
RUN sed -i 's/^[Dd]eep[Ss]peed/# &/' ./requirements/requirements.txt
RUN sed -i 's/^torch/# &/' ./requirements/requirements.txt
RUN sed -i 's/^apex/# &/' ./requirements/requirements.txt
RUN pip install -r ./requirements/requirements.txt
RUN sed -i 's/^# \([Dd]eep[Ss]peed\)/\1/' ./requirements/requirements.txt
RUN sed -i 's/^# \(torch\)/\1/' ./requirements/requirements.txt
RUN sed -i 's/^# \(apex\)/\1/' ./requirements/requirements.txt
RUN pip install -r ./requirements/requirements_dev.txt

# Install GitHub actions runner
RUN mkdir -p /actions-runner
WORKDIR /actions-runner
RUN curl -o actions-runner-linux-x64-2.295.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.295.0/actions-runner-linux-x64-2.295.0.tar.gz
RUN echo "a80c1ab58be3cd4920ac2e51948723af33c2248b434a8a20bd9b3891ca4000b6  actions-runner-linux-x64-2.295.0.tar.gz" | shasum -a 256 -c
RUN tar xzf ./actions-runner-linux-x64-2.295.0.tar.gz

# Copy all other project files
WORKDIR /app
COPY . /app

# Create non-root user and give permissions to /app, /actions-runner, and /deepspeed
RUN useradd -ms /bin/bash docker
RUN chown -R docker:docker /app
RUN chown -R docker:docker /actions-runner
RUN chown -R docker:docker /deepspeed
USER docker

# expose the Jupyter port 8888
EXPOSE 8888

# print versions
RUN nvcc --version
RUN python --version
RUN python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
# RUN python -c "import transformers; print(transformers.__version__);"
# RUN ds_report

CMD ["jupyter", "notebook"]
