FROM  nvcr.io/nvidia/pytorch:22.04-py3	
# CUDA 11.6
# PyTorch 1.12.0a0+bd13bc6 

# 
# too new
# FROM nvcr.io/nvidia/pytorch:22.05-py3
# # Latest image with torch 1.12.x
# # See https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
# # Image provides:
# # - Python 3.8
# # - CUDA 11.7
# # - Apex (v?)
# # - PyTorch 1.12.0a0+8a1a93a (requirements override this)

USER root

RUN mkdir /app
WORKDIR /app

# Copy requirements first to use cache
RUN mkdir /app/requirements
COPY ./requirements/ /app/requirements/

# Install
RUN pip install -r ./requirements/requirements.txt
RUN pip install -r ./requirements/requirements_dev.txt
#RUN pip install -e .

# Copy all other project code
COPY . /app

# expose the Jupyter port 8888
EXPOSE 8888

# print versions
RUN nvcc --version
RUN python --version
RUN python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
RUN python -c "import transformers; print(transformers.__version__);"

CMD ["jupyter", "notebook"]
