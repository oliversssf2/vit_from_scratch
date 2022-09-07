FROM oliversssf1/python-cuda:poetry-conda-python-cuda11.7.0 as base
ENV PATH "$PATH:/root/.local/bin"

RUN apt update -y && apt install -y htop tmux vim ranger

ARG WORKSPACE_PATH
WORKDIR ${WORKSPACE_PATH}

RUN eval "$($HOME/miniconda/bin/conda shell.bash hook)" && \
  conda create --name vit python=3.10.4 && \
  conda activate vit
  # pip3 install -r requirements.txt
# COPY pyproject.toml pyproject.toml
# COPY poetry.lock poetry.lock
# RUN poetry install

