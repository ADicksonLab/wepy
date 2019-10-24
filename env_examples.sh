#!/bin/bash

conda deactivate

env_name="wepy-examples"

# remove and recreate the env
conda env remove -y -n "$env_name"
conda create -y -n "$env_name" python=3.7
conda activate "$env_name"

# install the openmmtools dependencies
conda install -y -c 'conda-forge' \
      netcdf4 mpiplus pymbar pyyaml

# stick to an explicit openmm build for dev
echo "Installing openmm built with CUDA 10.0"
conda install --freeze-installed -y -c omnia/label/cuda100 openmm

pip install -r requirements.txt
pip install -r requirements_wepy_tools.txt

pip install -e .

