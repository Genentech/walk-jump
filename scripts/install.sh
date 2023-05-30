#!/usr/bin/env bash

set -eux

if command -v mamba
then
 mamba env create --file env.yml
else
 conda env create --file env.yml
fi

eval "$(conda shell.bash hook)"
conda activate walkjump
pip install -e .