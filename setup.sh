#!/bin/bash

conda env create -f environment.yml

conda activate selfrag

# Install flash-attn package
pip3 install flash-attn==2.3.6

# Install faiss-gpu as it is not included in the `environment.yml`
conda install -c conda-forge faiss-gpu