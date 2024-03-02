#!/bin/bash

conda env create -f environment.yml

conda activate selfrag

# Install flash-attn package
pip3 install flash-attn==2.3.6