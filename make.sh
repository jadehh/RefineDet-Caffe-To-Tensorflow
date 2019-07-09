#!/usr/bin/env bash
cd ./utils/

CUDA_PATH=/usr/local/cuda/

/home/jade/anaconda2/bin/python build.py build_ext --inplace

cd ..