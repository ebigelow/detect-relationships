#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/localdisk/ebigelow/lib/cudnn-7.0/lib64/
#export CUDNN_ROOT='/u/ebigelow/lib/cuda/lib64/libcudnn/'
#export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/u/yli/.local/cudnn-6.5-linux-x64-v2/
source ~/.virtualenvs/tf/bin/activate


python test_run.py
