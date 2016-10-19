#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
#export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/localdisk/ebigelow/lib/cudnn-7.0/lib64/
source ~/.virtualenvs/tf/bin/activate

python vrd_model_tf.py
