#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/localdisk/ebigelow/lib/cudnn-7.0/lib64/
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/localdisk/ebigelow/lib/cudnn-7.5/lib64/
source ~/.virtualenvs/tf/bin/activate

# GPU=true
GPU_MEM=0.9
BATCH_SIZE=100
START_IDX=90000
END_IDX=90100

# IMG_DIR=""
# IMG_MEAN=""
# JSON_DIR=""
# JSON_ID_DIR=""
# LABEL_DICT=""
#MODEL_DIR="../data/vg/models"

#python vg_cnn.py --layer "prob" --which_net "objnet" --weights "../data/vg/models/objnet_vgg16_vg_9.npy" --save_file "../data/vg/models/obj_probs.npy" --use_gpu true --gpu_mem_fraction $GPU_MEM --batch_size $BATCH_SIZE --start_idx $START_IDX --end_idx $END_IDX
#python vg_cnn.py --layer "fc7" --which_net "relnet" --weights "../data/vg/models/relnet_vgg16_vg_8.npy" --save_file "../data/vg/models/rel_feats.npy" --use_gpu true --gpu_mem_fraction $GPU_MEM --batch_size $BATCH_SIZE --start_idx $START_IDX --end_idx $END_IDX
python vg_cnn.py --layer "prob" --which_net "relnet" --weights "../data/vg/models/relnet_vgg16_vg_8.npy" --save_file "../data/vg/models/rel_probs.npy" --use_gpu true --gpu_mem_fraction $GPU_MEM --batch_size $BATCH_SIZE --start_idx $START_IDX --end_idx $END_IDX
