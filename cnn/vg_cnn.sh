#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
#export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/localdisk/ebigelow/lib/cudnn-7.0/lib64/
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/u/ebigelow/lib/cudnn-8.0/lib64/
source ~/.virtualenvs/tf/bin/activate

GPU_MEM=0.95
BATCH_SIZE=70
START_IDX=90000
END_IDX=90100

# IMG_DIR=""
# IMG_MEAN=""
# JSON_DIR=""
# JSON_ID_DIR=""
# LABEL_DICT=""
MODEL_DIR="../data/vg/models/"

python vg_cnn.py --which_net "objnet" --weights $MODEL_DIR"objnet_vgg16_vg_9.npy" --save_file $MODEL_DIR"obj_out.npy" --use_gpu true --gpu_mem_fraction $GPU_MEM --batch_size $BATCH_SIZE --start_idx $START_IDX --end_idx $END_IDX
python vg_cnn.py --which_net "relnet" --weights $MODEL_DIR"relnet_vgg16_vg_8.npy" --save_file $MODEL_DIR"rel_out.npy" --use_gpu true --gpu_mem_fraction $GPU_MEM --batch_size $BATCH_SIZE --start_idx $START_IDX --end_idx $END_IDX
