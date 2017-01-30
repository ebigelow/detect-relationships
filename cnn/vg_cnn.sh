#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/u/ebigelow/lib/cudnn-8.0/lib64/
source ~/.virtualenvs/tf/bin/activate

GPU_MEM=0.95

BATCH_SIZE=70
DATA_EPOCHS=2000

START_IDX=0
END_IDX=-1

#IMG_DIR="../data/vg/images/"
#IMG_MEAN="../data/vg/mean.npy"
#JSON_DIR="../data/vg/json/"
#JSON_ID_DIR="../data/vg/json/by-id/"
#LABEL_DICT="../data/vg/label_dict.npy"
MODEL_DIR="../data/vg/models/"

python vg_cnn.py --which_net "objnet" --weights $MODEL_DIR"objnet_vgg16_vg_9.npy" --save_file $MODEL_DIR"obj_out.npy" --use_gpu true --gpu_mem_fraction $GPU_MEM --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx $START_IDX --end_idx $END_IDX
python vg_cnn.py --which_net "relnet" --weights $MODEL_DIR"relnet_vgg16_vg_8.npy" --save_file $MODEL_DIR"rel_out.npy" --use_gpu true --gpu_mem_fraction $GPU_MEM --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx $START_IDX --end_idx $END_IDX

skicka upload $MODEL_DIR"obj_out.npy" vrd/VisualGenome/out/obj_out.npy
skicka upload $MODEL_DIR"rel_out.npy" vrd/VisualGenome/out/rel_out.npy
