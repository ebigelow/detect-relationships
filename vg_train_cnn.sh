#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/localdisk/ebigelow/lib/cudnn-7.0/lib64/
source ~/.virtualenvs/tf/bin/activate


GPU_FRACTION=0.9

WHICH_NET="objnet"
INIT_PATH="data/models/objnet/vgg16_vg_trained_4.npy"
SAVE_PATH="data/models/objnet/vgg16_vg_trained.npy"
#WHICH_NET="relnet"
#INIT_PATH="data/models/vgg16.npy"
#SAVE_PATH="data/models/relnet/vgg16_vg_trained.npy"

LEARNING_RATE=0.5
BATCH_SIZE=20
DATA_EPOCHS=2000
META_EPOCHS=10
TRAIN_IDX=90000

IMG_DIR="data/vg/images/"
IMG_MEAN="./mean.npy"
LABEL_DICT="data/vg/json/vg_short/label_dict.npy"
JSON_DIR="data/vg/json/vg_short/"
JSON_ID_DIR="data/vg/json/vg_short/by-id/"

python vg_train_cnn.py \
  --gpu_mem_fraction $GPU_FRACTION \
  --which_net $WHICH_NET \
  --init_path $INIT_PATH \
  --save_path $SAVE_PATH \
  --batch_size $BATCH_SIZE \
  --data_epochs $DATA_EPOCHS \
  --meta_epochs $META_EPOCHS \
  --train_idx $TRAIN_IDX \
  --img_dir $IMG_DIR \
  --img_mean $IMG_MEAN \
  --label_dict $LABEL_DICT \
  --json_dir $JSON_DIR \
  --json_id_dir $JSON_ID_DIR \
  --learning_rate $LEARNING_RATE
