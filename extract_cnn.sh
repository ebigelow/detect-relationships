#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/localdisk/ebigelow/lib/cudnn-7.0/lib64/
source ~/.virtualenvs/tf/bin/activate



GPU_FRACTION=0.9

WHICH_NET='objnet'
OUTPUT_SIZE=100
WEIGHTS="data/models/objnet/vgg16_trained3.npy"
SAVE_FILE="data/models/objnet/obj_probs.npy"
LAYER_NAME="prob"
#WHICH_NET='relnet'
#OUTPUT_SIZE=70
#WEIGHTS="data/models/relnet/vgg16_trained2.npy"
#SAVE_FILE="data/models/relnet/feature_dict.npy"
#LAYER_NAME="fc7"

BATCH_SIZE=100

MAT_FILE="data/vrd/annotation_train.mat"
IMG_DIR="data/vrd/images/train/"
MEAN_FILE="./mean.npy"



python extract_cnn.py --gpu_mem_fraction $GPU_FRACTION --which_net $WHICH_NET --output_size $OUTPUT_SIZE --layer $LAYER_NAME --WEIGHTS $WEIGHTS --save_file $SAVE_FILE --batch_size $BATCH_SIZE --mat_file $MAT_FILE --img_dir $IMG_DIR --mean_file $MEAN_FILE
