#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/localdisk/ebigelow/lib/cudnn-7.0/lib64/
source ~/.virtualenvs/tf/bin/activate



GPU_FRACTION=0.9

OUTPUT_SIZE=100
WHICH_NET="objnet"
INIT_PATH="data/models/objnet/vgg16_trained2.npy"
SAVE_PATH="data/models/objnet/vgg16_trained3.npy"
#OUTPUT_SIZE=70
#WHICH_NET="relnet"
#INIT_PATH="data/models/relnet/vgg16_trained.npy"
#SAVE_PATH="data/models/relnet/vgg16_trained2.npy"

BATCH_SIZE=10
SAVE_FREQ=1
DATA_EPOCHS=25
META_EPOCHS=20

OBJ_LIST="data/vrd/objectListN.mat"
REL_LIST="data/vrd/predicate.mat"
TRAIN_MAT="data/vrd/annotation_train.mat"
TEST_MAT="data/vrd/annotation_test.mat"
TRAIN_IMGS="data/vrd/images/train/"
TEST_IMGS="data/vrd/images/test/"
MEAN="./mean.npy"


python train_cnn.py --gpu_mem_fraction $GPU_FRACTION --output_size $OUTPUT_SIZE --which_net $WHICH_NET --init_path $INIT_PATH --save_path $SAVE_PATH --batch_size $BATCH_SIZE --save_freq $SAVE_FREQ --data_epochs $DATA_EPOCHS --meta_epochs $META_EPOCHS --obj_list $OBJ_LIST --rel_list $REL_LIST --train_mat $TRAIN_MAT --test_mat $TEST_MAT --train_imgs $TRAIN_IMGS --test_imgs $TEST_IMGS --mean $MEAN

#python train_cnn.py --gpu_mem_fraction 0.7 --output_size 100 --init_path "data/models/objnet/vgg16.npy" --save_path "data/models/objnet/vgg16_trained.npy" --batch_size 10 --save_freq 10 --meta_epochs 20 --obj_list "data/vrd/objectListN.mat" --rel_list "data/vrd/predicate.mat" --train_mat "data/vrd/annotation_train.mat" --test_mat "data/vrd/annotation_test.mat"
