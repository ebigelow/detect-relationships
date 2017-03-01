#!/bin/bash




OUTPUT_SIZE=100
WHICH_NET="objnet"
INIT_PATH="../data/vgg16.npy"
SAVE_DIR="../data/vrd/models/objnet_run1/"
#OUTPUT_SIZE=70
#WHICH_NET="relnet"
#INIT_PATH="data/models/relnet/vgg16_trained.npy"
#SAVE_PATH="data/models/relnet/vgg16_trained2.npy"

BATCH_SIZE=10
DATA_EPOCHS=25
META_EPOCHS=1

DATA_DIR="../data/vrd/"
OBJ_LIST=$DATA_DIR"mat/objectListN.mat"
REL_LIST=$DATA_DIR"mat/predicate.mat"
TRAIN_MAT=$DATA_DIR"mat/annotation_train.mat"
TEST_MAT=$DATA_DIR"mat/annotation_test.mat"
TRAIN_IMGS=$DATA_DIR"images/train/"
TEST_IMGS=$DATA_DIR"images/test/"
MEAN="./mean.npy"


mkdir $SAVE_DIR
mkdir $SAVE_DIR"summaries"

python vrd_train_cnn.py --output_size $OUTPUT_SIZE --which_net $WHICH_NET --init_path $INIT_PATH --save_path $SAVE_PATH --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --meta_epochs $META_EPOCHS --obj_list $OBJ_LIST --rel_list $REL_LIST --train_mat $TRAIN_MAT --test_mat $TEST_MAT --train_imgs $TRAIN_IMGS --test_imgs $TEST_IMGS --mean $MEAN

#python train_cnn.py --gpu_mem_fraction 0.7 --output_size 100 --init_path "data/models/objnet/vgg16.npy" --save_path "data/models/objnet/vgg16_trained.npy" --batch_size 10 --save_freq 10 --meta_epochs 20 --obj_list "data/vrd/objectListN.mat" --rel_list "data/vrd/predicate.mat" --train_mat "data/vrd/annotation_train.mat" --test_mat "data/vrd/annotation_test.mat"
