#!/bin/bash

VRD_DIR="/home/eric/data/vrd/"
OBJ_LIST=$VRD_DIR"mat/objectListN.mat"
REL_LIST=$VRD_DIR"mat/predicate.mat"

BATCH_SIZE=25
DATA_EPOCHS=60



# ==================================================================================================

CNN_DIR="/home/eric/data/mini/models/cnn1/obj_vg2/"
WEIGHTS=$CNN_DIR"weights_3.npy"
run_objnet() # save_file, mat_file, img_dir, mean_file
{
  python vrd_cnn.py --which_net "obj" --output_size 96 --weights $WEIGHTS --save_file $1 --batch_size $BATCH_SIZE --mat_file $2 --img_dir $3 --mean_file $4 --data_epochs $DATA_EPOCHS --obj_list $OBJ_LIST --rel_list $REL_LIST
}

run_objnet $CNN_DIR"out/vrd_train.npy" $VRD_DIR"mat/annotation_train.mat" $VRD_DIR"images/train/" $VRD_DIR"images/mean_train.npy"

run_objnet $CNN_DIR"out/vrd_test.npy" $VRD_DIR"mat/annotation_test.mat" $VRD_DIR"images/test/" $VRD_DIR"images/mean_train.npy"

# ==================================================================================================

CNN_DIR="/home/eric/data/mini/models/cnn1/rel_vg1/"
WEIGHTS=$CNN_DIR"weights_4.npy"

run_relnet() # save_file, mat_file, img_dir, mean_file
{
  python vrd_cnn.py --which_net "rel" --output_size 40 --weights $WEIGHTS --save_file $1 --batch_size $BATCH_SIZE --mat_file $2 --img_dir $3 --mean_file $4 --data_epochs $DATA_EPOCHS --obj_list $OBJ_LIST --rel_list $REL_LIST
}

run_relnet $CNN_DIR"out/vrd_train.npy" $VRD_DIR"mat/annotation_train.mat" $VRD_DIR"images/train/" $VRD_DIR"images/mean_train.npy"

run_relnet $CNN_DIR"out/vrd_test.npy" $VRD_DIR"mat/annotation_test.mat" $VRD_DIR"images/test/" $VRD_DIR"images/mean_train.npy"
