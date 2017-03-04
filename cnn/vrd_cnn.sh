#!/bin/bash

BATCH_SIZE=25
DATA_EPOCHS=30

DATA_DIR="../data/vrd/"
OBJ_LIST=$DATA_DIR"mat/objectListN.mat"
REL_LIST=$DATA_DIR"mat/predicate.mat"

# ==================================================================================================
OUTPUT_SIZE=100
WHICH_NET="objnet"
MODEL_DIR="../data/vrd/models/cnn/objnet_run2/"
WEIGHTS=$MODEL_DIR"vrd_trained_8.npy"
# ==========================================================

MAT_FILE=$DATA_DIR"mat/annotation_train.mat"
IMG_DIR=$DATA_DIR"images/train/"
MEAN_FILE=$DATA_DIR"images/mean_train.npy"
SAVE_FILE=$MODEL_DIR"out_train.npy"

python vrd_cnn.py --which_net $WHICH_NET --output_size $OUTPUT_SIZE --weights $WEIGHTS --save_file $SAVE_FILE --batch_size $BATCH_SIZE --mat_file $MAT_FILE --img_dir $IMG_DIR --mean_file $MEAN_FILE --data_epochs $DATA_EPOCHS --obj_list $OBJ_LIST --rel_list $REL_LIST

# ----------------------------------------------------------

MAT_FILE=$DATA_DIR"mat/annotation_test.mat"
IMG_DIR=$DATA_DIR"images/test/"
MEAN_FILE=$DATA_DIR"images/mean_test.npy"
SAVE_FILE=$MODEL_DIR"out_test.npy"

python vrd_cnn.py --which_net $WHICH_NET --output_size $OUTPUT_SIZE --weights $WEIGHTS --save_file $SAVE_FILE --batch_size $BATCH_SIZE --mat_file $MAT_FILE --img_dir $IMG_DIR --mean_file $MEAN_FILE --data_epochs $DATA_EPOCHS --obj_list $OBJ_LIST --rel_list $REL_LIST

# ==================================================================================================
OUTPUT_SIZE=70
WHICH_NET="relnet"
MODEL_DIR="../data/vrd/models/cnn/relnet_run1/"
WEIGHTS=$MODEL_DIR"vrd_trained_9.npy"
# ==========================================================

MAT_FILE=$DATA_DIR"mat/annotation_train.mat"
IMG_DIR=$DATA_DIR"images/train/"
MEAN_FILE=$DATA_DIR"images/mean_train.npy"
SAVE_FILE=$MODEL_DIR"out_train.npy"

python vrd_cnn.py --which_net $WHICH_NET --output_size $OUTPUT_SIZE --weights $WEIGHTS --save_file $SAVE_FILE --batch_size $BATCH_SIZE --mat_file $MAT_FILE --img_dir $IMG_DIR --mean_file $MEAN_FILE --data_epochs $DATA_EPOCHS --obj_list $OBJ_LIST --rel_list $REL_LIST

# ----------------------------------------------------------

MAT_FILE=$DATA_DIR"mat/annotation_test.mat"
IMG_DIR=$DATA_DIR"images/test/"
MEAN_FILE=$DATA_DIR"images/mean_test.npy"
SAVE_FILE=$MODEL_DIR"out_test.npy"

python vrd_cnn.py --which_net $WHICH_NET --output_size $OUTPUT_SIZE --weights $WEIGHTS --save_file $SAVE_FILE --batch_size $BATCH_SIZE --mat_file $MAT_FILE --img_dir $IMG_DIR --mean_file $MEAN_FILE --data_epochs $DATA_EPOCHS --obj_list $OBJ_LIST --rel_list $REL_LIST
