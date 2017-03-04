#!/bin/bash



INIT_PATH="../data/vgg16.npy"

OUTPUT_SIZE=100
WHICH_NET="objnet"
SAVE_DIR="../data/vrd/models/cnn/objnet_run3/"
# OUTPUT_SIZE=70
# WHICH_NET="relnet"
# SAVE_DIR="../data/vrd/models/cnn/relnet_run2/"

BATCH_SIZE=25
DATA_EPOCHS=30
META_EPOCHS=10
TEST_FREQ=20

DATA_DIR="../data/vrd/"
OBJ_LIST=$DATA_DIR"mat/objectListN.mat"
REL_LIST=$DATA_DIR"mat/predicate.mat"
TRAIN_MAT=$DATA_DIR"mat/annotation_train.mat"
TEST_MAT=$DATA_DIR"mat/annotation_test.mat"
TRAIN_IMGS=$DATA_DIR"images/train/"
TEST_IMGS=$DATA_DIR"images/test/"
MEAN_FILE=$DATA_DIR"images/mean_train.npy"


if [ -d "$SAVE_DIR" ]; then
  rm -r $SAVE_DIR
fi
mkdir $SAVE_DIR
mkdir $SAVE_DIR"summaries"


LEARNING_RATE=0.05

python vrd_train_cnn.py --output_size $OUTPUT_SIZE --which_net $WHICH_NET --init_path $INIT_PATH --save_dir $SAVE_DIR --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --meta_epochs $META_EPOCHS --obj_list $OBJ_LIST --rel_list $REL_LIST --train_mat $TRAIN_MAT --test_mat $TEST_MAT --train_imgs $TRAIN_IMGS --test_imgs $TEST_IMGS --mean_file $MEAN_FILE --test_freq $TEST_FREQ --learning_rate $LEARNING_RATE --optimizer "gradient"

# python vrd_train_cnn.py --output_size $OUTPUT_SIZE --which_net $WHICH_NET --init_path $INIT_PATH --save_dir $SAVE_DIR --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --meta_epochs $META_EPOCHS --obj_list $OBJ_LIST --rel_list $REL_LIST --train_mat $TRAIN_MAT --test_mat $TEST_MAT --train_imgs $TRAIN_IMGS --test_imgs $TEST_IMGS --mean_file $MEAN_FILE --test_freq $TEST_FREQ --optimizer "adagrad" --learning_rate 0.005 --initial_accumulator_value 0.1
