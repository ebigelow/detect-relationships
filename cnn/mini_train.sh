#!/bin/bash



# Set up save directory
setup_savedir()
{
  SAVE_DIR=$1
  if [ -d "$SAVE_DIR" ]; then
    rm -r $SAVE_DIR
  fi
  mkdir $SAVE_DIR
  mkdir $SAVE_DIR"summaries"
  mkdir $SAVE_DIR"out"
}



# ------------------------------------------------------------------------------------------


MINI_DIR="/home/eric/data/mini/"
MEAN_FILE="/home/eric/data/vrd/images/mean_train.npy"
INIT_PATH="/home/eric/data/vgg16.npy"

LEARNING_RATE=0.1
BATCH_SIZE=25
META_EPOCHS=10
DATA_EPOCHS=60

TEST_FREQ=20
TEST_SAMPLES=30

REL_CAP=2000





# ------------------------------------------------------------------------------------------
# VRD Dataset

TEST_DATA=$MINI_DIR"vrd_test.npy"
TRAIN_DATA=$MINI_DIR"vrd_train.npy"
TRAIN_IMGS="/home/eric/data/vrd/images/train/"
TEST_IMGS="/home/eric/data/vrd/images/test/"



# ========================================================
# ObjNet

OUTPUT_SIZE=96
WHICH_NET="obj"

setup_savedir $MINI_DIR"models/cnn/obj_vrd3/"
python mini_train.py --output_size $OUTPUT_SIZE --which_net $WHICH_NET --init_path $INIT_PATH --save_dir $SAVE_DIR --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --meta_epochs $META_EPOCHS --learning_rate $LEARNING_RATE --optimizer "gradient" --test_data $TEST_DATA --train_data $TRAIN_DATA --test_imgs $TEST_IMGS --train_imgs $TRAIN_IMGS --mean_file $MEAN_FILE --test_freq $TEST_FREQ --test_samples $TEST_SAMPLES --rel_cap $REL_CAP

# # ========================================================
# # RelNet
#
# OUTPUT_SIZE=40
# WHICH_NET="rel"
#
# setup_savedir $MINI_DIR"models/cnn/rel_vrd3/"
# python mini_train.py --output_size $OUTPUT_SIZE --which_net $WHICH_NET --init_path $INIT_PATH --save_dir $SAVE_DIR --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --meta_epochs $META_EPOCHS --learning_rate $LEARNING_RATE --optimizer "gradient" --test_data $TEST_DATA --train_data $TRAIN_DATA --test_imgs $TEST_IMGS --train_imgs $TRAIN_IMGS --mean_file $MEAN_FILE --test_freq $TEST_FREQ --test_samples $TEST_SAMPLES --rel_cap $REL_CAP
#
#
#
# # ------------------------------------------------------------------------------------------
# # Visual Genome
#
# TEST_DATA=$MINI_DIR"vg_test.npy"
# TRAIN_DATA=$MINI_DIR"vg_train.npy"
# TRAIN_IMGS="/home/eric/data/vg/images/"
# TEST_IMGS="/home/eric/data/vg/images/"
#
# META_EPOCHS=5
#
# # ========================================================
# # ObjNet
#
# OUTPUT_SIZE=96
# WHICH_NET="obj"
#
# setup_savedir $MINI_DIR"models/cnn/obj_vg3/"
# python mini_train.py --output_size $OUTPUT_SIZE --which_net $WHICH_NET --init_path $INIT_PATH --save_dir $SAVE_DIR --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --meta_epochs $META_EPOCHS --learning_rate $LEARNING_RATE --optimizer "gradient" --test_data $TEST_DATA --train_data $TRAIN_DATA --test_imgs $TEST_IMGS --train_imgs $TRAIN_IMGS --mean_file $MEAN_FILE --test_freq $TEST_FREQ --test_samples $TEST_SAMPLES --rel_cap $REL_CAP
#
# # ========================================================
# # RelNet
#
# OUTPUT_SIZE=40
# WHICH_NET="rel"
#
# setup_savedir $MINI_DIR"models/cnn/rel_vg3/"
# python mini_train.py --output_size $OUTPUT_SIZE --which_net $WHICH_NET --init_path $INIT_PATH --save_dir $SAVE_DIR --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --meta_epochs $META_EPOCHS --learning_rate $LEARNING_RATE --optimizer "gradient" --test_data $TEST_DATA --train_data $TRAIN_DATA --test_imgs $TEST_IMGS --train_imgs $TRAIN_IMGS --mean_file $MEAN_FILE --test_freq $TEST_FREQ --test_samples $TEST_SAMPLES --rel_cap $REL_CAP
