#!/bin/bash


MINI_DIR="/home/eric/data/mini/"
MEAN_FILE="/home/eric/data/vrd/images/mean_train.npy"

BATCH_SIZE=25
DATA_EPOCHS=60


# ====================================================================================================

# --------------------------------------------------------------------------
# Objnet trained on VRD
CNN_DIR=$MINI_DIR"models/cnn2/obj_vrd1/"
WEIGHTS=$MINI_DIR"models/cnn2/obj_vrd1/weights_4.npy"  # TODO number
runtest()    # $1 : save_file   $2 : data_file   $3 : img_dir
{
  python mini_test.py --output_size 96 --which_net "obj" --weights $WEIGHTS --save_file $1 --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --data_file $2 --img_dir $3 --mean_file $MEAN_FILE
}
# runtest $CNN_DIR"out/vrd_test2.npy" $MINI_DIR"vrd_test2.npy" "/home/eric/data/vrd/images/test/"
# runtest $CNN_DIR"out/vrd_train2.npy" $MINI_DIR"vrd_train2.npy" "/home/eric/data/vrd/images/train/"
runtest $CNN_DIR"out/vg_test.npy" $MINI_DIR"vg_test.npy" "/home/eric/data/vg/images/"
runtest $CNN_DIR"out/vg_train.npy" $MINI_DIR"vg_train.npy" "/home/eric/data/vg/images/"

#--------------------------------------------------------------------------
# Relnet trained on VRD
CNN_DIR=$MINI_DIR"models/cnn2/rel_vrd1/"
WEIGHTS=$MINI_DIR"models/cnn2/rel_vrd1/weights_4.npy"  # TODO number
runtest()
{
 python mini_test.py --output_size 40 --which_net "rel" --weights $WEIGHTS --save_file $1 --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --data_file $2 --img_dir $3 --mean_file $MEAN_FILE
}
# runtest $CNN_DIR"out/vrd_test2.npy" $MINI_DIR"vrd_test2.npy" "/home/eric/data/vrd/images/test/"
# runtest $CNN_DIR"out/vrd_train2.npy" $MINI_DIR"vrd_train2.npy" "/home/eric/data/vrd/images/train/"
runtest $CNN_DIR"out/vg_test.npy" $MINI_DIR"vg_test.npy" "/home/eric/data/vg/images/"
runtest $CNN_DIR"out/vg_train.npy" $MINI_DIR"vg_train.npy" "/home/eric/data/vg/images/"


# # --------------------------------------------------------------------------
# # Objnet trained on VG
# CNN_DIR=$MINI_DIR"models/cnn2/obj_vg1/"
# WEIGHTS=$MINI_DIR"models/cnn2/obj_vg1/weights_1500.npy"  # TODO number
# runtest()
# {
#   python mini_test.py --output_size 96 --which_net "obj" --weights $WEIGHTS --save_file $1 --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --data_file $2 --img_dir $3 --mean_file $MEAN_FILE
# }
# runtest $CNN_DIR"out/vrd_test.npy" $MINI_DIR"vrd_test.npy" "/home/eric/data/vrd/images/test/"
# runtest $CNN_DIR"out/vrd_train.npy" $MINI_DIR"vrd_train.npy" "/home/eric/data/vrd/images/train/"
# runtest $CNN_DIR"out/vg_test.npy" $MINI_DIR"vg_test.npy" "/home/eric/data/vg/images/"
# runtest $CNN_DIR"out/vg_train.npy" $MINI_DIR"vg_train.npy" "/home/eric/data/vg/images/"
#
#
# # --------------------------------------------------------------------------
# # Relnet trained on VG
# CNN_DIR=$MINI_DIR"models/cnn2/rel_vg1/"
# WEIGHTS=$MINI_DIR"models/cnn2/rel_vg1/weights_1000.npy"  # TODO number
# runtest()
# {
#   python mini_test.py --output_size 40 --which_net "rel" --weights $WEIGHTS --save_file $1 --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --data_file $2 --img_dir $3 --mean_file $MEAN_FILE
# }
# runtest $CNN_DIR"out/vrd_test.npy" $MINI_DIR"vrd_test.npy" "/home/eric/data/vrd/images/test/"
# runtest $CNN_DIR"out/vrd_train.npy" $MINI_DIR"vrd_train.npy" "/home/eric/data/vrd/images/train/"
# runtest $CNN_DIR"out/vg_test.npy" $MINI_DIR"vg_test.npy" "/home/eric/data/vg/images/"
# runtest $CNN_DIR"out/vg_train.npy" $MINI_DIR"vg_train.npy" "/home/eric/data/vg/images/"
