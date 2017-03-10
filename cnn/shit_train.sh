
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





SAVE_FREQ=500




INIT_PATH="/home/eric/data/vgg16.npy"

BATCH_SIZE=25
DATA_EPOCHS=60
META_EPOCHS=5
TEST_FREQ=20

# DATA_DIR="/home/eric/data/vg/"
# IMG_DIR=$DATA_DIR"images/"
# MEAN_FILE="/home/eric/data/vrd/images/mean_train.npy"


LEARNING_RATE=0.1

# 
#
#
# OUTPUT_SIZE=40
# WHICH_NET="rel"
# SAVE_DIR="/home/eric/data/mini/models/cnn1/rel_vg1/"
#
# setup_savedir $SAVE_DIR
# python vg_train_cnn2.py --output_size $OUTPUT_SIZE --which_net $WHICH_NET --save_dir $SAVE_DIR --init_path $INIT_PATH --learning_rate $LEARNING_RATE --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --meta_epochs $META_EPOCHS --test_freq $TEST_FREQ --save_freq $SAVE_FREQ
#
#
# OUTPUT_SIZE=96
# WHICH_NET="obj"
# SAVE_DIR="/home/eric/data/mini/models/cnn1/obj_vg2/"
#
# setup_savedir $SAVE_DIR
# python vg_train_cnn2.py --output_size $OUTPUT_SIZE --which_net $WHICH_NET --save_dir $SAVE_DIR --init_path $INIT_PATH --learning_rate $LEARNING_RATE --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --meta_epochs $META_EPOCHS --test_freq $TEST_FREQ --save_freq $SAVE_FREQ
#
#
#
#
#
#
#
#
#















BATCH_SIZE=25
DATA_EPOCHS=60
META_EPOCHS=5
TEST_FREQ=20

DATA_DIR="/home/eric/data/vrd/"
OBJ_LIST=$DATA_DIR"mat/objectListN.mat"
REL_LIST=$DATA_DIR"mat/predicate.mat"
TRAIN_MAT=$DATA_DIR"mat/annotation_train.mat"
TEST_MAT=$DATA_DIR"mat/annotation_test.mat"
TRAIN_IMGS=$DATA_DIR"images/train/"
TEST_IMGS=$DATA_DIR"images/test/"
MEAN_FILE=$DATA_DIR"images/mean_train.npy"




LEARNING_RATE=0.1



OUTPUT_SIZE=96
WHICH_NET="obj"
SAVE_DIR="/home/eric/data/mini/models/cnn1/obj_vrd1/"

setup_savedir $SAVE_DIR
python vrd_train_cnn.py --output_size $OUTPUT_SIZE --which_net $WHICH_NET --init_path $INIT_PATH --save_dir $SAVE_DIR --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --meta_epochs $META_EPOCHS --obj_list $OBJ_LIST --rel_list $REL_LIST --train_mat $TRAIN_MAT --test_mat $TEST_MAT --train_imgs $TRAIN_IMGS --test_imgs $TEST_IMGS --mean_file $MEAN_FILE --test_freq $TEST_FREQ --learning_rate $LEARNING_RATE --optimizer "gradient" --save_freq $SAVE_FREQ


OUTPUT_SIZE=40
WHICH_NET="rel"
SAVE_DIR="/home/eric/data/mini/models/cnn1/rel_vrd1/"

setup_savedir $SAVE_DIR
python vrd_train_cnn.py --output_size $OUTPUT_SIZE --which_net $WHICH_NET --init_path $INIT_PATH --save_dir $SAVE_DIR --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --meta_epochs $META_EPOCHS --obj_list $OBJ_LIST --rel_list $REL_LIST --train_mat $TRAIN_MAT --test_mat $TEST_MAT --train_imgs $TRAIN_IMGS --test_imgs $TEST_IMGS --mean_file $MEAN_FILE --test_freq $TEST_FREQ --learning_rate $LEARNING_RATE --optimizer "gradient" --save_freq $SAVE_FREQ
