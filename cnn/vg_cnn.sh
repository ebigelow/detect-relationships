

BATCH_SIZE=100
DATA_EPOCHS=500

# START_IDX=90000
# END_IDX=-1

MODEL_DIR="../data/models/"

# python vg_cnn.py --which_net "objnet" --weights $MODEL_DIR"objnet/vgg16_vg_trained_9.npy" --save_file $MODEL_DIR"out/obj_test1.npy" --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx 90000  --end_idx 92000 
# python vg_cnn.py --which_net "objnet" --weights $MODEL_DIR"objnet/vgg16_vg_trained_9.npy" --save_file $MODEL_DIR"out/obj_test2.npy" --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx 92000  --end_idx 94000 
# python vg_cnn.py --which_net "objnet" --weights $MODEL_DIR"objnet/vgg16_vg_trained_9.npy" --save_file $MODEL_DIR"out/obj_test3.npy" --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx 94000  --end_idx 96000 
# python vg_cnn.py --which_net "objnet" --weights $MODEL_DIR"objnet/vgg16_vg_trained_9.npy" --save_file $MODEL_DIR"out/obj_test4.npy" --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx 96000  --end_idx 98000 
# python vg_cnn.py --which_net "objnet" --weights $MODEL_DIR"objnet/vgg16_vg_trained_9.npy" --save_file $MODEL_DIR"out/obj_test5.npy" --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx 98000  --end_idx 100000
# python vg_cnn.py --which_net "objnet" --weights $MODEL_DIR"objnet/vgg16_vg_trained_9.npy" --save_file $MODEL_DIR"out/obj_test6.npy" --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx 100000 --end_idx 102000
# python vg_cnn.py --which_net "objnet" --weights $MODEL_DIR"objnet/vgg16_vg_trained_9.npy" --save_file $MODEL_DIR"out/obj_test7.npy" --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx 102000 --end_idx 104000
# python vg_cnn.py --which_net "objnet" --weights $MODEL_DIR"objnet/vgg16_vg_trained_9.npy" --save_file $MODEL_DIR"out/obj_test8.npy" --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx 104000 --end_idx 106000
python vg_cnn.py --which_net "objnet" --weights $MODEL_DIR"objnet/vgg16_vg_trained_9.npy" --save_file $MODEL_DIR"out/obj_test9.npy"  --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx 106000 --end_idx 108000
python vg_cnn.py --which_net "objnet" --weights $MODEL_DIR"objnet/vgg16_vg_trained_9														.npy" --save_file $MODEL_DIR"out/obj_test10.npy" --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx 108000 --end_idx -1

python vg_cnn.py --which_net "relnet" --weights $MODEL_DIR"relnet/vgg16_vg_trained_8.npy" --save_file $MODEL_DIR"out/rel_test1.npy"  --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx 90000  --end_idx 92000 
python vg_cnn.py --which_net "relnet" --weights $MODEL_DIR"relnet/vgg16_vg_trained_8.npy" --save_file $MODEL_DIR"out/rel_test2.npy"  --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx 92000  --end_idx 94000 
python vg_cnn.py --which_net "relnet" --weights $MODEL_DIR"relnet/vgg16_vg_trained_8.npy" --save_file $MODEL_DIR"out/rel_test3.npy"  --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx 94000  --end_idx 96000 
python vg_cnn.py --which_net "relnet" --weights $MODEL_DIR"relnet/vgg16_vg_trained_8.npy" --save_file $MODEL_DIR"out/rel_test4.npy"  --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx 96000  --end_idx 98000 
python vg_cnn.py --which_net "relnet" --weights $MODEL_DIR"relnet/vgg16_vg_trained_8.npy" --save_file $MODEL_DIR"out/rel_test5.npy"  --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx 98000  --end_idx 100000
python vg_cnn.py --which_net "relnet" --weights $MODEL_DIR"relnet/vgg16_vg_trained_8.npy" --save_file $MODEL_DIR"out/rel_test6.npy"  --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx 100000 --end_idx 102000
python vg_cnn.py --which_net "relnet" --weights $MODEL_DIR"relnet/vgg16_vg_trained_8.npy" --save_file $MODEL_DIR"out/rel_test7.npy"  --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx 102000 --end_idx 104000
python vg_cnn.py --which_net "relnet" --weights $MODEL_DIR"relnet/vgg16_vg_trained_8.npy" --save_file $MODEL_DIR"out/rel_test8.npy"  --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx 104000 --end_idx 106000
python vg_cnn.py --which_net "relnet" --weights $MODEL_DIR"relnet/vgg16_vg_trained_8.npy" --save_file $MODEL_DIR"out/rel_test9.npy"  --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx 106000 --end_idx 108000
python vg_cnn.py --which_net "relnet" --weights $MODEL_DIR"relnet/vgg16_vg_trained_8.npy" --save_file $MODEL_DIR"out/rel_test10.npy" --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx 108000 --end_idx -1
# python vg_cnn.py --which_net "relnet" --weights $MODEL_DIR"relnet/vgg16_vg_trained_8.npy" --save_file $MODEL_DIR"out/rel_out.npy" --batch_size $BATCH_SIZE --data_epochs $DATA_EPOCHS --start_idx $START_IDX --end_idx $END_IDX
