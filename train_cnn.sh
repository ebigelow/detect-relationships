#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/localdisk/ebigelow/lib/cudnn-7.0/lib64/
#export CUDNN_ROOT='/u/ebigelow/lib/cuda/lib64/libcudnn/'
#export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/u/yli/.local/cudnn-6.5-linux-x64-v2/
source ~/.virtualenvs/tf/bin/activate


python train_cnn.py                                     \
  -gpu_mem_fraction 0.9                                 \
  -output_size 100                                      \
  -init_path "data/models/objnet/vgg16.npy"             \
  -save_path "data/models/objnet/vgg16_trained.npy"     \
  -batch_size 10                                        \
  -save_freq 200                                         \
  -meta_epochs 20                                       \
  -obj_list  "data/vrd/objectListN.mat"                 \
  -rel_list  "data/vrd/predicate.mat"                   \
  -train_mat "data/vrd/annotation_train.mat"            \
  -test_mat  "data/vrd/annotation_test.mat"             

#python train_cnn.py -gpu_mem_fraction 0.7 -output_size 100 -init_path "data/models/objnet/vgg16.npy" -save_path "data/models/objnet/vgg16_trained.npy" -batch_size 10 -save_freq 10 -meta_epochs 20 -obj_list "data/vrd/objectListN.mat" -rel_list "data/vrd/predicate.mat" -train_mat "data/vrd/annotation_train.mat" -test_mat "data/vrd/annotation_test.mat" 

