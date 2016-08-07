import sys
import numpy as np
from detect import ConvNets, Model
from utils import make_word_list, make_w2v, make_w2v_dict
from utils2 import loadmat, get_data

# Prepare data
obj_dict = {r:i for i,r in enumerate(loadmat('objectListN.mat')['objectListN'])}
rel_dict = {r:i for i,r in enumerate(loadmat('predicate.mat')['predicate'])}

a_test  = loadmat('annotation_test.mat')['annotation_test']
a_train = loadmat('annotation_train.mat')['annotation_train']

obj_train, rel_train = get_data(a_train, obj_dict, rel_dict)
obj_test, rel_test = get_data(a_test, obj_dict, rel_dict)


# Train model
conv = ConvNets('data/model/objnet/', 'data/model/relnet/', 'data/img/')

conv.train_cnn('data/model/objnet/', obj_data, new_layer=100, ckpt_file='trained.ckpt', init_weights='data/model/vgg16.npy')
conv.train_cnn('data/model/relnet/', rel_data, new_layer=70, ckpt_file='trained.ckpt', init_weights='data/model/vgg16.npy')









