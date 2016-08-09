import sys
import numpy as np
from detect import ConvNets, Model
from utils import make_word_list, make_w2v, make_w2v_dict
from utils2 import loadmat, get_data


# Initialize model
conv = ConvNets('data/models/objnet/', 'data/models/relnet/', 'data/img/')

# Prepare data
obj_dict = {r:i for i,r in enumerate(loadmat('data/vrd/objectListN.mat')['objectListN'])}
rel_dict = {r:i for i,r in enumerate(loadmat('data/vrd/predicate.mat')['predicate'])}

#a_test  = loadmat('annotation_test.mat')['annotation_test']
a_train = loadmat('data/vrd/annotation_train.mat')['annotation_train']
# obj_train, rel_train = get_data(a_train, obj_dict, rel_dict, 'images/train/')
# obj_test, rel_test   = get_data(a_test,  obj_dict, rel_dict, 'images/test/')


train_splits = 200
s = len(a_train) / train_splits
for e in range(0, train_splits):
    iter_data = a_train[e*s : (e+1)*s]
    obj_data, rel_data = get_data(iter_data, obj_dict, rel_dict, 'data/vrd/images/train/')
    conv.train_cnn('data/models/objnet/', obj_data, new_layer=100, ckpt_file='trained.ckpt', init_weights='data/models/objnet/vgg16.npy')
    conv.train_cnn('data/models/relnet/', rel_data, new_layer=70,  ckpt_file='trained.ckpt', init_weights='data/models/relnet/vgg16.npy')



# obj_test, rel_test   = get_data(a_test,  obj_dict, rel_dict, 'images/test/')


# Train model










