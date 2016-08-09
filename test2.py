import sys
import numpy as np
from detect import ConvNets, Model
from utils import make_word_list, make_w2v, make_w2v_dict
from utils2 import loadmat, get_data, batchify_data


# Initialize model
conv = ConvNets('data/model/objnet/', 'data/model/relnet/', 'data/img/')

# Prepare data
obj_dict = {r:i for i,r in enumerate(loadmat('objectListN.mat')['objectListN'])}
rel_dict = {r:i for i,r in enumerate(loadmat('predicate.mat')['predicate'])}

a_test  = loadmat('annotation_test.mat')['annotation_test']
a_train = loadmat('annotation_train.mat')['annotation_train']
# obj_train, rel_train = get_data(a_train, obj_dict, rel_dict, 'images/train/')
# obj_test, rel_test   = get_data(a_test,  obj_dict, rel_dict, 'images/test/')


train_splits = 10
s = np.ceil(float(len(a_train)) / train_splits)
for e in range(0, train_splits):
    iter_data = a_train[e*s : (e+1)*s]
    obj_data, rel_data = get_data(iter_data, obj_dict, rel_dict, 'images/train/')
    obj_data, rel_data = (batchify_data(obj_data, 10), batchify_data(rel_data, 10))
    conv.train_cnn('data/model/objnet/', obj_data, new_layer=100, ckpt_file='trained.ckpt', init_weights='data/model/vgg16.npy')
    conv.train_cnn('data/model/relnet/', rel_data, new_layer=70,  ckpt_file='trained.ckpt', init_weights='data/model/vgg16.npy')



# obj_test, rel_test   = get_data(a_test,  obj_dict, rel_dict, 'images/test/')


# Train model










