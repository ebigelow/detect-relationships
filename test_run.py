import sys
import numpy as np
from detect import ConvNets, Model
from utils import make_word_list, make_w2v, make_w2v_dict
from utils2 import loadmat, get_data, batchify_data


# Initialize model
conv = ConvNets('data/models/objnet/', 'data/models/relnet/', 'data/img/')

# Prepare data
obj_dict = {r:i for i,r in enumerate(loadmat('data/vrd/objectListN.mat')['objectListN'])}
rel_dict = {r:i for i,r in enumerate(loadmat('data/vrd/predicate.mat')['predicate'])}

a_test  = loadmat('data/vrd/annotation_test.mat')['annotation_test']
obj_test, rel_test = get_data(a_test[:20], obj_dict, rel_dict, 'data/vrd/images/test/')

# Test Model
accuracy_o = conv.test(obj_test, 'data/models/objnet/', ckpt_file='trained.ckpt', new_layer=100)
accuracy_r = conv.test(rel_test, 'data/models/relnet/', ckpt_file='trained.ckpt', new_layer=70)

print 'OBJ ACCURACY: ', accuracy_o
print 'REL ACCURACY: ', accuracy_r

