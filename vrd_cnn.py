import sys
import numpy as np
from utils import loadmat, get_data
from cnn import load_cnn, train_cnn, test_cnn


# Initialize model
conv = ConvNets('data/models/objnet/', 'data/models/relnet/', 'data/img/')

# Prepare data
obj_dict = {r:i for i,r in enumerate(loadmat('data/vrd/objectListN.mat')['objectListN'])}
rel_dict = {r:i for i,r in enumerate(loadmat('data/vrd/predicate.mat')['predicate'])}

a_train = loadmat('data/vrd/annotation_train.mat')['annotation_train']
a_test  = loadmat('data/vrd/annotation_test.mat')['annotation_test'][:30]

obj_test, rel_test = get_data(a_test, obj_dict, rel_dict, 'data/vrd/images/test/')

ckpt_file = 'ckpt/model1.ckpt'
meta_epochs = 20
bl = np.ceil(float(len(a_train)) / train_splits).astype(int)

for e in range(0, meta_epochs):
    print '~~~~~ Meta Batch: {}, [{}:{}] ~~~~~'.format(e, e*s, (e+1)*s)
    iter_data = a_train[e*bl : (e+1)*bl]
    obj_data, rel_data = get_data(iter_data, obj_dict, rel_dict, 'data/vrd/images/train/')

    obj_params = {'cnn_dir':'data/models/objnet/',
                  'ckpt_file':ckpt_file,
                  'new_layer':100 }
    train_cnn(obj_data,
              batch_size=10, init_weights='data/models/objnet/vgg16.npy', **obj_params)
    print '{} | OBJ ACCURACY: {}'.format(e, test_cnn(obj_test, **obj_params))

    rel_params = {'cnn_dir':'data/models/relnet/',
                  'ckpt_file':ckpt_file,
                  'new_layer':70 }
    train_cnn(rel_data,
              batch_size=10, init_weights='data/models/relnet/vgg16.npy', **rel_params)
    print '{} | REL ACCURACY: {}'.format(e, test_cnn(rel_test, **rel_params))
