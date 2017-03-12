import numpy as np
import os, sys; sys.path.append('..')
from utils import loadmat, mat_to_triplets, group_triplets, ruid2feats
from model.model import Model


# --------------------------------------------------------------------------------------------------
# Arguments

data_dir = '../data/vrd/'

save_dir   = data_dir + 'models/vrd/run1/'
objnet_dir = data_dir + 'models/cnn/objnet_run2/'
relnet_dir = data_dir + 'models/cnn/relnet_run1/'

obj_mat   = data_dir + 'mat/objectListN.mat'
rel_mat   = data_dir + 'mat/predicate.mat'
train_mat = data_dir + 'mat/annotation_train.mat'
test_mat  = data_dir + 'mat/annotation_test.mat'

w2v_file  = data_dir   + 'w2v.npy'
train_obj = objnet_dir + 'out_train.npy'
test_obj  = objnet_dir + 'out_test.npy'
train_rel = relnet_dir + 'out_train.npy'
test_rel  = relnet_dir + 'out_test.npy'


# --------------------------------------------------------------------------------------------------
# Load data

obj_dict = {r:i for i,r in enumerate(loadmat(obj_mat)['objectListN'])}
rel_dict = {r:i for i,r in enumerate(loadmat(rel_mat)['predicate'])}

word2idx = {'obj':obj_dict, 'rel':rel_dict}
w2v = np.load(w2v_file).item()

# CNN features and probabilities
obj_train = { (k[0], k[2]):v['prob'] for k,v in np.load(train_obj).item().items()  if k is not None}
obj_test  = { (k[0], k[2]):v['prob'] for k,v in np.load(test_obj).item().items()   if k is not None }
rel_train = { ruid2feats(k):v['fc7']  for k,v in np.load(train_rel).item().items() if k is not None }
rel_test  = { ruid2feats(k):v['fc7']  for k,v in np.load(test_rel).item().items()  if k is not None}

# Training data (triplets)
mat_train = loadmat(train_mat)['annotation_train']
D_train   = mat_to_triplets(mat_train, word2idx)
Ds_train  = group_triplets(D_train).values()

mat_test = loadmat(test_mat)['annotation_test']
D_test   = mat_to_triplets(mat_test, word2idx)
Ds_test  = group_triplets(D_test).values()
test_data = (Ds_test, obj_test, rel_test)

# --------------------------------------------------------------------------------------------------
# Run model

model = Model(obj_train, rel_train, D_train, w2v, word2idx,
              learning_rate=0.05, lamb1=.01, lamb2=.005,
              noise=1e-10, K_samples=500000, max_iters=20)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model.SGD(Ds_train, test_data, save_file=save_dir+'weights.npy', ablate=[])
