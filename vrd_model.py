import numpy as np
from utils import loadmat, mat_to_triplets, batch_triplets
from model import Model

# import sys
# sys.path.append('/localdisk/ebigelow/lib/visual_genome_python_driver')
# import src.local as vg


# --------------------------------------------------------------------------------------------------
# Arguments

obj_mat   = 'data/vrd/objectListN.mat'
rel_mat   = 'data/vrd/predicate.mat'
w2v_file  = 'data/vrd/w2v.npy'

train_obj = 'data/vrd/obj_probs.npy'
train_rel = 'data/vrd/rel_feats.npy'
train_mat = 'data/vrd/annotation_train.mat'

test_obj = 'data/vrd/obj_probs_test.npy'
test_rel = 'data/vrd/rel_feats_test.npy'
test_mat = 'data/vrd/annotation_test.mat'


# --------------------------------------------------------------------------------------------------
# Load data

obj_dict = {r:i for i,r in enumerate(loadmat(obj_mat)['objectListN'])}
rel_dict = {r:i for i,r in enumerate(loadmat(rel_mat)['predicate'])}

word2idx = {'obj':obj_dict, 'rel':rel_dict}
w2v = np.load(w2v_file).item()

# Training data
obj_probs = np.load(train_obj).item()
rel_feats = np.load(train_rel).item()

mat_train = loadmat(train_mat)['annotation_train']
D_train   = mat_to_triplets(mat_train, word2idx)
Ds_train  = batch_triplets(D_train)

# Test data
obj_probs_ = np.load(test_obj).item()
rel_feats_ = np.load(test_rel).item()

mat_test = loadmat(test_mat)['annotation_test']
D_test   = mat_to_triplets(mat_test, word2idx)
Ds_test  = batch_triplets(D_test)

test_data = (Ds_test, obj_probs_, rel_feats_)


save_file = 'data/models/vrd_weights_3.npy'

# --------------------------------------------------------------------------------------------------
# Run model

model = Model(obj_probs, rel_feats, w2v, word2idx, learning_rate=1.0, lamb1=5e-3, max_iters=20, noise=1.0)
#import ipdb; ipdb.set_trace()
model.SGD(Ds_train, test_data, save_file=save_file)
