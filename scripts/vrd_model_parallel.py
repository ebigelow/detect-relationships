import numpy as np
from utils import loadmat, mat_to_triplets, batch_triplets
from model import Model


# --------------------------------------------------------------------------------------------------
# Arguments

obj_mat   = 'data/vrd/objectListN.mat'
rel_mat   = 'data/vrd/predicate.mat'
w2v_file  = 'data/word2vec/w2v.npy'
obj_file  = 'data/models/objnet/obj_probs.npy'
rel_file  = 'data/models/relnet/rel_feats.npy'
train_mat = 'data/vrd/annotation_train.mat'

# --------------------------------------------------------------------------------------------------
# Load data

obj_dict = {r:i for i,r in enumerate(loadmat(obj_mat)['objectListN'])}
rel_dict = {r:i for i,r in enumerate(loadmat(rel_mat)['predicate'])}

word2idx = {'obj':obj_dict, 'rel':rel_dict}
w2v = np.load(w2v_file)

obj_probs = np.load(obj_file).item()
rel_feats = np.load(rel_file).item()

mat = loadmat(train_mat)['annotation_train']
D   = mat_to_triplets(mat, word2idx)
Ds  = batch_triplets(D)

# --------------------------------------------------------------------------------------------------
# Run model

model = Model(obj_probs, rel_feats, w2v, word2idx, learning_rate=0.1, lamb1=5e-2, max_iters=50, noise=1.0)
#import ipdb; ipdb.set_trace()
# model.SGD_parallel_1(D)
model.SGD_parallel_2(D)
