import numpy as np
from utils import sg_to_triplets, batch_triplets
from model import Model

import sys
sys.path.append('/localdisk/ebigelow/lib/visual_genome_python_driver')
import src.local as vg


# --------------------------------------------------------------------------------------------------
# Arguments

# obj_mat   = 'data/vrd/objectListN.mat'
# rel_mat   = 'data/vrd/predicate.mat'
# w2v_file  = 'data/word2vec/w2v.npy'
# obj_file  = 'data/models/objnet/obj_probs.npy'
# rel_file  = 'data/models/relnet/rel_feats.npy'
# train_mat = 'data/vrd/annotation_train.mat'

json_dir    = 'data/vg/json/'
json_id_dir = 'data/vg/json/by-id/'
w2v_file    = 'data/word2vec/vg_w2v.npy'

obj_file  = 'data/models/objnet/vg_obj_probs.npy'
rel_file  = 'data/models/relnet/vg_rel_feats.npy'

# train_idx = 90000
train_idx = 100

# --------------------------------------------------------------------------------------------------
# Load training data

word2idx = TODO
w2v = np.load(w2v_file)


scene_graphs = vg.GetSceneGraphs(startIndex=0, endIndex=train_idx,
                                 dataDir=json_dir, imageDataDir=json_id_dir,
                                 minRels=1, maxRels=100)

obj_probs = np.load(obj_file).item()
rel_feats = np.load(rel_file).item()

D   = sg_to_triplets(scene_graphs, word2idx)
Ds  = batch_triplets(D)

# --------------------------------------------------------------------------------------------------
# Run model

model = Model(obj_probs, rel_feats, w2v, word2idx, learning_rate=0.1, lamb1=5e-2, max_iters=50, noise=1.0)
#import ipdb; ipdb.set_trace()
model.SGD(Ds[:-50])
