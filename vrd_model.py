import numpy as np
from utils import make_w2v, loadmat, scene_graphs2triplets
from model import Model

import sys
sys.path.append('/localdisk/ebigelow/lib/visual_genome_python_driver')
import src.local as vg


# --------------------------------------------------------------------------------------------------
# Arguments

obj_mat   = 'data/vrd/objectListN.mat'
rel_mat   = 'data/vrd/predicate.mat'
w2v_file  = 'data/w2v.npy'
obj_file  = 'data/models/objnet/obj_probs.npy'
rel_file  = 'data/models/relnet/rel_feats.npy'
json_file = 'data/vrd/train.json'

# --------------------------------------------------------------------------------------------------
# Load data

obj_dict = {r:i for i,r in enumerate(loadmat(obj_mat)['objectListN'])}
rel_dict = {r:i for i,r in enumerate(loadmat(rel_mat)['predicate'])}
word2idx = {'obj':obj_dict, 'rel':rel_dict}

w2v = make_w2v(word2idx)
np.save(w2v_file, w2v)
# w2v = np.load('data/w2v.npy')

obj_probs = np.load(obj_file).item()
rel_feats = np.load(rel_file).item()

scene_graphs = vg.GetSceneGraphsVRD(json_file)
D = sg_to_triplets(scene_graphs, word2idx)

# --------------------------------------------------------------------------------------------------
# Run model

model = Model(obj_probs, rel_feats, w2v, word2idx)
model.SGD(D)
