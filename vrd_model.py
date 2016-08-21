import sys
import numpy as np
from detect import ConvNets, Model
from utils import make_word_list, make_w2v, make_w2v_dict

# sys.path.append('/Users/eric/code/visual_genome_python_driver')
sys.path.append('/localdisk/ebigelow/lib/visual_genome_python_driver')
import src.local as vg


# Parameters
ckpt_file = 'ckpt/model1.ckpt'


obj_params = {'cnn_dir':'data/models/objnet/',
              'ckpt_file':ckpt_file,
              'new_layer':100 }
rel_params = {'cnn_dir':'data/models/relnet/',
              'ckpt_file':ckpt_file,
              'new_layer':70 }


obj_probs = run_cnn(obj_train, cnn_dir='data/models/objnet/', ckpt_file='model.ckpt', layer='prob', new_layer=100)
rel_feats = run_cnn(rel_train, cnn_dir='data/models/relnet/', ckpt_file='model.ckpt', layer='fc7',  new_layer=70)

# Prepare data
obj_dict = {r:i for i,r in enumerate(loadmat('data/vrd/objectListN.mat')['objectListN'])}
rel_dict = {r:i for i,r in enumerate(loadmat('data/vrd/predicate.mat')['predicate'])}








# data_params = {'dataDir':'data/vrd/json/',
#                'imageDataDir':'data/vrd/json/by-id/'}
# vg.SaveSceneGraphsById(**data_params)
# scene_graphs = vg.GetSceneGraphs(0, -1, maxRels=100000, **data_params)
#
# conv = ConvNets('data/model/', 'data/model/', 'data/img/')
# obj_probs, rel_feats, obj_dict, rel_dict = run_cnns(TODO: code)
#
# print '*'*100, '\nCNN STUFF DONE'
#
# n = 26
# k = 12
#
# word_list = make_word_list(scene_graphs, n)
# w2v_dict = make_w2v_dict(word_list)
# w2v = make_w2v(word_list)
# np.save('data/w2v.npy', w2v)
# # w2v = np.load('data/w2v.npy')
#
# model = Model(obj_probs, rel_feats, obj_dict, rel_dict, w2v, w2v_dict, n)
# D = model.load_data(scene_graphs)
# model.SGD(D)
