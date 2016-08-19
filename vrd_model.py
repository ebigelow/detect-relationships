import sys
import numpy as np
from detect import ConvNets, Model
from utils import make_word_list, make_w2v, make_w2v_dict

# sys.path.append('/Users/eric/code/visual_genome_python_driver')
sys.path.append('/localdisk/ebigelow/lib/visual_genome_python_driver')
import src.local as vg

data_params = {'dataDir':'data/vrd/json/',
               'imageDataDir':'data/vrd/json/by-id/'}
vg.SaveSceneGraphsById(**data_params)
scene_graphs = vg.GetSceneGraphs(0, -1, maxRels=100000, **data_params)

conv = ConvNets('data/model/', 'data/model/', 'data/img/')
obj_probs, rel_feats, obj_dict, rel_dict = run_cnns(TODO: code)

print '*'*100, '\nCNN STUFF DONE'

n = 26
k = 12

word_list = make_word_list(scene_graphs, n)
w2v_dict = make_w2v_dict(word_list)
w2v = make_w2v(word_list)
np.save('data/w2v.npy', w2v)
# w2v = np.load('data/w2v.npy')

model = Model(obj_probs, rel_feats, obj_dict, rel_dict, w2v, w2v_dict, n)
D = model.load_data(scene_graphs)
model.SGD(D)
