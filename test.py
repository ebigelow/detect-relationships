import sys
import numpy as np
from detect import ConvNets, Model
from utils import make_word_list, make_w2v, make_w2v_dict

# sys.path.append('/Users/eric/code/visual_genome_python_driver')
sys.path.append('/localdisk/ebigelow/lib/visual_genome_python_driver')
import src.local as vg


# scene_graphs = vg.GetSceneGraphs(0, -1, maxRels=100000)
scene_graphs = vg.GetSceneGraphsModified(0, -1, maxRels=100000)
# 82239 scene graphs
# 1641647 objects
# 705317 relationships


## sg1 = vg.GetSceneGraphs(200); sg2 = vg.GetSceneGraphs(205)
## scene_graphs = [sg1, sg2, sg1, sg2, sg1, sg2, sg1, sg2, sg1, sg2]

conv = ConvNets('data/model/', 'data/model/', 'data/img/')
obj_probs, rel_feats, obj_dict, rel_dict = conv.run_cnns(scene_graphs)

print '*'*100, '\nCNN STUFF DONE... now the hard part!!!!'

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












