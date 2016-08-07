import numpy as np
import scipy.io as spio
from utils import square_crop
from cv2 import imread

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    
    from: `StackOverflow <http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries>`_
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict







def get_data(mat_data, obj_dict, rel_dict):
    obj_data = []
    rel_data = []

    for datum in mat_data:
        img = imread(datum.filename)
        img_rels = datum.relationship

        for rel in img_rels:
            ymin1, ymax1, xmin1, xmax1 = rel.subBox
            ymin2, ymax2, xmin2, xmax2 = rel.objBox
            ymin3, ymax3, xmin3, xmax3 = (min(ymin1, ymin2), max(ymax1, ymax2),
                                          min(xmin1, xmin2), max(xmax1, xmax2))
            h1, w1 = (ymax1 - ymin1, xmax1 - xmin1)
            h2, w2 = (ymax2 - ymin2, xmax2 - xmin2)
            h3, w3 = (ymax3 - ymin3, xmax3 - xmin3)

            img1 = square_crop(img, 226, xmin1, ymin1, w1, h1)
            img2 = square_crop(img, 226, xmin2, ymin2, w2, h2)
            img3 = square_crop(img, 226, xmin3, ymin3, w3, h3)

            s,v,o = rel.phrase
            obj_data.append(img1, obj_dict[s])
            obj_data.append(img2, obj_dict[o])
            rel_data.append(img3, rel_dict[v])

    return obj_data, rel_data



'''

obj_dict = {r:i for i,r in enumerate(loadmat('objectListN.mat')['objectListN'])}
rel_dict = {r:i for i,r in enumerate(loadmat('predicate.mat')['predicate'])}

a_test  = loadmat('annotation_test.mat')['annotation_test']
a_train = loadmat('annotation_train.mat')['annotation_train']

obj_train, rel_train = get_data(a_train, obj_dict, rel_dict)
obj_test, rel_test = get_data(a_test, obj_dict, rel_dict)


img_name = a_test[0].filename
img_rels = a_test[0].relationship
rel = img_rels[0]
s,v,o = rel.phrase
print rel.subBox, rel.objBox



'''









