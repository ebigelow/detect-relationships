from skimage.transform import resize
from skimage.io import imread
import pickle
import numpy as np
import scipy.io as spio
from cv2 import cvtColor, COLOR_RGB2BGR
import inspect


# ---------------------------------------------------------------------------------------------------------
# Image processing

def square_crop(img, crop_size, x, y, w, h):
    """
    TODO: test

    """
    ih, iw, _ = img.shape

    if w == h:
        x1 = x
        x2 = x + w
        y1 = y
        y2 = y + h

    elif w > h:
        d1 = np.floor((w - h) / 2.)
        d2 = np.ceil((w - h) / 2.)

        if y - d1 < 0:
            y1 = 0
            y2 = y + h + d2 - (y - d1)
        elif (y + h + d2) > (ih - 1):
            y1 = y - d1 - ((y + h + d2) - (ih - 1))
            y2 = ih - 1
        else:
            y1 = y - d1
            y2 = y + h + d2
        x1 = x
        x2 = x + w

    elif h > w:
        d1 = np.floor((h - w) / 2.)
        d2 = np.ceil((h - w) / 2.)

        if x - d1 < 0:
            x1 = 0
            x2 = x + w + d2 - (x - d1)
        elif (x + w + d2) > (iw - 1):
            x1 = x - d1 - ((x + w + d2) - (iw - 1))
            x2 = iw - 1
        else:
            x1 = x - d1
            x2 = x + w + d2
        y1 = y
        y2 = y + h

    f = lambda c: int(max(c, 0))
    crop = img[f(y1):f(y2), f(x1):f(x2)]
    new_img = resize(crop, (crop_size, crop_size))

    return new_img


def rel_coords(scene_graphs):
    for sg in scene_graphs:
        for r in sg.relationships:
            s = r.subject
            o = r.object
            r.x = min(s.x, o.x)
            r.y = min(s.y, o.y)
            r.width  = max(s.x + s.width,  o.x + o.width ) - r.x
            r.height = max(s.y + s.height, o.y + o.height) - r.y

    return scene_graphs


# ---------------------------------------------------------------------------------------------------------
# Word2Vec

def fix_r(r):
    if r.predicate == 'of':
        r.predicate = 'NULL'
    if r.predicate == 'on the':
        r.predicate = 'on'
    if r.predicate == 'front of':
        r.predicate = 'front'
    r.subject = fix_o(r.subject)
    r.object = fix_o(r.object)
    return o

def fix_o(o):
    if o.names[0] == 'windw':
        o.names[0] = 'window'
    if o.names[0] == 'sky light':
        o.names[0] = 'skylight'
    return o

def fix_names(scene_graphs):
    for sg in scene_graphs:
        for r in sg.relationships:
            r = fix_r(r)
        for o in sg.objects:
            o = fix_o(o)
    return scene_graphs


def make_word_list(scene_graphs, N=26, K=12):
    """
    TODO - describe

    """
    O1 = set(r.subject.names[0] for sg in scene_graphs for r in sg.relationships)
    O2 = set(r.object.names[0]  for sg in scene_graphs for r in sg.relationships)
    O3 = set(o.names[0] for sg in scene_graphs for o in sg.objects)
    O = tuple(O1.union(O2).union(O3))
    diff = N - len(O)
    if diff:
        O += tuple('NULL' for _ in range(diff))

    R = tuple(set(r.predicate for sg in scene_graphs for r in sg.relationships))
    diff = K - len(R)
    if diff:
        R += tuple('NULL' for _ in range(diff))

    word_list = {'obj': O, 'rel': R}
    return word_list


def make_w2v_dict(word_list):
    n = len(word_list['obj'])
    k = len(word_list['rel'])

    # Keep track of w2v word indexes
    w2v_dict = {}
    w2v_dict['obj'] = {word:i for i, word in enumerate(word_list['obj'])}
    w2v_dict['rel'] = {word:i for i, word in enumerate(word_list['rel'])}
    return w2v_dict


def make_w2v(word_list, w2v_bin='data/GoogleNews-vectors-negative300.bin'):
    # Build matrix of w2v vectors
    import gensim.models as gm
    M = gm.Word2Vec.load_word2vec_format(w2v_bin, binary=True)
    w2v = [M[word] for word in word_list['obj'] + word_list['rel']]
    return np.vstack(w2v)


# ---------------------------------------------------------------------------------------------------------
# Other

def prune_scenes(scene_graphs, rword_fname='data/pk/rel_words.pk',
                 ofilter_fname='data/pk/obj_counts.pk', rfilter_fname='data/pk/rel_counts.pk'):
    rel_words = pickle.load(open(rword_fname,'r'))
    obj_filter = pickle.load(open(ofilter_fname,'r'))
    rel_filter = pickle.load(open(rfilter_fname,'r'))

    # rename = lambda w, d: d[w] if w in d else w.lower().strip().replace(' ','_')
    fix = lambda s: s.lower().strip().replace(' ','_')
    rename = lambda w, d: d[fix(w)] if fix(w) in d else fix(w)

    for sg in scene_graphs:
        for r in sg.relationships:
            s = rename(r.subject.names[0], [])
            v = rename(r.predicate, rel_words)
            o = rename(r.object.names[0],  [])
            if (s not in obj_filter) or (v not in rel_filter) or (o not in obj_filter):
                sg.relationships.remove(r)
            else:
                r.subject.names[0] = s
                r.predicate        = v
                r.object.names[0]  = o

        for o in sg.objects:
            o_ = rename(o.names[0], [])
            if o_ not in obj_filter:
                sg.objects.remove(o)
            else:
                o.names[0] = o_

        if len(sg.objects) == 0 or len(sg.relationships) == 0:
            scene_graphs.remove(sg)
            del sg

    import gc
    gc.collect()

    return scene_graphs


def prune_scenes(scene_graphs, rword_fname='data/pk/rel_words.pk',
                 ofilter_fname='data/pk/obj_counts.pk', rfilter_fname='data/pk/rel_counts.pk'):
    rel_words = pickle.load(open(rword_fname,'r'))
    obj_filter = pickle.load(open(ofilter_fname,'r'))
    rel_filter = pickle.load(open(rfilter_fname,'r'))

    fix = lambda s: s.lower().strip().replace(' ','_')
    rename = lambda w, d: d[fix(w)] if fix(w) in d else fix(w)

    scene_graphs_ = []
    for i, sg in enumerate(scene_graphs):
        for j, r in enumerate(sg.relationships):
            s = rename(r.subject.names[0], [])
            v = rename(r.predicate, rel_words)
            o = rename(r.object.names[0],  [])
            if (s not in obj_filter) or (v not in rel_filter) or (o not in obj_filter):
                sg.relationships[j] = None
            else:
                sg.relationships[j].subject.names[0] = s
                sg.relationships[j].predicate        = v
                sg.relationships[j].object.names[0]  = o
        sg.relationships = [r for r in sg.relationships if r is not None]

        for j, o in enumerate(sg.objects):
            o_ = rename(o.names[0], [])
            if o_ not in obj_filter:
                sg.objects[j] = None
            else:
                sg.objects[j].names[0] = o_
        sg.objects = [o for o in sg.objects if o is not None]

        if len(sg.objects) == 0 or len(sg.relationships) == 0:
            # scene_graphs.remove(sg)
            # del sg
            continue
        else:
            # scene_graphs[i] = sg
            scene_graphs_.append(sg)

    import gc
    gc.collect()

    return scene_graphs_


def prune_scene(sg, rel_words, obj_filter, rel_filter):
    fix = lambda s: s.lower().strip().replace(' ','_')
    rename = lambda w, d: d[fix(w)] if fix(w) in d else fix(w)

    for j, r in enumerate(sg.relationships):
        s = rename(r.subject.names[0], [])
        v = rename(r.predicate, rel_words)
        o = rename(r.object.names[0],  [])
        if (s not in obj_filter) or (v not in rel_filter) or (o not in obj_filter):
            sg.relationships[j] = None
        else:
            sg.relationships[j].subject.names[0] = s
            sg.relationships[j].predicate        = v
            sg.relationships[j].object.names[0]  = o
    sg.relationships = [r for r in sg.relationships if r is not None]

    for j, o in enumerate(sg.objects):
        o_ = rename(o.names[0], [])
        if o_ not in obj_filter:
            sg.objects[j] = None
        else:
            sg.objects[j].names[0] = o_
    sg.objects = [o for o in sg.objects if o is not None]

    if len(sg.objects) == 0 or len(sg.relationships) == 0:
        del sg
        return []
    else:
        return [sg]

# oc = sum(len(sg.objects) for sg in scene_graphs)
# rc = sum(len(sg.relationships) for sg in scene_graphs)
# print oc, rc

# # PRUNED OBJECTS
# In [29]: 998782 / 3319187.
# Out[29]: 0.30091163890434613

# # PRUNED RELATIONSHIPS
# In [30]: 848001 / 2032830.
# Out[30]: 0.4171529345788876

# 2320405 -> 1753128

# 1184829 -> 517156


# ---------------------------------------------------------------------------------------------------------
# Load matrix

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


# ---------------------------------------------------------------------------------------------------------
# .mat files to data

def get_data(mat_data, obj_dict, rel_dict, img_dir):
    obj_data = []
    rel_data = []

    for datum in mat_data:
        if not hasattr(datum, 'relationship'):
            continue
        img_rels = datum.relationship
        if not hasattr(img_rels, '__getitem__'):
            if not all(i in dir(img_rels) for i in ['objBox', 'phrase', 'subBox']):
                print [_ for _ in dir(img_rels) if '_' not in _]
                continue
            img_rels = [img_rels]

        img = imread(img_dir + datum.filename)
        img = cvtColor(img, COLOR_RGB2BGR)
        # print datum.filename; print img.shape
        for rel in img_rels:
            ymin1, ymax1, xmin1, xmax1 = rel.subBox
            ymin2, ymax2, xmin2, xmax2 = rel.objBox
            ymin3, ymax3, xmin3, xmax3 = (min(ymin1, ymin2), max(ymax1, ymax2),
                                          min(xmin1, xmin2), max(xmax1, xmax2))
            h1, w1 = (ymax1 - ymin1, xmax1 - xmin1)
            h2, w2 = (ymax2 - ymin2, xmax2 - xmin2)
            h3, w3 = (ymax3 - ymin3, xmax3 - xmin3)

            img1 = square_crop(img, 224, xmin1, ymin1, w1, h1)
            img2 = square_crop(img, 224, xmin2, ymin2, w2, h2)
            img3 = square_crop(img, 224, xmin3, ymin3, w3, h3)

            s,v,o = rel.phrase
            sd = np.zeros((100)); sd[obj_dict[s]] = 1
            od = np.zeros((100)); od[obj_dict[o]] = 1
            vd = np.zeros((70));  vd[rel_dict[v]] = 1
            obj_data.append((img1, sd))
            obj_data.append((img2, od))
            rel_data.append((img3, vd))

    return obj_data, rel_data


def batchify_data(data, batch_size):
    n = np.ceil(float(len(data)) / batch_size).astype(int)
    batched_data = []
    for b in range(n):
        batch_data = data[b*batch_size : (b + 1)*batch_size]
        if len(batch_data) == 0: continue
        batch_imgs, batch_labs = zip(*batch_data)
        pad_len = batch_size - len(batch_data)
        # Pad by repeating 1 image . . . probably not the best way to do this
        if pad_len > 0:
            batch_imgs += tuple(batch_imgs[0] for _ in range(pad_len))
            batch_labs += tuple(batch_labs[0] for _ in range(pad_len))
        new_imgs   = np.concatenate([i[np.newaxis, ...] for i in batch_imgs], axis=0)
        new_labels = np.concatenate([i[np.newaxis, ...] for i in batch_labs], axis=0)
        batched_data.append((new_imgs, new_labels))

    return batched_data


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

# ---------------------------------------------------------------------------------------------------------
# Scene graph to data

def id_(a, b):
    return str(a) + '_' + str(b)


def parse_scenes(scene_graphs):
    """
    Function to take scene graphs and return list of (image_id, (x,y,w,h)) pairs

    """
    scene_graphs = rel_coords(scene_graphs)
    r_id = lambda r: (r.subject.id, r.object.id)
    obj_data = {(sg.image.id, o.id, (o.x, o.y, o.width, o.height)) for sg in scene_graphs for o in sg.objects}
    ## TODO: this shouldn't be necessary, right..?
    obj_data.update({(sg.image.id, o.id, (o.x, o.y, o.width, o.height)) for sg in scene_graphs for r in sg.relationships for o in (r.subject, r.object)})
    rel_data = {(sg.image.id, r_id(r), (r.x, r.y, r.width, r.height)) for sg in scene_graphs for r in sg.relationships}
    return list(obj_data), list(rel_data)


def load_images(batch_data, mean_file, img_path, crop_size=224, batch=True):
    """
    - load and crop images, return list in same order as given
    - if fewer images than batch size, pad batch with black frames

    """
    if not batch:
        batch_data = [batch_data]

    img_ids =  set(zip(*batch_data)[0])
    imgs = {img_id:imread(img_path + str(img_id) + '.jpg') for img_id in img_ids}

    crops = []
    for img_id, uid, coord in batch_data:
        img = imgs[img_id]
        crop = square_crop(img, crop_size, *coord)
        crop -= np.load(mean_file)
        crops.append(crop)

def get_dicts(obj_data, rel_data):
    """
    Create dictionaries that give indexes for CNN outputs for object & relationship UIDs.

    """
    rid = lambda ii, oi1, oi2: frozenset([id_(ii,oi1), id_(ii,oi2)])
    obj_dict = {id_(img_id, obj_id)   : idx for idx, (img_id, obj_id, coord) in enumerate(obj_data)}
    rel_dict = {rid(img_id, *obj_ids) : idx for idx, (img_id, obj_ids, coord) in enumerate(rel_data)}
    return obj_dict, rel_dict

# def batchify_data2(data, batch_size=10):
#     pad_len = batch_size - len(batch_data)
#     if batch and (pad_len > 0):
#         pad_imgs = [np.zeros(224,224,3) for _ in range(pad_len)]
#         crops = crops + pad_imgs
#         return crops
#     else:
#         return crop
