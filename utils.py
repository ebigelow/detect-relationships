# from skimage.transform import resize
# from skimage.io import imread
import pickle
import os
import numpy as np
import scipy.io as spio
from cv2 import imread, resize
import tensorflow as tf
from collections import defaultdict


# ---------------------------------------------------------------------------------------------------------
# Image processing

def square_crop(img, crop_size, x, y, w, h):
    """
    Crop square image at (y,x) with dims (h,w), resize to crop_size.

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

# Visual Genome Only
# ------------------




def fix_r(r):
    if r.predicate == 'of':
        r.predicate = 'NULL'
    if r.predicate == 'on the':
        r.predicate = 'on'
    if r.predicate == 'front of':
        r.predicate = 'front'
    r.subject = fix_o(r.subject)
    r.object = fix_o(r.object)
    return r

# def fix_o(o):
#
#     removes = ['of','the', 'to']
#
#     if o.names[0] == 'windw':
#         o.names[0] = 'window'
#     if o.names[0] == 'sky light':
#         o.names[0] = 'skylight'
#
#
#
#     s = w.replace(' ','_')
#     if s in M:
#         return M[s]
#     elif w == 'traffic light':
#         return M['traffic_signalization']
#
#     cond w:
#     {'street lamp': 'streetlight',
#      'white clouds': 'clouds',
#      'a':  , # TODO: DELETE --> also delete other preds!
#      'street light': 'streetlight'}
#
#
#     return o

def fix_names(scene_graphs):
    for sg in scene_graphs:
        for r in sg.relationships:
            r = fix_r(r)
        for o in sg.objects:
            o = fix_o(o)
    return scene_graphs


def make_word_list(scene_graphs, N=100, K=70):
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


def make_word2idx(word_list):
    n = len(word_list['obj'])
    k = len(word_list['rel'])

    # Keep track of w2v word indexes
    word2idx = {}
    word2idx['obj'] = {word:i for i, word in enumerate(word_list['obj'])}
    word2idx['rel'] = {word:i for i, word in enumerate(word_list['rel'])}
    return word2idx



def convert_rel(rel, word2idx):
    """
    Converting training data `Relationship` instance to `<i,j,k>` format.

    """
    i = word2idx['obj'][rel.subject.names[0]]
    j = word2idx['obj'][rel.object.names[0]]
    k = word2idx['rel'][rel.predicate]
    return i,j,k



def sg_to_triplets(scene_graphs, word2idx):
    """
    Load a list of data triplets, one for each scene graph: (R, O1, O2)

    """
    D = []
    for sg in scene_graphs:
        img_id = sg.image.id
        for rel in sg.relationships:
            R = convert_rel(rel, word2idx)
            O1 = rel.subject.id
            O2 = rel.object.id
            D.append((R, O1, O2))

    return D





# Visual Genome Only
# ------------------


def save_w2v(word2idx, w2v_bin='data/word2vec/GoogleNews-vectors-negative300.bin'):
    w2v = make_w2v(word2idx, w2v_bin)
    np.save(w2v_file, w2v)

def make_w2v(word2idx, w2v_bin='data/GoogleNews-vectors-negative300.bin'):
    # Build matrix of w2v vectors
    import gensim.models as gm
    M = gm.Word2Vec.load_word2vec_format(w2v_bin, binary=True)
    obj_w2v = [get_w2v(w,M) for w, i in sorted(word2idx['obj'].items(), key=lambda x: x[1])]
    rel_w2v = [get_w2v(w,M) for w, i in sorted(word2idx['rel'].items(), key=lambda x: x[1])]
    return np.vstack(obj_w2v + rel_w2v)




# ---------------------------------------------------------------------------------------------------------
# Scene graph stuff

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
# scene graphs to cnn training data

def sg_indexes(scene_graphs, label_dict):
    for sg in scene_graphs:
        for r in sg.relationships:
            for y in reversed(r.synset):
                if y in label_dict['rel']:
                    r.index = label_dict['rel'][y]
        for o in sg.objects:
            for y in reversed(o.synsets):
                if y in label_dict['obj']:
                    o.index = label_dict['obj'][y]
    return scene_graphs

def load_sg_batcher(data_dir, data_id_dir, label_dict, start_idx=0, end_idx=-1,
                    batch_size=10, data_epochs=20, which_net='objnet',
                    img_dir='data/vg/images/', img_mean=None):
    import sys
    sys.path.append('/Users/eric/code/visual_genome_python_driver/')
    import src.local as vg

    n, k = (len(label_dict['obj']), len(label_dict['rel']))
    N = end_idx - start_idx
    batch_len = np.ceil(float(N) / data_epochs).astype(int)

    for e in range(data_epochs):
        scene_graphs = vg.GetSceneGraphs(start_idx, end_idx, data_dir, data_id_dir)
        scene_graphs = rel_coords(scene_graphs)
        scene_graphs = sg_indexes(scene_graphs, label_dict)
        obj_meta, rel_meta = get_sg_data(scene_graphs, img_mean, img_dir, n, k)

        if which_net == 'objnet':
            yield batchify_data(obj_meta, batch_size)
        else:
            yield batchify_data(rel_meta, batch_size)

def get_sg_data(scene_graphs, img_mean, img_dir, n_objs=600, n_rels=100):
    obj_data = []
    rel_data = []

    for sg in scene_graphs:
        img = imread(img_dir + sg.image.url)

        for r in sg.relationships:
            s, o = (r.subject, r.object)
            img_s = square_crop(img, 224, s.x, s.y, s.width, s.height) - img_mean
            img_o = square_crop(img, 224, o.x, o.y, o.width, o.height) - img_mean
            img_r = square_crop(img, 224, r.x, r.y, r.width, r.height) - img_mean

            sd = np.zeros((n_objs));  sd[s.index] = 1
            od = np.zeros((n_objs));  od[o.index] = 1
            rd = np.zeros((n_rels));  rd[r.index] = 1

            obj_data.append((img_s, sd))
            obj_data.append((img_o, od))
            rel_data.append((img_r, rd))

    return list(obj_data), list(rel_data)





# ---------------------------------------------------------------------------------------------------------
# .mat files to data


def get_data(mat_data, obj_dict, rel_dict, img_dir, mean_file='mean.npy'):
    obj_data = []
    rel_data = []

    for datum in mat_data:
        if not hasattr(datum, 'relationship'):
            #print 'skipping image {}, no relationship'.format(img_dir + datum.filename)
            continue
        img_rels = datum.relationship
        if not hasattr(img_rels, '__getitem__'):
            if not all(i in dir(img_rels) for i in ['objBox', 'phrase', 'subBox']):
                print 'skipping relation, dir contains:', [_ for _ in dir(img_rels) if '_' not in _]
                continue
            img_rels = [img_rels]

        img = imread(img_dir + datum.filename)
        # print img_dir+datum.filename; print img.shape
        for rel in img_rels:
            ymin1, ymax1, xmin1, xmax1 = rel.subBox
            ymin2, ymax2, xmin2, xmax2 = rel.objBox
            ymin3, ymax3, xmin3, xmax3 = (min(ymin1, ymin2), max(ymax1, ymax2),
                                          min(xmin1, xmin2), max(xmax1, xmax2))
            h1, w1 = (ymax1 - ymin1, xmax1 - xmin1)
            h2, w2 = (ymax2 - ymin2, xmax2 - xmin2)
            h3, w3 = (ymax3 - ymin3, xmax3 - xmin3)

            img1 = square_crop(img, 224, xmin1, ymin1, w1, h1) - np.load(mean_file)
            img2 = square_crop(img, 224, xmin2, ymin2, w2, h2) - np.load(mean_file)
            img3 = square_crop(img, 224, xmin3, ymin3, w3, h3) - np.load(mean_file)

            s,v,o = rel.phrase
            sd = np.zeros((100)); sd[obj_dict[s]] = 1
            od = np.zeros((100)); od[obj_dict[o]] = 1
            vd = np.zeros((70));  vd[rel_dict[v]] = 1
            obj_data.append((img1, sd))
            obj_data.append((img2, od))
            rel_data.append((img3, vd))

    return list(obj_data), list(rel_data)


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


# ---------------------------------------------------------------------------------------------------------
# Scene graph to data

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




# ---------------------------------------------------------------------------------------------------------
# For `train_cnn.py`


def load_data_batcher(mat_path, obj_list_path, rel_list_path,
                      batch_size=10, data_epochs=20, which_net='objnet',
                      img_dir='data/vrd/images/train/', mean_file='mean.npy'):
    obj_dict = {r:i for i,r in enumerate(loadmat(obj_list_path)['objectListN'])}
    rel_dict = {r:i for i,r in enumerate(loadmat(rel_list_path)['predicate'])}

    mat = loadmat(mat_path)[mat_path.split('/')[-1].split('.')[0]]

    batch_len = np.ceil(float(len(mat)) / data_epochs).astype(int)
    for e in range(data_epochs):
        meta_batch_data = mat[e*batch_len : (e+1)*batch_len]
        # obj_test, rel_test = get_data(a_test, obj_dict, rel_dict, 'data/vrd/images/test/')
        obj_meta, rel_meta = get_data(meta_batch_data, obj_dict, rel_dict, img_dir)

        if which_net == 'objnet':
            yield batchify_data(obj_meta, batch_size)
        else:
            yield batchify_data(rel_meta, batch_size)





def test_cnn(net, ground_truth, N_test=1000, which_net='objnet',
             obj_list_path='data/vrd/objectListN.mat', rel_list_path='data/vrd/predicate.mat',
             mat_path='data/vrd/annotation_test.mat', images_dir='data/vrd/images/test/'):
    images_var = tf.placeholder('float', [num_imgs, 224, 224, 3])
    ground_truth = tf.placeholder(tf.float32, shape=[num_imgs, net.prob.get_shape()[1]])
    accuracy = net.get_accuracy(ground_truth)

    d_test = load_data_batcher(mat_path, obj_list_path, rel_list_path, N_test, 1, images_dir, which_net)
    images, labels = d_test.next()[:N_test]

    with session_init() as sess:
        tf.initialize_all_variables().run()
        feed_dict = {ground_truth: labels,
                     images_var: images}
        summary, acc = sess.run(accuracy, feed_dict=feed_dict)
        print('Accuracy at step %s: %s' % (i, acc))




# ---------------------------------------------------------------------------------------------------------
# For `extract_cnn.py`


def batch_mats(imdata, img_dir, mean, batch_len=10, crop_size=224):

    im_path = lambda fn: os.path.join(img_dir, fn)
    fnames, labels, coords = zip(*imdata)

    for b in xrange(0, len(imdata), batch_len):

        batch_uids   = imdata[b:b + batch_len]
        batch_coords = coords[b:b + batch_len]
        batch_fnames = fnames[b:b + batch_len]

        batch_imgs   = [imread(im_path(fn)) - mean for fn in batch_fnames]
        batch_imdata = zip(batch_imgs, batch_coords)

        batch_crops  = [square_crop(img, crop_size, *co)[None, ...] for img, co in batch_imdata]
        batch_crops  = np.concatenate(batch_crops, axis=0)
        batch_crops -= mean

        idx2uid = {idx:uid for idx, uid in enumerate(batch_uids)}

        # TODO: make sure idxs refer to original imgs, not padding
        pad_len = batch_len - len(batch_uids)
        if pad_len > 0:
            pad_imgs = np.zeros((pad_len, crop_size, crop_size, 3))
            batch_crops = np.concatenate([batch_crops, pad_imgs], axis=0)

        yield idx2uid, batch_crops




# def batch_scene_graphs(sg_imdata, img_dir, mean, batch_len=10, crop_size=224):
#
#     im_path = lambda fn: os.path.join(img_dir, fn)
#     fnames, uids, coords = zip(*sg_imdata)
#
#     for b in xrange(0, len(scene_graphs), batch_len):
#
#         batch_fnames = fnames[b:b + batch_len]
#         batch_uids   = uids[b:b + batch_len]
#         batch_coords = coords[b:b + batch_len]
#
#         batch_imgs   = [imread(im_path(fn)) - mean for fn in batch_fnames]
#         batch_imdata = zip(batch_imgs, batch_coords)
#
#         batch_crops  = [square_crop(img, crop_size, *co)[None, ...] for img, co in batch_imdata]
#         batch_crops  = np.concatenate(batch_crops, axis=0)
#         batch_crops -= mean
#
#         idx2uid = {idx:uid for idx, uid in enumerate(batch_uids)}
#
#         # TODO: make sure idxs refer to original imgs, not padding
#         pad_len = batch_len - len(batch_uids)
#         if pad_len > 0:
#             pad_imgs = np.zeros((pad_len, crop_size, crop_size, 3))
#             batch_crops = np.concatenate([batch_crops, pad_imgs], axis=0)
#
#         yield idx2uid, batch_crops
#
#
#

#
# def sg_to_imdata(scene_graphs):
#     """
#     Use filename, phrase, & coords as UID, since there are no object/rel ids.
#
#     """
#     obj_data = set()
#     rel_data = set()
#
#     for sg in scene_graphs:
#         for r in sg.relationships:
#             s,o = (r.subject.id, r.object.id)
#             O1 = (datum.filename, s, box_to_coords(*r.subBox))
#             O2 = (datum.filename, o, box_to_coords(*r.objBox))
#             reluid = objs2reluid_vrd(O1, O2)
#
#             reluid = frozenset([O1, O2])
#
#             obj_data.update({O1, O2})
#             rel_data.add(reluid)
#
#     return list(obj_data), list(rel_data)

# def tf_mat_batcher(data_dir, data_id_dir, start_idx, end_idx,
#                    label_dict, batch_size, data_epochs,
#                    which_net, img_dir, img_mean):

def mat_to_tf(mat, word2idx, obj_probs, rel_feats,
              n_=100, k_=70, batch_size=34):
    """
    """
    D_imgs = defaultdict(lambda: list())
    for datum in mat:
        if not hasattr(datum, 'relationship'):
            continue
        img_rels = datum.relationship
        if not hasattr(img_rels, '__getitem__'):
            if not all(i in dir(img_rels) for i in ['objBox', 'phrase', 'subBox']):
                print 'skipping relation, dir contains:', [_ for _ in dir(img_rels) if '_' not in _]
                continue
            img_rels = [img_rels]

        for r in img_rels:
            s,v,o  = r.phrase
            i = word2idx['obj'][s]
            j = word2idx['obj'][o]
            k = word2idx['rel'][v]

            sub_id = (datum.filename, s, box_to_coords(*r.subBox))
            obj_id = (datum.filename, o, box_to_coords(*r.objBox))
            rel_id = objs2reluid_vrd(sub_id, obj_id)
            sub_prob = obj_probs[sub_id]
            obj_prob = obj_probs[obj_id]
            rel_feat = rel_feats[rel_id]

            D_imgs[datum.filename].append([i, j, k, sub_prob, obj_prob, rel_feat])

    for fname in D_imgs:
        I, J, K, s_, o_, r_ = [np.vstack(x) for x in zip(*D_imgs[fname])]

        # For padding, we add an extra dimension to obj_probs with zeros
        #  (this will also ensure we never predict this class at test time)
        n_rels = I.shape[0]
        s_ = np.concatenate([s_, np.zeros((n_rels, 1))], axis=1)
        o_ = np.concatenate([o_, np.zeros((n_rels, 1))], axis=1)
        D_imgs[fname] = (I, J, K, s_, o_, r_)

        # TODO this will break if batch_size < n_rels
        pad_len = batch_size - n_rels
        if pad_len > 0:
            pad_q = lambda a,q: np.concatenate([a, q * np.ones((pad_len, a.shape[1]))], axis=0)
            pad_z = lambda a:   np.concatenate([a,    np.zeros((pad_len, a.shape[1]))], axis=0)
            D_imgs[fname] = [pad_q(I, n_), pad_q(J, n_), pad_q(K, k_),
                             pad_z(s_),    pad_z(o_),   pad_z(r_)]

    I, J, K, subs_, objs_, rels_ = zip(*D_imgs.values())
    objs_final = [np.concatenate([s_[None, ...], o_[None, ...]], axis=0) for s_,o_ in zip(subs_, objs_)]
    return I, J, K, objs_final, rels_


def mat_to_imdata(mat):
    """
    Use filename, phrase, & coords as UID, since there are no object/rel ids.

    """
    obj_data = set()
    rel_data = set()

    for datum in mat:
        if not hasattr(datum, 'relationship'):
            continue
        img_rels = datum.relationship
        if not hasattr(img_rels, '__getitem__'):
            if not all(i in dir(img_rels) for i in ['objBox', 'phrase', 'subBox']):
                continue
            img_rels = [img_rels]

        for r in img_rels:
            s,v,o = r.phrase
            O1 = (datum.filename, s, box_to_coords(*r.subBox))
            O2 = (datum.filename, o, box_to_coords(*r.objBox))
            reluid = objs2reluid_vrd(O1, O2)

            obj_data.update({O1, O2})
            rel_data.add(reluid)

    return list(obj_data), list(rel_data)


def box_to_coords(ymin, ymax, xmin, xmax):
    return xmin, ymin, (xmax-xmin), (ymax-ymin)

def objs2reluid_vrd(O1, O2):
    fname, o1, coords1 = O1
    fname, o2, coords2 = O2

    x1, y1, w1, h1 = coords1
    x2, y2, w2, h2 = coords2
    ymin, ymax, xmin, xmax = (min(y1, y2), max(y1+h1, y2+h2),
                              min(x1, x2), max(x1+w1, x2+w2))
    h, w = (ymax-ymin, xmax-xmin)
    y, x = (ymin    , xmin)
    coords = (x, y, w, h)

    return (fname, frozenset([o1,o2]), coords)


def mat_to_triplets(mat_data, word2idx):
    D = []

    for datum in mat_data:
        if not hasattr(datum, 'relationship'):
            continue
        img_rels = datum.relationship
        if not hasattr(img_rels, '__getitem__'):
            if not all(i in dir(img_rels) for i in ['objBox', 'phrase', 'subBox']):
                print 'skipping relation, dir contains:', [_ for _ in dir(img_rels) if '_' not in _]
                continue
            img_rels = [img_rels]

        for r in img_rels:
            s,v,o  = r.phrase
            sub_id = (datum.filename, s, box_to_coords(*r.subBox))
            obj_id = (datum.filename, o, box_to_coords(*r.objBox))

            i = word2idx['obj'][s]
            j = word2idx['obj'][o]
            k = word2idx['rel'][v]
            R = (i,j,k)
            D.append((R, sub_id, obj_id))

    return D



from collections import defaultdict

def batch_triplets(D):
    E = defaultdict(lambda: list())

    for d in D:
        fname = d[1][0]
        E[fname].append(d)

    return E.values()
