import pickle
import os
import numpy as np
import scipy.io as spio
from cv2 import imread, resize
import tensorflow as tf
from collections import defaultdict




# Used in VRD - train_cnn.py  and  VG
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
# VRD - Load matrix

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
# VRD - .mat files to model data

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


# ---------------------------------------------------------------------------------------------------------
# VRD - for `train_cnn.py`

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
# VRD - for `extract_cnn.py`

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

def batch_triplets(D):
    E = defaultdict(lambda: list())

    for d in D:
        fname = d[1][0]
        E[fname].append(d)

    return E.values()


# ---------------------------------------------------------------------------------------------------------
# VG - scene graphs to cnn training data

import sys
sys.path.append('/u/ebigelow/lib/visual_genome_python_driver/')
sys.path.append('/Users/eric/code/visual_genome_python_driver/')
import src.local as vg

def sg_indexes(scene_graphs, label_dict):
    for sg in scene_graphs:
        for r in sg.relationships:

            rind = None
            for y in reversed(r.synset):
                if y in label_dict['rel']:
                    rind = label_dict['rel'][y]

            if rind is None:
                print 'oh no!', r.id, r.synset
            else:
                r.index = rind
        for o in sg.objects:
            oind = None
            for y in reversed(o.synsets):
                if y in label_dict['obj']:
                    oind = label_dict['obj'][y]
            if oind is None:
                print 'oh no!', o.id, o.synsets
                sg.objects.remove(o)
            else:
                o.index = oind
    return scene_graphs

def load_sg_batcher(data_dir, data_id_dir, label_dict, img_mean,
                    start_idx=0, end_idx=-1, batch_size=10, data_epochs=20, 
                    which_net='objnet', output_size=100, img_dir='data/vg/images/'):

    #n, k = (len(label_dict['obj']), len(label_dict['rel']))
    N = end_idx - start_idx
    batch_len = np.ceil(float(N) / data_epochs).astype(int)

    for e in range(data_epochs):
        #if e % (data_epochs / 20) == 0: print 'data batch: {}'.format(e)
        scene_graphs = vg.GetSceneGraphs(e, e+batch_len, data_dir, data_id_dir)
        scene_graphs = rel_coords(scene_graphs)
        #scene_graphs = sg_indexes(scene_graphs, label_dict)
        obj_meta, rel_meta = get_sg_data(scene_graphs, img_dir, label_dict)

        if which_net == 'objnet':
            yield batchify_sg_data(obj_meta, img_mean, batch_size, img_dir, output_size)
        else:
            yield batchify_sg_data(rel_meta, img_mean, batch_size, img_dir, output_size)

def get_sg_data(scene_graphs, img_dir, label_dict):
    obj_data = []
    rel_data = []

    for sg in scene_graphs:
        fname = sg.image.url.split('/')[-1]

        for r in sg.relationships:
            s, o = (r.subject, r.object)

            i = [label_dict['obj'][sy] for sy in s.synsets if sy in label_dict['obj']][0]
            j = [label_dict['obj'][sy] for sy in o.synsets if sy in label_dict['obj']][0]
            k = [label_dict['rel'][sy] for sy in r.synset  if sy in label_dict['rel']][0]

            s_coords = (s.x, s.y, s.width, s.height)
            o_coords = (o.x, o.y, o.width, o.height)
            r_coords = (r.x, r.y, r.width, r.height)

            s_uid = ( fname,  s,  s_coords )
            o_uid = ( fname,  o,  o_coords )
            r_uid = frozenset((s_uid, o_uid))

            obj_data.append((s_uid, i,  s_coords))
            obj_data.append((o_uid, j,  o_coords))
            rel_data.append((r_uid, k,  r_coords))

    return list(obj_data), list(rel_data)

def one_hot(idx, shape):
    v = np.zeros(shape)
    v[idx] = 1
    return v

def batchify_sg_data(data, mean, batch_size, img_dir, output_size=100):
    n = np.ceil(float(len(data)) / batch_size).astype(int)
    batched_data = []

    for b in range(n):
        batch_data = data[b*batch_size : (b + 1)*batch_size]
        if len(batch_data) == 0: continue

        batch_uids, batch_labs, batch_coords = zip(*batch_data)
        batch_imgs = []

        for uid, label, coords in batch_data:
            fname = uid[0] if (type(uid) != frozenset) else list(uid)[0]
            img = imread(img_dir + fname)
            x, y, w, h = coords
            crop = square_crop(img, 224, x, y, w, h) - mean
            batch_imgs.append(crop)

        # Pad by repeating zeros
        pad_len = batch_size - len(batch_data)
        if pad_len > 0:
            batch_imgs += tuple(np.zeros_like(batch_imgs[0]) for _ in range(pad_len))
            batch_labs += tuple(np.zeros_like(batch_labs[0]) for _ in range(pad_len))

        padded_imgs   = np.concatenate([i[np.newaxis, ...] for i in batch_imgs], axis=0)
        padded_labels = np.vstack([ one_hot(i, output_size) for i in batch_labs ])
        batched_data.append((batch_uids, padded_imgs, padded_labels))

    return batched_data





# ---------------------------------------------------------------------------------------------------------
# VG - scene graphs to model training data

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

