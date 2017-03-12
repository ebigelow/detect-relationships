import os
import numpy as np
import scipy.io as spio
from scipy.ndimage import imread
from scipy.misc import imresize
from collections import defaultdict
from tqdm import trange, tqdm


def flatten(ls):
    return [i for sl in ls for i in sl]


def ruid2feats(ruid):
    obs, rc, k = ruid
    # new_obs = frozenset([(o[0], o[1]) for o in obs])
    new_obs = (obs[0][:2], obs[1][:2])
    return new_obs


# ---------------------------------------------------------------------------------------------------------
# Image processing

def square_crop(img, crop_size, x, y, w, h, reshape=False):
    """
    Crop square image at (y,x) with dims (h,w), resize to crop_size.

    reshape (bool) :  do we squash (reshape) the original bounding box crop image,
                      or keep its proportions and crop to a square shape?

    """
    # ih, iw, _ = img.shape
    # if w == h:
    #     x1 = x
    #     x2 = x + w
    #     y1 = y
    #     y2 = y + h
    # elif w > h:
    #     d1 = np.floor((w - h) / 2.)
    #     d2 = np.ceil((w - h) / 2.)
    #     if y - d1 < 0:
    #         y1 = 0
    #         y2 = y + h + d2 - (y - d1)
    #     elif (y + h + d2) > (ih - 1):
    #         y1 = y - d1 - ((y + h + d2) - (ih - 1))
    #         y2 = ih - 1
    #     else:
    #         y1 = y - d1
    #         y2 = y + h + d2
    #     x1 = x
    #     x2 = x + w
    # elif h > w:
    #     d1 = np.floor((h - w) / 2.)
    #     d2 = np.ceil((h - w) / 2.)
    #     if x - d1 < 0:
    #         x1 = 0
    #         x2 = x + w + d2 - (x - d1)
    #     elif (x + w + d2) > (iw - 1):
    #         x1 = x - d1 - ((x + w + d2) - (iw - 1))
    #         x2 = iw - 1
    #     else:
    #         x1 = x - d1
    #         x2 = x + w + d2
    #     y1 = y
    #     y2 = y + h
    # f = lambda c: int(max(c, 0))
    # if reshape:
    #     crop = img
    # else:
    #     crop = img[f(y1):f(y2), f(x1):f(x2)]
    crop = img[y:y+h, x:x+w]
    new_img = imresize(crop, (crop_size, crop_size))
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

def convert_coords(sc, oc):
    sx, sy, sw, sh = sc
    ox, oy, ow, oh = oc
    rx = min(sx, ox)
    ry = min(sy, oy)
    rw = max(sx + sw, ox + ow) - rx
    rh = max(sy + sh, oy + oh) - ry
    return (rx, ry, rw, rh)


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



def get_data_mini(mat_data, obj_dict, rel_dict, img_dir,
                  mean_file='mean.npy', mini_file='/home/eric/data/mini/vrd2mini.npy'):
    obj_data = []
    rel_data = []

    vrd2mini = np.load(mini_file).item()
    n_mini = max(vrd2mini['obj'].values()) + 1  # 96
    k_mini = max(vrd2mini['rel'].values()) + 1  # 40
    # import ipdb; ipdb.set_trace()

    for datum in mat_data:
        fn = datum.filename

        # Skip images with no relationship data present -- there are some
        if not hasattr(datum, 'relationship'):
            #print 'skipping image {}, no relationship'.format(img_dir + datum.filename)
            continue

        img_rels = datum.relationship
        if not hasattr(img_rels, '__getitem__'):
            if not all(i in dir(img_rels) for i in ['objBox', 'phrase', 'subBox']):
                print 'skipping relation, dir contains:', [_ for _ in dir(img_rels) if '_' not in _]
                continue
            img_rels = [img_rels]

        # Swap RGB channels to BGR
        rgb = imread(img_dir + datum.filename)
        bgr = rgb[:,:,::-1]

        for rel in img_rels:
            # ('subject', 'verb', 'object')
            s,v,o = rel.phrase
            i,j,k = (obj_dict[s], obj_dict[o], rel_dict[v])

            if (i in vrd2mini['obj']) and (j in vrd2mini['obj']) and (k in vrd2mini['rel']):
                i_ = vrd2mini['obj'][i]
                j_ = vrd2mini['obj'][j]
                k_ = vrd2mini['rel'][k]

                # Get unique id (uid) for subject-verb-object triple
                sc = box_to_coords(*rel.subBox)
                oc = box_to_coords(*rel.objBox)
                rc = convert_coords(sc, oc)

                s_uid = (fn, sc, i_)
                o_uid = (fn, oc, j_)
                r_uid = (frozenset([s_uid, o_uid]), rc, k_)

                # UID format: (img_filename, label_word, bbox_coordinates)
                s_img = square_crop(bgr, 224, *sc) - np.load(mean_file)
                o_img = square_crop(bgr, 224, *oc) - np.load(mean_file)
                r_img = square_crop(bgr, 224, *rc) - np.load(mean_file)

                # One-hot label vectors
                s_label = np.zeros((n_mini));  s_label[i_] = 1
                o_label = np.zeros((n_mini));  o_label[j_] = 1
                v_label = np.zeros((k_mini));  v_label[k_] = 1

                # Data format: (img, label, uid)
                obj_data.append((s_img, s_label, s_uid))
                obj_data.append((o_img, o_label, o_uid))
                rel_data.append((r_img, v_label, r_uid))

    return {'obj': list(obj_data), 'rel': list(rel_data)}


def get_trips_mini(mat_data, obj_dict, rel_dict,
                   mini_file='/home/eric/data/mini/vrd2mini.npy'):
    data = []
    vrd2mini = np.load(mini_file).item()

    for datum in mat_data:
        if not hasattr(datum, 'relationship'):
            continue

        img_rels = datum.relationship
        if not hasattr(img_rels, '__getitem__'):
            if not all(i in dir(img_rels) for i in ['objBox', 'phrase', 'subBox']):
                print 'skipping relation, dir contains:', [_ for _ in dir(img_rels) if '_' not in _]
                continue
            img_rels = [img_rels]

        for rel in img_rels:
            s,v,o = rel.phrase
            i,j,k = (obj_dict[s], obj_dict[o], rel_dict[v])
            fn = datum.filename

            if (i in vrd2mini['obj']) and (j in vrd2mini['obj']) and (k in vrd2mini['rel']):
                i_ = vrd2mini['obj'][i]
                j_ = vrd2mini['obj'][j]
                k_ = vrd2mini['rel'][k]

                sc = box_to_coords(*rel.subBox)
                oc = box_to_coords(*rel.objBox)
                rc = convert_coords(sc, oc)

                s_uid = (fn, sc, i_)
                o_uid = (fn, oc, j_)
                r_uid = (frozenset([s_uid, o_uid]), rc, k_)

                R = (i_, j_, k_)
                data.append((R, s_uid, o_uid))

    return data



def get_data(mat_data, obj_dict, rel_dict, img_dir, mean_file='mean.npy'):
    obj_data = []
    rel_data = []

    for datum in mat_data:
        # Skip images with no relationship data present -- there are some
        if not hasattr(datum, 'relationship'):
            #print 'skipping image {}, no relationship'.format(img_dir + datum.filename)
            continue

        # TODO is this bit necessary?
        img_rels = datum.relationship
        if not hasattr(img_rels, '__getitem__'):
            if not all(i in dir(img_rels) for i in ['objBox', 'phrase', 'subBox']):
                print 'skipping relation, dir contains:', [_ for _ in dir(img_rels) if '_' not in _]
                continue
            img_rels = [img_rels]

        # Swap RGB channels to BGR
        rgb = imread(img_dir + datum.filename)
        bgr = rgb[:,:,::-1]

        for rel in img_rels:
            # ('subject', 'verb', 'object')
            s,v,o = rel.phrase

            # Get unique id (uid) for subject-verb-object triple
            s_uid = (datum.filename, s, box_to_coords(*rel.subBox))
            o_uid = (datum.filename, o, box_to_coords(*rel.objBox))
            r_uid = objs2reluid_vrd(s_uid, o_uid)

            # UID format: (img_filename, label_word, bbox_coordinates)
            s_img = square_crop(bgr, 224, *s_uid[2]) - np.load(mean_file)
            o_img = square_crop(bgr, 224, *o_uid[2]) - np.load(mean_file)
            r_img = square_crop(bgr, 224, *r_uid[2]) - np.load(mean_file)

            # One-hot label vectors
            s_label = np.zeros((100));  s_label[obj_dict[s]] = 1
            o_label = np.zeros((100));  o_label[obj_dict[o]] = 1
            v_label = np.zeros((70));   v_label[rel_dict[v]] = 1

            # Data format: (img, label, uid)
            obj_data.append((s_img, s_label, s_uid))
            obj_data.append((o_img, o_label, o_uid))
            rel_data.append((r_img, v_label, r_uid))

    return {'obj': list(obj_data), 'rel': list(rel_data)}


def load_vrd_batcher(mat_path, obj_list_path, rel_list_path,
                     batch_size=10, data_epochs=20, which_net='obj',
                     img_dir='data/vrd/images/train/', mean_file='mean.npy'):
    """
    In this case, we load images in get_data, then we batch & tensorify in batchify_data

    """
    obj_dict = {r:i for i,r in enumerate(loadmat(obj_list_path)['objectListN'])}
    rel_dict = {r:i for i,r in enumerate(loadmat(rel_list_path)['predicate'])}

    mat = loadmat(mat_path)[mat_path.split('/')[-1].split('.')[0]]

    batch_len = np.ceil(float(len(mat)) / data_epochs).astype(int)
    for e in trange(data_epochs):
        mat_batch = mat[e*batch_len : (e+1)*batch_len]
        # obj_test, rel_test = get_data(a_test, obj_dict, rel_dict, 'data/vrd/images/test/')
        vrd_data = get_data_mini(mat_batch, obj_dict, rel_dict, img_dir, mean_file)

        yield batchify_data2(vrd_data[which_net], batch_size)

def batchify_data2(data, batch_size):
    """
    Split a list of images and labels into batches, then
      merge into tensors with first dimension `batch_size`.

    Add padding if not enough images/labels in last batch.
    """
    n = np.ceil(float(len(data)) / batch_size).astype(int)
    batched_data = []
    for b in range(n):
        batch_data = data[b*batch_size : (b + 1)*batch_size]
        if len(batch_data) == 0: continue
        batch_imgs, batch_labs, batch_uids = zip(*batch_data)

        # Pad by repeating 1 image . . . probably not the best way to do this
        pad_len = batch_size - len(batch_data)
        if pad_len > 0:
            batch_imgs += tuple(batch_imgs[0] for _ in range(pad_len))
            batch_labs += tuple(batch_labs[0] for _ in range(pad_len))

        new_imgs   = np.concatenate([i[np.newaxis, ...] for i in batch_imgs], axis=0)
        new_labels = np.concatenate([i[np.newaxis, ...] for i in batch_labs], axis=0)
        new_uids   = batch_uids +  (None,) * pad_len
        batched_data.append((new_imgs, new_labels, new_uids))

    return batched_data


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
    y, x = (ymin, xmin)
    coords = (x, y, w, h)

    return (frozenset([o1,o2]), coords)


# ---------------------------------------------------------------------------------------------------------
# vrd_model.py

def mat_to_triplets(mat_data, word2idx):
    """Convert matrix data to triplets."""
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
            sub_id = (datum.filename, box_to_coords(*r.subBox), s)
            obj_id = (datum.filename, box_to_coords(*r.objBox), o)

            i = word2idx['obj'][s]
            j = word2idx['obj'][o]
            k = word2idx['rel'][v]      # different uid format here
            R = (i,j,k)
            D.append((R, sub_id, obj_id))
    return D

def group_triplets(D):
    """Group triplets according to their source image."""
    E = defaultdict(lambda: list())
    for d in D:
        fname = d[1][0]
        E[fname].append(d)

    E = {k:v for k,v in E.items()}
    for fn in E.keys():
        if len(E[fn]) < 2:
            del E[fn]
            print 'removed ' + fn
    return E



def mat_to_triplets_mini(mat_data, word2idx,
                         mini_file='/home/eric/data/mini/vrd2mini.npy'):
    """Convert matrix data to triplets."""
    D = []

    for datum in mat_data:
        fname = datum.filename

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

            V = np.load(mini_file).item()
            vo = V['obj']
            vr = V['rel']

            if all((i in vo, j in vo, k in vr)):
                i_, j_, k_ = (vo[i], vo[j], vr[k])
                R = (i_,j_,k_)

                sc = box_to_coords(*r.subBox)
                oc = box_to_coords(*r.objBox)
                sub_uid = (fname, sc, i)
                obj_uid = (fname, oc, j)
                rel_uid = (fname, convert_coords(sc, oc), k)

                D.append((R, sub_uid, obj_uid))
    return D


# ---------------------------------------------------------------------------------------------------------
# VG - scene graphs to cnn training data

import sys
sys.path.append('/home/eric/lib/visual_genome_python_driver_eric/')
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
                    which_net='obj', output_size=100, img_dir='data/vg/images/'):

    batch_len = np.ceil(108000.0 / data_epochs).astype(int)
    for e in trange(data_epochs):

        scene_graphs = vg.GetSceneGraphs(e, e+batch_len, data_dir, data_id_dir)
        scene_graphs = rel_coords(scene_graphs)
        #scene_graphs = sg_indexes(scene_graphs, label_dict)

        sg_data = get_sg_data_mini(scene_graphs, label_dict)

        yield batchify_data(sg_data[which_net], img_mean, batch_size, img_dir, output_size)


def tenumerate(ls):
    return enumerate(tqdm(ls))


def get_sg_data(scene_graphs, label_dict):
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

            s_uid = (fname,  s_coords, i)
            o_uid = (fname,  o_coords, j)
            r_uid = ((s_uid, o_uid), r_coords, k)

            obj_data.append(s_uid)
            obj_data.append(o_uid)
            rel_data.append(r_uid)

    return {'obj': list(obj_data), 'rel': list(rel_data)}



def get_sg_data_mini(scene_graphs, label_dict, mini_file='/home/eric/data/mini/vrd2mini.npy'):

    obj_data = []
    rel_data = []

    vrd2mini = np.load(mini_file).item()
    vo = vrd2mini['obj']
    vr = vrd2mini['rel']

    for sg in scene_graphs:
        fname = sg.image.url.split('/')[-1]

        for r in sg.relationships:
            s, o = (r.subject, r.object)

            i = [label_dict['obj'][sy] for sy in s.synsets if sy in label_dict['obj']][0]
            j = [label_dict['obj'][sy] for sy in o.synsets if sy in label_dict['obj']][0]
            k = [label_dict['rel'][sy] for sy in r.synset  if sy in label_dict['rel']][0]

            if (i in vo) and (j in vo) and (k in vr):
                s_coords = (s.x, s.y, s.width, s.height)
                o_coords = (o.x, o.y, o.width, o.height)
                r_coords = (r.x, r.y, r.width, r.height)

                s_uid = (fname,          s_coords, vo[i])
                o_uid = (fname,          o_coords, vo[j])
                r_uid = ((s_uid, o_uid), r_coords, vr[k])

                obj_data.append(s_uid)
                obj_data.append(o_uid)
                rel_data.append(r_uid)

    return {'obj': list(obj_data), 'rel': list(rel_data)}



def one_hot(idx, shape):
    v = np.zeros(shape)
    v[idx] = 1
    return v

def load_mini_batcher(data, img_mean, batch_size=10, data_epochs=20, output_size=100,
                      img_dir='~/data/mini/vrd/images/train/'):
    batch_len = np.ceil(len(data) / data_epochs).astype(int)

    for e in trange(data_epochs):
        data_batch = data[batch_len * e : batch_len * (e+1)]
        yield batchify_data(data_batch, img_mean, batch_size, img_dir, output_size)

def batchify_data(data, mean, batch_size, img_dir,
                  output_size=100, reshape_imgs=False):
    """
    Load images and concat them into batch_size -sized tensors.

    In:
        [  ( fname/reluid, coords, label ), . . .  ]
    Out:
        [  ( image_tensor, label_tensor, list_of_uids ), . . .  ]
    """
    n = np.ceil(float(len(data)) / batch_size).astype(int)
    batched_data = []

    for b in range(n):
        batch_uids = data[b*batch_size : (b + 1)*batch_size]
        if len(batch_uids) == 0: continue

        batch_images = []
        _, batch_coords, batch_labels = zip(*batch_uids)

        for fname, coords, label in batch_uids:
            if type(fname) not in [str, unicode]:
                fname = list(fname)[0][0]

            rgb = imread(img_dir + fname)
            if len(rgb.shape) == 2:
                rgb = np.tile(rgb[..., None], [1,1,3])
            bgr = rgb[:,:,::-1]

            # Crop image to coordinates for this data
            x, y, w, h = coords
            rs = reshape_imgs
            crop = square_crop(bgr, 224, x, y, w, h, reshape=rs) - mean

            batch_images.append(crop)

        # Pad by repeating zeros
        pad_len = batch_size - len(batch_uids)
        if pad_len > 0:
            batch_images += tuple(np.zeros_like(batch_images[0]) for _ in range(pad_len))
            batch_labels += tuple(np.zeros_like(batch_labels[0]) for _ in range(pad_len))

        padded_images = np.concatenate([ i[np.newaxis, ...] for i in batch_images ], axis=0)
        padded_labels = np.vstack([ one_hot(i, output_size) for i in batch_labels ])
        batched_data.append((padded_images, padded_labels, batch_uids))

    return batched_data
