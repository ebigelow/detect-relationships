# from skimage.transform import resize
# from skimage.io import imread
import pickle
import os
import numpy as np
import scipy.io as spio
from cv2 import imread, resize
import tensorflow as tf









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

# TODO: these are for VG, not VRD

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


# TODO this fixes words for VRD only
def get_w2v(w, M):
    removes = ['of','the', 'to']

    s = w.replace(' ','_')
    if s in M:
        return M[s]
    elif w == 'traffic light':
        return M['traffic_signalization']
    else:
        ls = [x for x in  w.split(' ') if x not in removes]
        return reduce(np.add, [M[word] for word in ls])

def make_w2v(word2idx, w2v_bin='data/GoogleNews-vectors-negative300.bin'):
    # Build matrix of w2v vectors
    import gensim.models as gm
    M = gm.Word2Vec.load_word2vec_format(w2v_bin, binary=True)
    obj_w2v = [get_w2v(w,M) for w, i in sorted(word2idx['obj'].items(), key=lambda x: x[1])]
    rel_w2v = [get_w2v(w,M) for w, i in sorted(word2idx['rel'].items(), key=lambda x: x[1])]
    return np.vstack(obj_w2v + rel_w2v)


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
            O1 = id_(img_id, rel.subject.id)
            O2 = id_(img_id, rel.object.id)
            D.append((R, O1, O2))

    return D



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


def batch_images(uid2imdata, img_dir, mean, batch_len=10, crop_size=224):

    im_path = lambda fn: os.path.join(img_dir, fn)
    uids = uid2imdata.keys()
    fnames, coords = zip(*uid2imdata.values())

    for b in xrange(0, len(uids), batch_len):

        batch_uids   = uids[b:b + batch_len]
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


def add_to_dict(key, item, D):
    if key in D:
        return D
    else:
        D[key] = item
        return D

def get_uid2imdata(scene_graphs):
    uid2coords = {}

    for sg in scene_graphs:
        get_objid = lambda x: str(sg.image.id) + '_' + str(x.id)
        get_relid = lambda y: frozenset([get_objid(y.subject), get_objid(y.object)])
        fname = sg.image.url

        for o in sg.objects:
            uid = get_objid(o)
            coords = (o.x, o.y, o.width, o.height)
            uid2coords = add_to_dict(uid, (fname, coords), uid2coords)

        for r in sg.relationships:
            uid = get_relid(r)
            coords = (r.x, r.y, r.width, r.height)
            uid2coords = add_to_dict(uid, (fname, coords), uid2coords)

    return uid2coords
