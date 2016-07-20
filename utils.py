from numpy import floor, ceil
from cv2 import resize
import gensim.models as gm

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
        d1 = floor((w - h) / 2.)
        d2 = ceil((w - h) / 2.)

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
        d1 = floor((h - w) / 2.)
        d2 = ceil((h - w) / 2.)

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

    crop = img[y1:y2, x1:x2]
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


def make_w2v(word_list, w2v_bin='../data/GoogleNews-vectors-negative300.bin'):
    # Build matrix of w2v vectors
    M = gm.Word2Vec.load_word2vec_format(w2v_bin, binary=True)
    w2v = [M[word] for word in word_list['obj'] + word_list['rel']]
    return np.vstack(w2v)

