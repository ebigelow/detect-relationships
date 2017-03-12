import numpy as np
import os, sys; sys.path.append('..')
from utils import loadmat, get_trips_mini, group_triplets, ruid2feats, ouid2ruid
from model.model import Model
from optparse import OptionParser




# --------------------------------------------------------------------------------------------------
# TODO currently we're just training/testing `model.py` on VRD data
# TODOne save run metadata
# TODOne mini word2vec



data_dir = '/home/eric/data/'
cnn_dir = '/home/eric/data/mini/models/cnn2/'

parser = OptionParser()

parser.add_option("--save_dir",   default=data_dir + "mini/models/embed/vrd-on-vg1/")

parser.add_option("--objnet_dir", default=cnn_dir + "obj_vrd1/")
parser.add_option("--relnet_dir", default=cnn_dir + "rel_vrd1/")
parser.add_option("--w2v_file",   default=data_dir + "mini/word2vec.npy")

parser.add_option("--obj_mat",    default=data_dir + "vrd/mat/objectListN.mat")
parser.add_option("--rel_mat",    default=data_dir + "vrd/mat/predicate.mat")

parser.add_option("--learning_rate", default=0.1,   type="float")
parser.add_option("--lamb1",         default=.005,    type="float")
parser.add_option("--lamb2",         default=.01,   type="float")
parser.add_option("--noise",         default=1e-10,  type="float")
parser.add_option("--K_samples",     default=500000, type="int")
parser.add_option("--max_iters",     default=20,     type="int")



# TODO vrd-on-vg2/weights_2.npy
# TODO vrd2/weights_0.npy



if __name__ == '__main__':

    (O, args) = parser.parse_args()

    # --------------------------------------------------------------------------------------------------
    # Load data

    obj_dict = {r:i for i,r in enumerate(loadmat(O.obj_mat)['objectListN'])}
    rel_dict = {r:i for i,r in enumerate(loadmat(O.rel_mat)['predicate'])}

    word2idx = {'obj':obj_dict, 'rel':rel_dict}
    w2v = np.load(O.w2v_file).item()

    # CNN features and probabilities
    obj_fn = lambda fn: O.objnet_dir + 'out/' + fn
    rel_fn = lambda fn: O.relnet_dir + 'out/' + fn
    # obj_train = { o[:2]:v['prob'] for o,v in np.load(obj_fn('vrd_train2.npy')).item().items() if o is not None }
    # obj_test  = { o[:2]:v['prob'] for o,v in np.load(obj_fn('vrd_test2.npy')).item().items()  if o is not None }
    # rel_train = { ruid2feats(r):v['fc7']  for r,v in np.load(rel_fn('vrd_train2.npy')).item().items() if r is not None }
    # rel_test  = { ruid2feats(r):v['fc7']  for r,v in np.load(rel_fn('vrd_test2.npy')).item().items()  if r is not None }

    # # VRD data (triplets)
    # D_train = np.load('/home/eric/data/mini/vrd_train2_.npy').item()['rel']
    # D_train = [((o[0][2], o[1][2], k), o[0], o[1]) for o, rc, k in list(D_train)]
    # Ds_train  = group_triplets(D_train).values()
    #
    # D_test = np.load('/home/eric/data/mini/vrd_test2_.npy').item()['rel']
    # D_test = [((o[0][2], o[1][2], k), o[0], o[1]) for o, rc, k in list(D_test)]
    # Ds_test  = group_triplets(D_test).values()
    # test_data = (Ds_test, obj_test, rel_test)


    # --------------------------------------------------------------------------------------------------
    # VG data (triplets)
    D_train = np.load('/home/eric/data/mini/vg_train.npy').item()['rel']
    D_train = [((o[0][2], o[1][2], k), o[0], o[1]) for o, rc, k in list(D_train)]

    D_test  = np.load('/home/eric/data/mini/vg_test.npy').item()['rel']
    D_test  = [((o[0][2], o[1][2], k), o[0], o[1]) for o, rc, k in list(D_test)]
    D_test  = group_triplets(D_test).values()





    # --------------------------------------------------------------------------------------------------
    # Run model

    model_dir = '/home/eric/data/mini/models/embed/vrd1/'

    M = Model(obj_probs_train, rel_feats_train, w2v, word2idx)
    M.load_weights(model_dir + 'weights_2.npy')

    save_file = model_dir + 'test/vg_train.npy'

    # Recall @ k
    topn = 20
    predicts = [(ouid2ruid(O1, O2), M.predict_preds(R, O1, O2, topn))
                for R, O1, O2 in D_train]

    hits = [int(gt[2] in preds) for gt, preds in predicts]
    accuracy = np.mean(hits)
    print 'Accuracy on VG train: {}'.format(accuracy)

    np.save(save_file, predicts)
    print 'Saved to ' + save_file
