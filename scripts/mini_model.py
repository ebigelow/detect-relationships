import numpy as np
import os, sys; sys.path.append('..')
from utils import loadmat, get_trips_mini, group_triplets, ruid2feats
from model.model import Model
from optparse import OptionParser




# --------------------------------------------------------------------------------------------------
# TODO currently we're just training/testing `model.py` on VRD data
# TODOne save run metadata
# TODOne mini word2vec



data_dir = '/home/eric/data/'
cnn_dir = '/home/eric/data/mini/models/cnn2/'

parser = OptionParser()

parser.add_option("--save_dir",   default=data_dir + "mini/models/embed/vrd3/")

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





if __name__ == '__main__':

    (O, args) = parser.parse_args()

    # --------------------------------------------------------------------------------------------------
    # Load data

    obj_dict = {r:i for i,r in enumerate(loadmat(O.obj_mat)['objectListN'])}
    rel_dict = {r:i for i,r in enumerate(loadmat(O.rel_mat)['predicate'])}

    word2idx = {'obj':obj_dict, 'rel':rel_dict}
    w2v = np.load(O.w2v_file).item()

    # CNN features and probabilities
    obj_fn = lambda fn: np.load(O.objnet_dir + 'out/' + fn).item()
    rel_fn = lambda fn: np.load(O.relnet_dir + 'out/' + fn).item()
    # TODO why are there Nones in this????
    # TODO remove images w one triplet (no ranking here)
    obj_train = { o[:2]:v['prob'] for o,v in obj_fn('vrd_train2.npy').items() if o is not None }
    obj_test  = { o[:2]:v['prob'] for o,v in obj_fn('vrd_test2.npy').items()  if o is not None }
    rel_train = { ruid2feats(r):v['fc7']  for r,v in rel_fn('vrd_train2.npy').items() if r is not None }
    rel_test  = { ruid2feats(r):v['fc7']  for r,v in rel_fn('vrd_test2.npy').items()  if r is not None }

    # VRD data (triplets)
    D_train = np.load('/home/eric/data/mini/vrd_train2_.npy').item()['rel']
    D_train = [((o[0][2], o[1][2], k), o[0], o[1]) for o, rc, k in list(D_train)]
    Ds_train, _ = group_triplets(D_train)

    D_test = np.load('/home/eric/data/mini/vrd_test2_.npy').item()['rel']
    D_test = [((o[0][2], o[1][2], k), o[0], o[1]) for o, rc, k in list(D_test)]
    Ds_test, _ = group_triplets(D_test)
    test_data = (Ds_test, obj_test, rel_test)


    # # --------------------------------------------------------------------------------------------------
    # # VG data (triplets)
    # D_train = np.load('/home/eric/data/mini/vg_train_.npy').item()['rel']
    # D_train = [((o[0][2], o[1][2], k), o[0], o[1]) for o, rc, k in list(D_train)]
    # Ds_train, rmv_train = group_triplets(D_train)
    #
    # D_test  = np.load('/home/eric/data/mini/vg_test_.npy').item()['rel']
    # D_test  = [((o[0][2], o[1][2], k), o[0], o[1]) for o, rc, k in list(D_test)]
    # Ds_test, rmv_test = group_triplets(D_test)
    #
    # obj_train = { o[:2]:v['prob'] for o,v in obj_fn('vg_train.npy').items() if o is not None }
    # obj_test  = { o[:2]:v['prob'] for o,v in obj_fn('vg_test.npy').items()  if o is not None }
    # rel_train = { ruid2feats(r):v['fc7']  for r,v in rel_fn('vg_train.npy').items() if r is not None }
    # rel_test  = { ruid2feats(r):v['fc7']  for r,v in rel_fn('vg_test.npy').items()  if r is not None }
    # test_data = (Ds_test, obj_test, rel_test)



    # --------------------------------------------------------------------------------------------------
    # Run model

    model = Model(obj_train, rel_train, w2v, word2idx, D_samples=D_train,
                  learning_rate=O.learning_rate, lamb1=O.lamb1, lamb2=O.lamb2,
                  noise=O.noise, K_samples=O.K_samples, max_iters=O.max_iters)

    save = O.save_dir
    if not os.path.exists(save):
        os.makedirs(save)
        os.makedirs(save+'out/')

    # Save run metadata
    np.save(save+'meta_data.npy', O.__dict__)

    # Train model
    model.SGD(Ds_train, test_data, save_file=save+'weights_{}.npy', ablate=[])
