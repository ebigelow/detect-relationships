import numpy as np
import os, sys; sys.path.append('..')
from utils import loadmat, mat_to_triplets_mini, group_triplets
from model.model import Model
from optparse import OptionParser






# --------------------------------------------------------------------------------------------------
# TODO currently we're just training/testing `model.py` on VRD data

# TODO save run metadata
# TODO mini word2vec



data_dir = '/home/eric/data/'
cnn_dir = '/home/eric/data/mini/models/cnn1/'

parser = OptionParser()

parser.add_option("--save_dir",   default=data_dir + "/vrd/run1/")

parser.add_option("--objnet_dir", default=cnn_dir + "obj_vg2/")
parser.add_option("--relnet_dir", default=cnn_dir + "rel_vg1/")
parser.add_option("--w2v_file",   default=data_dir + "mini/word2vec.npy")

parser.add_option("--obj_mat",    default=data_dir + "vrd/mat/objectListN.mat")
parser.add_option("--rel_mat",    default=data_dir + "vrd/mat/predicate.mat")
parser.add_option("--train_mat",  default=data_dir + "vrd/mat/annotation_train.mat")
parser.add_option("--test_mat",   default=data_dir + "vrd/mat/annotation_test.mat")

parser.add_option("--learning_rate", default=0.05,   type="float")
parser.add_option("--lamb1",         default=.01,    type="float")
parser.add_option("--lamb2",         default=.005,   type="float")
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
    obj_fn = lambda fn: O.objnet_dir + 'out/' + fn
    rel_fn = lambda fn: O.relnet_dir + 'out/' + fn
    obj_train = { k:v['prob'] for k,v in np.load(obj_fn('vrd_train.npy')).item().items() }
    obj_test  = { k:v['prob'] for k,v in np.load(obj_fn('vrd_test.npy')).item().items()  }
    rel_train = { k:v['fc7']  for k,v in np.load(rel_fn('vrd_train.npy')).item().items() }
    rel_test  = { k:v['fc7']  for k,v in np.load(rel_fn('vrd_test.npy')).item().items()  }

    # Training data (triplets)
    mat_train = loadmat(O.train_mat)['annotation_train']
    D_train   = mat_to_triplets_mini(mat_train, word2idx)
    Ds_train  = group_triplets(D_train).values()

    mat_test = loadmat(O.test_mat)['annotation_test']
    D_test   = mat_to_triplets_mini(mat_test, word2idx)
    Ds_test  = group_triplets(D_test).values()
    test_data = (Ds_test, obj_test, rel_test)

    # --------------------------------------------------------------------------------------------------
    # Run model

    model = Model(obj_train, rel_train, D_train, w2v, word2idx,
                  learning_rate=O.learning_rate, lamb1=O.lamb1, lamb2=O.lamb2,
                  noise=O.noise, K_samples=O.K_samples, max_iters=O.max_iters)

    save = O.save_dir
    if not os.path.exists(save):
        os.makedirs(save)
        os.makedirs(save+'out/')

    # Save run metadata
    np.save(save+'meta_data.npy', O.__dict__)

    # Train model
    model.SGD(Ds_train, test_data, save_file=save+'weights.npy', ablate=[])
