import numpy as np
from optparse import OptionParser
import sys
sys.path.append('..')

from utils import ouid2ruid, ruid2feats
from model.model import Model





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





if __name__ == '__main__':

    (O, args) = parser.parse_args()

    w2v = np.load(O.w2v_file).item()



    # --------------------------------------------------------------------------------------------------


    mini = '/home/eric/data/mini/'

    obj_dir = mini + 'models/cnn2/obj_vrd1/out/'
    rel_dir = mini + 'models/cnn2/rel_vrd1/out/'

    vrd_on_vrd = mini + 'models/embed/vrd2/'
    vrd_on_vg  = mini + 'models/embed/vrd-on-vg2/'
    weights_vrd = vrd_on_vrd + 'weights_0.npy'
    weights_vg  = vrd_on_vg  + 'weights_2.npy'


    E = (
        # (weight_file, save_file, obj_prob_file, rel_feat_file, D(data))
        (weights_vrd, vrd_on_vrd+'out/vrd_train', obj_dir+'vrd_train2.npy', rel_dir+'vrd_train2.npy', mini+'vrd_train2_.npy' ),
        (weights_vrd, vrd_on_vrd+'out/vrd_test',  obj_dir+'vrd_test2.npy',  rel_dir+'vrd_test2.npy',  mini+'vrd_test2_.npy'  ),
        (weights_vrd, vrd_on_vrd+'out/vg_train',  obj_dir+'vg_train.npy',   rel_dir+'vg_train.npy',   mini+'vg_train_.npy'   ),
        (weights_vrd, vrd_on_vrd+'out/vg_test',   obj_dir+'vg_test.npy',    rel_dir+'vg_test.npy',    mini+'vg_test_.npy'    ),
        (weights_vg,  vrd_on_vg+'out/vrd_train',  obj_dir+'vrd_train2.npy', rel_dir+'vrd_train2.npy', mini+'vrd_train2_.npy' ),
        (weights_vg,  vrd_on_vg+'out/vrd_test',   obj_dir+'vrd_test2.npy',  rel_dir+'vrd_test2.npy',  mini+'vrd_test2_.npy'  ),
        (weights_vg,  vrd_on_vg+'out/vg_train',   obj_dir+'vg_train.npy',   rel_dir+'vg_train.npy',   mini+'vg_train_.npy'   ),
        (weights_vg,  vrd_on_vg+'out/vg_test',    obj_dir+'vg_test.npy',    rel_dir+'vg_test.npy',    mini+'vg_test_.npy'    ),
    )



    # --------------------------------------------------------------------------------------------------
    # Run model

    def predict(M, D, topn=1):
        """
        Recall @ k

        output format:
            ( GT_ruid, list_of_predicted_Rs ),  . . .
        """
        predicts = [(ouid2ruid(O1, O2, R[2]), M.predict_preds(R, O1, O2, topn))
                    for R, O1, O2 in D]
        hits = [int(gt[2] in preds) for gt, preds in predicts]
        accuracy = np.mean(hits)
        return predicts, accuracy


    for weight_file, save_file, obj_file, rel_file, D_file in E:
        print 'Saving to file: ' + save_file

        obj_probs  = { o[:2]:v['prob']         for o,v in np.load(obj_file).item().items()  if o is not None }
        rel_feats  = { ruid2feats(r):v['fc7']  for r,v in np.load(rel_file).item().items()  if r is not None }

        M = Model(obj_probs, rel_feats, w2v, [])
        M.load_weights(weight_file)

        D = np.load(D_file).item()['rel']
        D = [((o[0][2], o[1][2], k), o[0], o[1]) for o, rc, k in list(D)]

        for topk in [1, 5, 10, 20]:
            # predicts, accuracy = predict(M, D, topk)
            accuracy = M.compute_accuracy2(D, topk)
            print 'Recall @ {}: {}'.format(topk, accuracy)

        # Save top 20 predictions to file
        # predicts, _ = predict(M, D, 20)
        # np.save(save_file, predicts)
