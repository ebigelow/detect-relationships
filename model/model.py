
import numpy as np
import itertools
from numpy.random import randint, random
from scipy.spatial.distance import cosine
from tqdm import tqdm, trange
from utils import convert_coords


def cap_weight(M, cap=1.0):
    m = abs(M).max()
    return (M / m) if (m > cap) else M


def objs2reluid_vg(O1, O2):
    return frozenset([O1, O2])

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

    return (fname, frozenset([o1,o2]), coords)

def flatten(ls):
    return [i for subl in ls for i in subl]

def shuffle(ls):
    return sorted(ls, key=lambda _: random())

def tshuffle(ls):
    return tqdm(shuffle(ls))

import time
tm = lambda: time.time()


class Model:
    """
    Arguments
    ---------
    w2v : w2v = {'obj': (100, 300),  'rel': (70, 300)}
    word2idx : map words to obj/pred index; w2i['obj']['llama'] == 42

    obj_probs : obj_probs[obj_uid] = (100,) final layer cnn output
    rel_feats : rel_feats[rel_uid] = (4096,) fc7 feature (PRESUMABLY this was used . . .)

    K_samples : number of random R samples to compute for equation 4 (K)
    noise : weight initialization variance (gaussian)
    lamb1 : weight of eq 5 in eq 7
    lamb2 : weight of eq 4 in eq 7

    Attributes
    ----------

    n : number of objects
    k : number of predicates
    W,b : language model weight & bias
    Z,s : visual model weight & bias
    R_samples : list of random (R, O1, O2) samples generated for equation 4




    """
    def __init__(self, obj_probs, rel_feats, D_samples, w2v, word2idx,
                 dataset='vrd', noise=0.05, learning_rate=1.0, max_iters=20,
                 K_samples=500000, lamb1=0.05, lamb2=0.001):
        self.obj_probs     = obj_probs
        self.rel_feats     = rel_feats
        self.w2v           = w2v
        self.n = len(w2v['obj'])
        self.k = len(w2v['rel'])
        self.word2idx      = word2idx
        self.noise         = noise
        self.learning_rate = learning_rate
        self.max_iters     = max_iters
        self.K_samples   = K_samples
        self.lamb1         = lamb1
        self.lamb2         = lamb2
        self.dataset       = dataset
        self.objs2reluid   = objs2reluid_vg if dataset=='vg' else objs2reluid_vrd
        self.init_weights()
        self.init_tabular()
        self.sample_R_pairs(D_samples)

    def init_weights(self):
        nfeats = self.rel_feats.values()[0].shape[0]
        v, k = (self.noise, self.k)
        self.W = v * np.random.rand(k, 600)
        self.b = v * np.random.rand(k, 1)
        self.Z = v * np.random.rand(k, nfeats)
        self.s = v * np.random.rand(k, 1)

    def save_weights(self, filename):
        data_dict = { 'W': self.W, 'b': self.b,
                      'Z': self.Z, 's': self.s,
                      'params': { 'dataset'       : self.dataset,
                                  'n'             : self.n,
                                  'k'             : self.k,
                                  'noise'         : self.noise,
                                  'learning_rate' : self.learning_rate,
                                  'max_iters'     : self.max_iters,
                                  'K_samples'     : self.K_samples,
                                  'lamb1'         : self.lamb1,
                                  'lamb2'         : self.lamb2     }  }
        np.save(filename, data_dict)
        print 'numpy file saved to: {}'.format(filename)

    def load_weights(self, filename):
        data_dict = np.load(filename).item()
        self.W = data_dict['W']
        self.b = data_dict['b']
        self.Z = data_dict['Z']
        self.s = data_dict['s']
        self.__dict__.update(data_dict['params'])
        print 'numpy file loaded from: {}'.format(filename)


    def update(self, W=None, b=None, Z=None, s=None):
        if W is not None:
            self.W = cap_weight(W)
            self.f_dict = {}
            self.f_full_tab = None
        if b is not None:
            self.b = cap_weight(b)
            self.f_dict = {}
        if Z is not None:
            self.Z = cap_weight(Z)
            self.V_dict = {}
        if s is not None:
            self.s = cap_weight(s)
            self.V_dict = {}

    def init_tabular(self):
        self.V_dict = {}
        self.f_dict = {}
        self.f_full_tab = None
        self.f_full()
        self.init_w2v_tab()

    def init_w2v_tab(self):
        w2v, N, K = (self.w2v, range(self.n), range(self.k))

        co = lambda i1, i2: cosine(w2v['obj'][i1],   w2v['obj'][i2])
        cr = lambda k1, k2: cosine(w2v['rel'][k1],   w2v['rel'][k2])

        # TODO the bug here is what *could* have been causing the issues earlier!!!
        self.w2v_nt = np.array([[co(i1, i2) for i2 in N] for i1 in N])
        self.w2v_vt = np.array([[cr(k1, k2) for k2 in K] for k1 in K])


    # -------------------------------------------------------------------------------------------------------

    def w2v_dist(self, R1, R2):
        i1, j1, k1 = R1
        i2, j2, k2 = R2
        return self.w2v_nt[i1, i2] + self.w2v_nt[j1,j2] + self.w2v_vt[k1, k2]

    def d(self, R1, R2):
        """
        Distance between two predicate triplets.

        """
        d_rel = self.f(R1) - self.f(R2)
        d_obj = self.w2v_dist(R1, R2)
        d = (d_rel ** 2) / d_obj
        return d if (d > 0) else 1e-10

    def f(self, R):
        """
        Reduce relationship <i,j,k> to scalar language space.

        """
        i,j,k = R

        if self.f_full_tab is not None:
            return self.f_full_tab[i,j,k]

        if R not in self.f_dict:
            wvec = self.word_vec(i, j)
            f = np.dot(self.W[k].T, wvec) + self.b[k]
            self.f_dict[R] = f

        return self.f_dict[R]

    def f_full(self):
        if self.f_full_tab is None:
            w_i = self.w2v['obj'][None, ...]                            # (1, N, 300)
            w_i = np.concatenate([w_i, np.zeros_like(w_i)], axis=2)     # (1, N, 600)
            w_j = self.w2v['obj'][:, np.newaxis, :]                     # (N, 1, 300)
            w_j = np.concatenate([np.zeros_like(w_j), w_j], axis=2)     # (1, N, 600)

            n,k = (self.n, self.k)
            B = np.reshape(np.tile(self.b, n**2), (k, n, n)).T          # (1,)  ->  (N, N, K)

            tile_wv = w_i + w_j                                         # np auto-tiles `w_i`, `w_j` to (N, N, 300)
            F = np.tensordot(tile_wv, self.W, axes=(2,1)) + B           # (N, N, 600)  x  (K, 600)

            self.f_full_tab = F
            return F
        else:
            return self.f_full_tab

    def V(self, R, O1, O2, verbose=False):
        """
        Reduce relationship <i,j,k> to scalar visual space.

        """
        i,j,k = R
        rel_uid = (frozenset([O1, O2]), convert_coords(O1[1], O2[1]), k)

        P_i = self.obj_probs[O1][i]
        P_j = self.obj_probs[O2][j]

        # V_dict keeps a table of previously computed values by input
        key = rel_uid + (k,)
        if key not in self.V_dict:
            cnn = self.rel_feats[rel_uid]
            Z,s = (self.Z, self.s)
            self.V_dict[key] = np.dot(Z[k], cnn) + s[k]

        P_k = self.V_dict[key]

        if verbose: print 'V: i {}  j {}  k {}'.format(P_i, P_j, P_k)

        return P_i * P_j * P_k

    # def V_full(self, O1, O2):
    #     rel_uid = self.objs2reluid(O1, O2)
    #
    #     P_I = self.obj_probs[O1]
    #     P_J = self.obj_probs[O2]
    #
    #     cnn = self.rel_feats[rel_uid]
    #     P_K = np.dot(self.Z, cnn) + self.s.flatten()
    #
    #     P_ij  = np.outer(P_I, P_J)
    #     P_ijk = np.tensordot(P_ij, P_K, axes=0)
    #     return P_ijk

    def predict_preds(self, R, O1, O2, topn=20):
        """
        Predict predicate given object labels.

        """
        i,j,k = R
        max_fun = lambda k_: self.V((i,j,k_),O1,O2) * self.f((i,j,k_))
        preds = sorted(range(self.k), key=max_fun, reverse=True)
        return preds[:topn]

    def compute_accuracy2(self, D, topn=20):
        """
        Compute accuracy, predicting predicates only.

        """
        predictions = [(self.predict_preds(R, O1, O2, topn), R[2]) for R, O1, O2 in D]
        accuracy = np.mean([int(truth in p) for p,truth in predictions])
        return accuracy

    def predict_preds2(self, R, O1, O2, topn=20):
        """
        Predict predicate given object labels.

        """
        i,j,k = R
        P = [(-self.V((i,j,x),O1,O2) * self.f((i,j,x)), (i,j,x)) for x in range(self.k)]
        confs, preds = zip(*sorted(P))
        return preds[:topn], confs[:topn]

    def rel2str(self, R):
        i,j,k = R
        I = self.word2idx
        idx2w = {'obj': {idx:w for w,idx in I['obj'].items()},
                 'rel': {idx:w for w,idx in I['rel'].items()}}
        return '-'.join([idx2w['obj'][i], idx2w['rel'][k], idx2w['obj'][j]])


    def compute_accuracy3(self, test_data, topn=20):
        Ds, obj_probs_, rel_feats_ = test_data

        obj_probs_train, rel_feats_train = (self.obj_probs, self.rel_feats)
        self.obj_probs, self.rel_feats = (obj_probs_, rel_feats_)

        accuracy = self.compute_accuracy2(flatten(Ds), topn=topn)

        self.obj_probs, self.rel_feats = (obj_probs_train, rel_feats_train)
        return accuracy

    def predict_Rs(self, O1, O2, topn=100):
        """
        Full list of predictions `R`, sorted by confidence

        """
        N,K = (self.n, self.k)
        Rs = [(i,j,k) for i,j,k in itertools.product(range(N),range(N),range(K))]
        M = sorted(Rs, key=lambda R: -self.V(R,O1,O2) * self.f(R))
        return M[:topn]

    def predict_Rs2(self, O1, O2, topn=100):
        """
        Full list of predictions `R`, sorted by confidence

        """
        M = self.V_full(O1,O2) * self.f_full()
        predictions = np.argsort(-M.flatten())[:topn]
        R_pred = np.unravel_index(predictions, [self.n, self.n, self.k])
        return zip(*R_pred)

    def compute_accuracy(self, test_data, topn=100):
        """

        MAP
        ---
        y, y_: ground truth label for data point `i`
        X: ordered list of label predictions for data point `i`

        Recall @ k
        ----------
        is the correct label for <O1,O2> within the top k predictions?

        """
        Ds, obj_probs_, rel_feats_ = test_data

        obj_probs_train, rel_feats_train = (self.obj_probs, self.rel_feats)
        self.obj_probs, self.rel_feats = (obj_probs_, rel_feats_)

        # Mean Avg Precision
        if topn == 'map':
            recall = 0.0
            for X, Y in test_data:
                rank = lambda y: float(X.index(y) + 1)
                prec = lambda y: len([y_ for y_ in Y if rank(y_) <= rank(y)]) / rank(y)

                avg_prec = sum(prec(y) for y in Y) / len(Y)
                recall += avg_prec / len(test_data)
            return recall

        # Recall @ k
        else:
            predictions = [(R, self.predict_Rs2(O1, O2, topn)) for D in tqdm(Ds) for R, O1, O2 in tqdm(D)]
            hits = [(R in Rs) for R,Rs in predictions]
            recall = float(sum(hits)) / len(hits)

        self.obj_probs, self.rel_feats = (obj_probs_train, rel_feats_train)
        return recall

    def compute_accuracy_FINAL(self, Ds, test_data, topn=100, ntest=10000, thing=2):
        print '\ncomputing accuracy!'

        if thing == 1:
            recall = self.compute_accuracy((test_data[0][:ntest],) + test_data[1:], topn=topn)
            print '\t\ttop {} accuracy: {}'.format(topn, recall)
        else:
            for topn in [1,5,10,20]:
                if test_data is None:
                    recall = self.compute_accuracy2(flatten(Ds[-ntest:]), topn=topn)
                else:
                    recall = self.compute_accuracy3((test_data[0][:ntest],) + test_data[1:], topn=topn)
                print '\t\ttop {} accuracy: {}'.format(topn, recall)


    def update_dLdW(self, D, R, lr=1.0):
        f, W, b  = (self.f, self.W, self.b)
        w2v   = self.w2v['obj']
        i,j,k = R

        for R_2, O1_2, O2_2, _ in D:
            i_2,j_2,k_2 = R_2
            if 1 + f(R_2) - f(R) > 0:
                W[k]   -= lr * self.lamb1 * np.concatenate((w2v[i], w2v[j]))
                b[k]   -= lr * self.lamb1
                W[k_2] += lr * self.lamb1 * np.concatenate((w2v[i_2], w2v[j_2]))
                b[k_2] += lr * self.lamb1
        return W,b

    def update_dCdW(self, D, R, O1, O2, Nd, lr=1.0):
        V, f  = (self.V, self.f)
        W, b  = (self.W, self.b)
        w2v   = self.w2v['obj']
        i,j,k = R

        D_ = [(R_,O1_,O2_) for R_,O1_,O2_,_ in D if (R_ != R) or (O1_ != O1 or O2_ != O2)]
        if D_:
            R_,O1_,O2_ = min(D_, key=lambda (r_,o1_,o2_): -V(r_,o1_,o2_) * f(r_))
            i_,j_,k_ = R_

            cost = max(1. - V(R,O1,O2) * f(R) + V(R_,O1_,O2_) * f(R_), 0.)
            if cost > 0:
                W[k]  += lr * np.concatenate((w2v[i], w2v[j]))    * V(R,O1,O2)
                b[k]  += lr                                       * V(R,O1,O2)
                W[k_] -= lr * np.concatenate((w2v[i_], w2v[j_]))  * V(R_,O1_,O2_)
                b[k_] -= lr                                       * V(R_,O1_,O2_)
        return W, b

    def update_dCdZ(self, D, R, O1, O2, Nd, lr=1.0):
        obj_probs, rel_feats = (self.obj_probs, self.rel_feats)
        V, f  = (self.V, self.f)
        Z, s  = (self.Z, self.s)
        i,j,k = R

        D_ = [(R_,O1_,O2_) for R_,O1_,O2_,_ in D if (R_ != R) and (O1_ != O1 or O2_ != O2)]
        if D_:
            R_,O1_,O2_ = min(D_, key=lambda (r_,o1_,o2_): -V(r_,o1_,o2_) * f(r_))
            i_,j_,k_ = R_

            cost = max(1. - V(R,O1,O2) * f(R) + V(R_,O1_,O2_) * f(R_), 0.)
            if cost > 0:
                id_rel  = self.objs2reluid(O1, O2)
                id_rel_ = self.objs2reluid(O1_, O2_)

                Z[k]  += lr * obj_probs[O1][i]   * obj_probs[O2][j]   * rel_feats[id_rel]   * f(R)
                s[k]  += lr * obj_probs[O1][i]   * obj_probs[O2][j]                         * f(R)
                Z[k_] -= lr * obj_probs[O1_][i_] * obj_probs[O2_][j_] * rel_feats[id_rel_]  * f(R_)
                s[k_] -= lr * obj_probs[O1_][i_] * obj_probs[O2_][j_]                       * f(R_)
        return Z, s

    def sample_R_pairs(self, triplets):
        # TODO other possibility is that we sample pairs in same image only
        m = self.K_samples
        samples = np.random.randint(0, len(triplets), m*2)
        R_samples = [triplets[s][0] for s in samples]
        self.R_pairs = zip(R_samples[:m], R_samples[m:])


    def updateall_dKdW(self, lr=1.0):
        """
        TODO
        ----
        - speed this up by computing K for i,j,k tensor (700k size) then index it by
          sampled relationship counts
        - could we sample these counts directly as a tensor and do an inner product w/ K?

        """
        f, d = (self.f, self.d)
        W, b = (self.W, self.b)

        G = 0.0

        # Precompute `f_full` if we haven't yet
        if self.f_full_tab is None: self.f_full()

        print 'Computing Mean(G) for K\n'
        for R1, R2 in tqdm(self.R_pairs):
            f_ = f(R1) - f(R2)
            d_ = d(R1, R2)
            G += f_ ** 2.  / d_

        # Compute mean(G) over [1/10] initial set of samples
        Ns = self.K_samples
        G_avg = G / Ns

        print 'Computing Mean for K\n'
        # Compute dG/dW & G over second set of samples, using mean(G)
        for R1, R2 in tqdm(self.R_pairs):
            i1, j1, k1 = R1
            i2, j2, k2 = R2
            w1 = self.word_vec(i1, j1)
            w2 = self.word_vec(i2, j2)
            f_ = f(R1) - f(R2)
            d_ = d(R1, R2)

            dG1 = w1 * f_ * 2.  / d_
            dG2 = w2 * f_ * 2.  / d_
            G   = f_ ** 2.  / d_

            step = lr * self.lamb2
            W[k1] -= step * (G - G_avg) * dG1 * (2. / Ns)
            W[k2] += step * (G - G_avg) * dG2 * (2. / Ns)
        return W

    def SGD(self, Ds, test_data=None, recall_topn=1, ablate=[],
            save_file='data/models/vrd_weights.npy'):
        """
        Perform SGD over eqs 4 (K), 5 (L), 6 (C)

        TODO
        ----
        - `self.collect_updates` then `self.update`

        """
        Nd = len(flatten(Ds))
        prev_obj = 0.0

        for epoch in trange(self.max_iters):
            lr = self.learning_rate  / np.sqrt( 1 + epoch/2 )

            for D in tshuffle(Ds):
                W, b, Z, s = (self.W, self.b, self.Z, self.s)
                for R, O1, O2, _ in tshuffle(D):
                    if 'L' not in ablate:
                        dW, db = self.update_dLdW(D, R, lr)                  # dL / dW
                        W += dW; b += db
                    if 'C' not in ablate:
                        dW, db = self.update_dCdW(D, R, O1, O2, Nd, lr)      # dC / dW
                        W += dW; b += db
                        dZ, ds = self.update_dCdZ(D, R, O1, O2, Nd, lr)      # dC / dTheta
                        Z += dZ; s += ds
                self.update(W=W, b=b, Z=Z, s=s)

            if 'K' not in ablate:
                W = self.updateall_dKdW(lr)                             # dK / dW
                self.update(W=W)

            print 'W: ', (np.max(self.W), np.min(self.W))
            print 'b: ', (np.max(self.b), np.min(self.b))
            print 'Z: ', (np.max(self.Z), np.min(self.Z))
            print 's: ', (np.max(self.s), np.min(self.s))

            self.save_weights(save_file)                            # Save weights & print cost

            t1 = tm()
            C,L,K = (self.C(D), (self.lamb1 * self.L(D)), (self.lamb2 * self.K()))
            print '\tit {} | C:{}  L:{}  K:{}  \n  dCost: {} |  Time: {}'.format(epoch, C,L,K, C+L+K-prev_obj, tm()-t1)
            prev_obj = C + L + K

            self.compute_accuracy_FINAL(Ds, test_data, topn=100, ntest=10000, thing=2)
            self.compute_accuracy_FINAL(Ds, None,      topn=100, ntest=10000, thing=2)



    def word_vec(self, i, j):
        return np.concatenate((self.w2v['obj'][i], self.w2v['obj'][j]))


    # -------------------------------------------------------------------------------------------------------
    # Original Equations

    def K(self):
        """
        Eq (4): randomly sample relationship pairs and minimize variance.

        """
        # import ipdb; ipdb.set_trace()
        #R_rand = lambda: (randint(self.n), randint(self.n), randint(self.k))
        #R_samples = ((R_rand(), R_rand()) for n in range(self.K_samples))
        R_dists = [self.d(R1, R2) for R1, R2 in self.R_pairs]
        return np.var(R_dists)

    def L(self, D):
        """
        Likelihood of relationships

        """
        # import ipdb; ipdb.set_trace()
        Rs, O1s, O2s = zip(*D)
        fn = lambda R1, R2: max(self.f(R1) - self.f(R2) + 1, 0)
        return sum(fn(R1, R2) for R1 in Rs for R2 in Rs)

    def C(self, D):
        """
        Rank loss function

        """
        # import ipdb; ipdb.set_trace()
        V,f = (self.V, self.f)
        C = 0.0
        for R, O1, O2, _ in D:
            c_max = max([ V(R_,O1_,O2_) * f(R_) for R_,O1_,O2_ in D
                                                if (R_ != R) and (O1_ != O1 or O2_ != O2) ])
            c = V(R,O1,O2) * f(R)
            C += max(0,  1 - c + c_max)
        return C

    def loss(self, D):
        """
        Final objective loss function.

        """
        C = self.C(D)
        L = self.lamb1 * self.L(D)
        K = self.lamb2 * self.K()
        return C + L + K
