
# from utils import *
import numpy as np
from numpy.random import randint
from scipy.spatial.distance import cosine
from tqdm import tqdm, trange
from utils import objs_to_reluid

class Model:
    """
    Arguments
    ---------
    w2v : w2v = (100 + 70, 300)  numpy vector, 1 for each object (100), then 1 for each predicate (70)
    word2idx : map words to obj/pred index; w2i['obj']['llama'] == 42

    ----
    TODO - turn w2v into a dictionary, map uid to w2v
    ----

    obj_probs : obj_probs[obj_uid] = (100,) final layer cnn output
    rel_feats : rel_feats[rel_uid] = (4096,) fc7 feature (PRESUMABLY this was used . . .)

    num_samples : number of random R samples to compute for equation 4
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


    TODO
    ----
    - change all object uids to `imageid_objectid`
    - should we only compute `R_samples` as needed? or for each call to K?



    """
    def __init__(self, obj_probs, rel_feats, w2v, word2idx,
                 noise=0.05, learning_rate=1.0, max_iters=20,
                 num_samples=500000, lamb1=0.05, lamb2=0.001):
        self.obj_probs     = obj_probs
        self.rel_feats     = rel_feats
        self.w2v           = w2v
        self.word2idx      = word2idx
        self.noise         = noise
        self.learning_rate = learning_rate
        self.max_iters     = max_iters
        self.num_samples   = num_samples
        self.lamb1         = lamb1
        self.lamb2         = lamb2

        self.n = obj_probs.values()[0].shape[0]
        self.k = w2v.shape[0] - self.n

        self.V_dict = {}
        self.f_dict = {}

        self.init_weights()
        self.init_R_samples()


    def init_weights(self):
        nfeats = self.rel_feats.values()[0].shape[0]
        v, k = (self.noise, self.k)
        self.W = v * np.random.rand(k, 600)
        self.b = v * np.random.rand(k, 1)
        self.Z = v * np.random.rand(k, nfeats)
        self.s = v * np.random.rand(k, 1)

    def save_weights(self, filename):
        data_dict = { 'W' : self.W,
                      'b' : self.b,
                      'Z' : self.Z,
                      's' : self.s  }
        np.save(filename, data_dict)
        print 'numpy file saved to: {}'.format(filename)

    def load_weights(self, filename):
        data_dict = np.load(filename)
        self.W = data_dict['W']
        self.b = data_dict['b']
        self.Z = data_dict['Z']
        self.s = data_dict['s']
        print 'numpy file loaded from: {}'.format(filename)


    def init_R_samples(self):
        """
        Draw a number of random (i,j,k) indices: 2 objects and 1 relationship

        """
        N, K, S = (self.n, self.k, self.num_samples)
        R_rand = lambda: (randint(N), randint(N), randint(K))
        R_pairs = [(R_rand(), R_rand()) for n in range(S)]
        self.R_samples = R_pairs

    def update(self, W=None, b=None, Z=None, s=None):
        if W is not None:
            self.W = W
            self.f_dict = {}
        if b is not None:
            self.b = b
            self.f_dict = {}
        if Z is not None:
            self.Z = Z
            self.V_dict = {}
        if s is not None:
            self.s = s
            self.V_dict = {}

    # -------------------------------------------------------------------------------------------------------

    def w2v_dist(self, R1, R2):
        N, w2v = (self.n, self.w2v)
        return cosine(w2v[R1[0]],   w2v[R2[0]]) +  \
               cosine(w2v[R1[1]],   w2v[R2[1]]) +  \
               cosine(w2v[N+R1[2]], w2v[N+R2[2]])

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
        if R not in self.f_dict:
            i,j,k = R
            W,b = (self.W, self.b)
            wvec = self.word_vec(i, j)
            self.f_dict[R] = np.dot(W[k].T, wvec) + b[k]

        return self.f_dict[R]

    def V(self, R, O1, O2):
        """
        Reduce relationship <i,j,k> to scalar visual space.

        """
        i,j,k = R
        rel_uid = objs_to_reluid(O1, O2)


        P_i = self.obj_probs[O1][i]
        P_j = self.obj_probs[O2][j]

        key = rel_uid+(k,)
        if key not in self.V_dict:
            cnn = self.rel_feats[rel_uid]
            Z,s = (self.Z, self.s)
            self.V_dict[key] = np.dot(Z[k], cnn) + s[k]

        P_k = self.V_dict[key]

        return P_i * P_j * P_k


    def predict_preds(self, R, O1, O2, topn=20):
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
        

    def compute_accuracy2(self, D, topn=20):
        """
        Compute accuracy, predicting predicates only.

        """
        predictions = [(self.predict_preds(R, O1, O2, topn), R) for R, O1, O2 in D]

        # TODO: new code
        if topn == 20:
            for (p, c), truth in predictions:
                print 'GT: ' + self.rel2str(truth)
                for p_, c_ in zip(p,c): print '\tR_:{} | conf: {}'.format(self.rel2str(p_),c_) 

        accuracy = np.mean([int(truth in p) for (p,c),(_,_,truth) in predictions])
        return accuracy

    def predict_Rs2(self, O1, O2, topn=100):
        """
        Full list of predictions `R`, sorted by confidence

        """
        N,K = (self.n, self.k)
        Rs = [(i,j,k) for i in range(N) for j in range(N) for k in range(K)]
        M = [(R, self.V(R,O1,O2) * self.f(R)) for R in Rs]
        M = sorted(M, key=lambda x: -x[1])
        return M[:topn]

    def predict_Rs(self, O1, O2, topn=100):
        """
        Full list of predictions `R`, sorted by confidence

        """
        N,K = (self.n, self.k)
        Rs = [(i,j,k) for i in range(N) for j in range(N) for k in range(K)]
        M = sorted(Rs, key=lambda R: -self.V(R,O1,O2) * self.f(R))
        return M[:topn]

    def compute_accuracy(self, Ds, topn=100):
        """

        MAP
        ---
        y, y_: ground truth label for data point `i`
        X: ordered list of label predictions for data point `i`

        Recall @ k
        ----------
        is the correct label for <O1,O2> within the top k predictions?

        """
        # MAP
        if topn == 'map':
            mean_ap = 0.0
            for X, Y in test_data:
                rank = lambda y: float(X.index(y) + 1)
                prec = lambda y: len([y_ for y_ in Y if rank(y_) <= rank(y)]) / rank(y)

                avg_prec = sum(prec(y) for y in Y) / len(Y)
                mean_ap += avg_prec / len(test_data)
            return mean_ap
        # Recall @ k
        else:
            predictions = [(R, self.predict_Rs(O1, O2, topn)) for D in Ds for R, O1, O2 in D]
            hits = [(R in Rs) for R,Rs in predictions]
            recall = hits / len(hits)
            if topn==20:
                for R, Rs in predictions:
                    print '~' * 30, '\nR: {}'.format(R)
                    for R_ in Rs: print '\t{}'
            return recall
            #hits = sum(float(R in self.predict_Rs(O1, O2, topn)) for D in Ds for R, O1, O2 in D)
            #recall = hits / len(hits)
            #return recall

    def SGD(self, Ds, save_file='data/models/vrd_weights.npy', recall_topn=1):
        """
        Perform SGD over eqs 5 (L) 6 (C)

        """
        obj_probs, rel_feats, w2v = (self.obj_probs, self.rel_feats, self.w2v)
        V, f, d = (self.V, self.f, self.d)
        W, b, Z, s = (self.W, self.b, self.Z, self.s)
        cost_prev = 0.0

        flatten = lambda ls: [i for subl in ls for i in subl]
        Df = flatten(Ds)

        #Ds = [Df[:1000]]

        for epoch in range(self.max_iters):

            # Use to get change in cost (mc = mean cost)
            mc = 0.0

            for D in tqdm(Ds):

                # Iterate over data points (stochastically)
                for R, O1, O2 in tqdm(D):


                    i,j,k = R

                    # Even epochs --> update W,b
                    if epoch % 2 == 0:
                        # Equation 5
                        # TODO: this is the only working section...
                        lr = self.learning_rate * self.lamb1
                        for R_2, O1_2, O2_2 in D:
                            i_2,j_2,k_2 = R_2
                            if 1 + f(R_2) - f(R) > 0:
                                W[k]   -= lr * np.concatenate((w2v[i], w2v[j]))
                                b[k]   -= lr
                                W[k_2] += lr * np.concatenate((w2v[i_2], w2v[j_2]))
                                b[k_2] += lr
                                self.update(W=W, b=b)

                        # Equation 6
                        # Get (R, O1, O2) that maximizes term in equation 6
                        D_ = [(R_,O1_,O2_) for R_,O1_,O2_ in D if (R_ != R) and (O1_ != O1 or O2_ != O2)]
                        R_,O1_,O2_ = min(D_, key=lambda (R,O1,O2): -V(R,O1,O2) * f(R)) if D_ else (R,O1,O2)
                        i_,j_,k_ = R_

                        # Compute value for ` max{cost, 0} ` in equation 6
                        cost = max(1. - V(R,O1,O2) * f(R) + V(R_,O1_,O2_) * f(R_), 0.)
                        mc  += cost / len(Df)
                        if cost > 0:
                            lr = self.learning_rate
                            W[k]  -= self.learning_rate * np.concatenate((w2v[i], w2v[j]))
                            b[k]  -= self.learning_rate
                            W[k_] += self.learning_rate * np.concatenate((w2v[i_], w2v[j_]))
                            b[k_] += self.learning_rate
                            #v = V(R,O1,O2)
                            #W[k]  -= self.learning_rate * v * np.concatenate((w2v[i], w2v[j]))
                            #b[k]  -= self.learning_rate * v
                            #v_ = V(R_,O1_,O2_)
                            #W[k_] += self.learning_rate * v_ * np.concatenate((w2v[i_], w2v[j_]))
                            #b[k_] += self.learning_rate * v_
                            self.update(W=W, b=b)


                    # Odd epochs --> update Z,s
                    else:
                        # Get (R, O1, O2) that maximizes term in equation 6
                        D_ = [(R_,O1_,O2_) for R_,O1_,O2_ in D if (R_ != R) and (O1_ != O1 or O2_ != O2)]
                        R_,O1_,O2_ = min(D_, key=lambda (R,O1,O2): -V(R,O1,O2) * f(R)) if D_ else (R,O1,O2)
                        i_,j_,k_ = R_

                        # Compute value for ` max{cost, 0} ` in equation 6
                        cost = max(1. - V(R,O1,O2) * f(R) + V(R_,O1_,O2_) * f(R_), 0.)
                        mc  += cost / len(Df)

                        # Equation 6
                        if cost > 0:
                            id_rel  = objs_to_reluid(O1, O2)
                            id_rel_ = objs_to_reluid(O1_, O2_)
                            lr = self.learning_rate

                            Z[k]  -= lr * obj_probs[O1][i]   * obj_probs[O2][j]   * rel_feats[id_rel]
                            s[k]  -= lr * obj_probs[O1][i]   * obj_probs[O2][j]
                            Z[k_] += lr * obj_probs[O1_][i_] * obj_probs[O2_][j_] * rel_feats[id_rel_]
                            s[k_] += lr * obj_probs[O1_][i_] * obj_probs[O2_][j_]
                            #Z[k]  -= lr * f(R)  * obj_probs[O1][i]   * obj_probs[O2][j]   * rel_feats[id_rel]
                            #s[k]  -= lr * f(R)  * obj_probs[O1][i]   * obj_probs[O2][j]
                            #Z[k_] += lr * f(R_) * obj_probs[O1_][i_] * obj_probs[O2_][j_] * rel_feats[id_rel_]
                            #s[k_] += lr * f(R_) * obj_probs[O1_][i_] * obj_probs[O2_][j_]
                            self.update(Z=Z, s=s)

                # # Equation 4  (W,b)
                # if epoch % 2 == 0:
                #     # Only update weights when f(R')-f(R)+1 > 0
                #     dK_dR = lambda R1, R2, v: v if f(R1) - f(R2) + 1 > 0 else 0
                #     dKfun = sum( dK_dR(R,R_, (2. / d(R,R_)) * (f(R) - f(R_)) *
                #                              (self.word_vec(*R[:-1]) - self.word_vec(*R_[:-1])) )
                #                  for R,R_ in self.R_samples )
                #
                #     Kfun = lambda R, R_: (f(R) - f(R_))**2 / d(R,R_)
                #     Ksum = sum(Kfun(R,R_) for R, R_ in self.R_samples)
                #     nr = self.num_samples
                #     dK_dW = ((2.0 - nr) / nr) * Ksum * dKfun
                #
                #     # TODO: separate `W[k] += ...` and `W[k_] -= ...`  ???
                #     print 'dK_dW shape: {}'.format(dK_dW.shape)
                #     # TODO: if (1 - v + v' > 0) : update
                #     W += self.learning_rate * dK_dW * self.lamb2
                #     self.update(W=W, b=b)

            self.save_weights(save_file)

            #Lv = self.lamb1 * self.L(D)
            #Kv = self.lamb2 * self.K()
            #final_obj = np.array([mc, Lv, Kv])
            final_obj = mc
            print '\tit {} | change in cost: {}'.format(epoch, final_obj - cost_prev)
            cost_prev = final_obj

            for q in [1, 5, 10, 20]:
                recall = self.compute_accuracy2(flatten(Ds[-50:]), topn=q)
                print '\ttop {} accuracy: {}'.format(q, recall)



    # -------------------------------------------------------------------------------------------------------
    # Helper Methods

    def word_vec(self, i, j):
        return np.concatenate((self.w2v[i], self.w2v[j]))



    # -------------------------------------------------------------------------------------------------------
    # Original Equations

    def K(self):
        """  UNUSED
        Eq (4): randomly sample relationship pairs and minimize variance.

        """
        R_dists = [self.d(R1, R2) for R1, R2 in self.R_samples]
        return np.var(R_dists)

    def L(self, D):
        """  UNUSED
        Likelihood of relationships

        """
        Rs, O1s, O2s = zip(*D)
        fn = lambda R1, R2: max(self.f(R1) - self.f(R2) + 1, 0)
        return sum(fn(R1, R2) for R1 in Rs for R2 in Rs)

    # def C(self, D):
    #     """  UNUSED
    #     Rank loss function
    #
    #     TODO -- wrong
    #
    #     """
    #     V,f = (self.V, self.f)
    #     C = 0.0
    #     for R, O1, O2 in D:
    #         c = max(V(*d2) * f(*d2[3:]) for d2 in img_data if  \
    #                 (d2[5] != d2[5]) and ((d1[3] != d2[3]) or (d1[4] != d2[4])))
    #         C += max(1 + V(*d1) * f(*d1[3:]) * c, 0)
    #     return C

    def loss(self, dir):
        """  UNUSED
        Final objective loss function.

        """
        C = self.C(D)
        L = self.lamb1 * self.L(D)
        K = self.lamb2 * self.K()
        return C + L + K
