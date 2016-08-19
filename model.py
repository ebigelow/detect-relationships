
import sys, os
from cv2 import imread
from utils import *
import numpy as np
import tensorflow as tf
from numpy.random import randint
from scipy.spatial.distance import cosine
import pickle
sys.path.append('/u/ebigelow/lib/caffe-tensorflow')


class Model:
    """
    Arguments
    ---------
    w2v : w2v = (100 + 70, 300)  numpy vector, 1 for each word, then 1 for each predicate
    obj_probs : obj_probs[rel_index] = (100,) final layer cnn output
    rel_feats : rel_feats[rel_index] = (4096,) fc7 feature (PRESUMABLY this was used . . .)
    obj_dict : match image-specific object id to index in obj_probs
    rel_dict : match image-specific relationship id to index in rel_feats

    init_type : TODO
    num_samples : number of random R samples to compute for equation 4

    noise : initialization noise
    lamb1 : weight of eq 5 in eq 7
    lamb2 : weight of eq 4 in eq 7


    TODO
    ----
    - should we only compute `R_samples` as needed? or for each call to K?


    """
    def __init__(self, obj_probs, rel_feats, obj_dict, rel_dict,
                 w2v, w2v_dict, n,
                 init_type='TODO', noise=0.05, learning_rate=1.0, max_iters=20,
                 num_samples=500000, lamb1=0.05, lamb2=0.001):
        self.obj_probs     = obj_probs
        self.rel_feats     = rel_feats
        self.obj_dict      = obj_dict
        self.rel_dict      = rel_dict
        self.w2v           = w2v
        self.w2v_dict      = w2v_dict
        self.init_type     = init_type
        self.noise         = noise
        self.learning_rate = learning_rate
        self.max_iters     = max_iters
        self.num_samples   = num_samples
        self.lamb1         = lamb1
        self.lamb2         = lamb2

        self.n = n
        self.k = w2v.shape[0] - n
        self.init_weights()
        self.init_R_samples()

    def init_weights(self):
        ndim = self.rel_feats.shape[1]
        ndim, v, k = (self.noise, self.k)
        self.W = v * np.random.rand(k, 600)
        self.b = v * np.random.rand(k, 1)
        self.Z = v * np.random.rand(k, ndim)
        self.s = v * np.random.rand(k, 1)


    def init_R_samples(self):
        """
        Draw a number of random (i,j,k) indices: 2 objects and 1 relationship

        """
        N, K, S = (self.n, self.k, self.num_samples)
        R_rand = lambda: (randint(N), randint(N), randint(K))
        R_pairs = [(R_rand(), R_rand()) for n in range(S)]
        self.R_samples = R_pairs

    def convert_rel(self, rel):
        """
        Converting training data `Relationship` instance to `<i,j,k>` format.

        """
        i = self.w2v_dict['obj'][rel.subject.names[0]]
        j = self.w2v_dict['obj'][rel.object.names[0]]
        k = self.w2v_dict['rel'][rel.predicate]
        return i,j,k

    def word_vec(self, i, j):
        return np.concatenate((self.w2v[i], self.w2v[j]))


    def w2v_dist(self, R1, R2):
        N, w2v = (self.n, self.w2v)
        return cosine(w2v[R1[0]],   w2v[R2[0]]) +  \
               cosine(w2v[R1[1]],   w2v[R2[1]]) +  \
               cosine(w2v[N+R1[2]], w2v[N+R2[2]])

    def d(self, R1, R2, W, b):
        """
        Distance between two predicate triplets.

        """
        d_rel = self.f(R1, W, b) - self.f(R2, W, b)
        d_obj = self.w2v_dist(R1, R2)
        d = (d_rel ** 2) / d_obj
        return d if (d > 0) else 1e-10


    def f(self, R, W, b):
        """
        Project relationship `R = <i,j,k>` to scalar space.

        """
        i,j,k = R
        wvec = self.word_vec(i, j)
        return np.dot(W[k].T, wvec) + b[k]

    def V(self, R, O1, O2, Z, s):
        i,j,k = R

        O1_id  = self.obj_dict[O1]
        O2_id  = self.obj_dict[O2]
        rel_id = self.rel_dict[frozenset([O1, O2])]

        P_i = self.obj_probs[O1_id, i]
        P_j = self.obj_probs[O2_id, j]
        cnn = self.rel_feats[rel_id]
        P_k = np.dot(Z[k], cnn) + s[k]
        return P_i * P_j * P_k


    def predict_Rs(self, O1, O2):
        """
        Full list of predictions `R`, sorted by confidence

        """
        N,K,W,b = (self.n, self.k, self.W, self.b)
        Rs = [(i,j,k) for i in range(N) for j in range(N) for k in range(K)]
        M = sorted(Rs, key=lambda R: -V(R,O1,O2,W,b) * f(R,W,b))
        return M

    def compute_accuracy(self, GT, k=None):
        """
        TODO
        ----
        input should be list of (R, O1, O2) for each relationship in each image



        - X: list of predictions `x` for each data point, sorted by confidence
        - Ys: list of ground truth labels `y` for each data point
        - each data point should be an `(i,j,k)` tuple

        MAP
        ---
        y, y_: ground truth label for data point `i`
        X: ordered list of label predictions for data point `i`

        Recall @ k
        ----------
        is the correct label within the top k predictions?

        """
        test_data = [[(predict_Rs(O1, O2), R) for R, O1, O2 in datum] for datum in GT]

        # MAP
        if k is None:
            mean_ap = 0.0
            for X, Y in test_data:
                rank = lambda y: float(X.index(y) + 1)
                prec = lambda y: len([y_ for y_ in Y if rank(y_) <= rank(y)]) / rank(y)

                avg_prec = sum(prec(y) for y in Y) / len(Y)
                mean_ap += avg_prec / len(test_data)
            return mean_ap
        # Recall @ k
        else:
            recall = 0.0
            for X, Y in test_data:
                z = float(len(Y) * len(test_data))
                recall += sum((y in X[:k]) for y in Y) / z
            return recall

    def SGD(self, D):
        """
        Perform SGD over eqs 5 (L) 6 (C)

        """
        obj_probs, rel_feats, w2v = (self.obj_probs, self.rel_feats, self.w2v)
        V, f, d = (self.V, self.f, self.d)
        W, b, Z, s = (self.W, self.b, self.Z, self.s)

        for i in range(self.max_iters):
            if i % 20 == 0:
                print 'SGD iteration [{}]'.format(i)

            D = sorted(D, key=lambda x: np.random.rand())
            mc = 0.0
            mc_prev = 1.0

            for R, O1, O2 in D:
                # Get max item
                D_ = [(R_,O1_,O2_) for R_,O1_,O2_ in D if (R_ != R) and (O1_ != O1 or O2_ != O2)]
                M = sorted(D_, key=lambda (R,O1,O2): V(R,O1,O2,Z,s) * f(R,W,b))
                R_,O1_,O2_ = M[0]
                i,j,k = R
                i_,j_,k_ = R_

                # Compute value for ` max{cost, 0} ` in equation 6
                cost = 1 - V(R,O1,O2,Z,s) * f(R,W,b) + V(R_,O1_,O2_,Z,s) * f(R_,W,b)
                mc  += cost / len(D)


                # Update W,b
                if i % 2 == 0:
                    # Equation 5
                    for R_2, O1_2, O2_2 in D:
                        i_2,j_2,k_2 = R_2
                        if f(R_2, W, b) - f(R, W, b) > 0:
                            W[k]   -= self.learning_rate * np.concatenate((w2v[i],  w2v[j])) * self.lamb1
                            b[k]   -= self.learning_rate * self.lamb1
                            W[k_2] += self.learning_rate * np.concatenate((w2v[i_2], w2v[j_2])) * self.lamb1
                            b[k_2] += self.learning_rate * self.lamb1

                    # Equation 6
                    if cost > 0:
                        v = V(R,O1,O2,Z,s)
                        W[k]  -= self.learning_rate * v * np.concatenate((w2v[i], w2v[j]))
                        b[k]  -= self.learning_rate * v
                        v_ = V(R_,O1_,O2_,Z,s)
                        W[k_] += self.learning_rate * v_ * np.concatenate((w2v[i_], w2v[j_]))
                        b[k_] += self.learning_rate * v_


                # Update Z,s
                else:
                    # Equation 6
                    if cost > 0:
                        id_o1  = self.obj_dict[O1]
                        id_o2  = self.obj_dict[O2]
                        id_o1_ = self.obj_dict[O1_]
                        id_o2_ = self.obj_dict[O2_]
                        id_r   = self.rel_dict[frozenset([O1, O2])]
                        id_r_  = self.rel_dict[frozenset([O1_, O2_])]

                        Z[k]  -= self.learning_rate * f(R,W,b) * obj_probs[id_o1,i] * obj_probs[id_o2,j] * rel_feats[id_r]
                        s[k]  -= self.learning_rate * f(R,W,b) * obj_probs[id_o1,i] * obj_probs[id_o2,j]
                        Z[k_] += self.learning_rate * f(R_,W,b)  * obj_probs[id_o1_,i_] * obj_probs[id_o2_,j_] * rel_feats[id_r_]
                        s[k_] += self.learning_rate * f(R_,W,b)  * obj_probs[id_o1_,i_] * obj_probs[id_o2_,j_]

            # Equation 4
            if i % 2 == 0:
                print 'PART 3'
                dKfun = sum(    (2. / d(R,R_,W,b)) *
                                (f(R,W,b) - f(R_,W,b)) *
                                (self.word_vec(*R[:-1]) - self.word_vec(*R_[:-1]))
                            for R,R_ in self.R_samples )

                Kfun = lambda R, R_: (f(R,W,b) - f(R_,W,b))**2 / d(R,R_,W,b)
                Ksum = sum(Kfun(R,R_) for R, R_ in self.R_samples)
                nr = self.num_samples
                dK_dW = ((2.0 - nr) / nr) * Ksum * dKfun
                print '\tKsum', Ksum
                print '\tdKfun', dKfun

                # TODO: separate `W[k] += ...` and `W[k_] -= ...`
                W += self.learning_rate * dK_dW

            print '\tchange in cost: {}'.format(mc_prev - mc)
            mc_prev = mc

        self.W, self.b, self.Z, self.s = (W, b, Z, s)
        return W, b, Z, s


    def load_data(self, scene_graphs):
        """
        Load a list of data triplets, one for each scene graph: (R, O1, O2)

        """
        obj_names = set([o.names[0] for sg in scene_graphs for o in sg.objects])
        rel_names = set([r.predicate for sg in scene_graphs for r in sg.relationships])
        obj_w2id  = {n:i for i,n in enumerate(obj_names)}
        rel_w2id  = {n:i for i,n in enumerate(rel_names)}

        D = []
        for sg in scene_graphs:
            img_id = sg.image.id
            for rel in sg.relationships:
                R = self.convert_rel(rel)
                O1 = id_(img_id, rel.subject.id)
                O2 = id_(img_id, rel.object.id)
                D.append((R, O1, O2))

        self.obj_names, self.rel_names = (obj_names, rel_names)
        self.obj_w2id,  self.rel_w2id  = (obj_w2id,  rel_w2id)
        return D


    # -------------------------------------------------------------------------------------------------------
    # UNUSED METHODS

    def K(self):
        """  UNUSED
        Eq (4): randomly sample relationship pairs and minimize variance.

        """
        D = []
        for R1, R2 in self.R_samples:
            d = dist(R1, R2)
            D.append(d)
        return np.var(D)

    def L(self, D):
        """  UNUSED
        Likelihood of relationships

        """
        Rs, O1s, O2s = zip(*D)
        Rs = zip(Is, Js, Ks)
        fn = lambda R1, R2: max(self.f(*R1) - self.f(*R2) + 1, 0)
        return sum(fn(R1, R2) for R1 in Rs for R2 in Rs)

    def C(self, img_data):
        """  UNUSED
        Rank loss function

        """
        C = 0.0
        for d1 in img_data:
            c = max(V(*d2) * f(*d2[3:]) for d2 in img_data if  \
                    (d2[5] != d2[5]) and ((d1[3] != d2[3]) or (d1[4] != d2[4])))
            C += max(1 + V(*d1) * f(*d1[3:]) * c, 0)
        return C

    def loss(self, img_data):
        """  UNUSED
        Final objective loss function.

        """
        C = self.C(img_data)
        K = self.lamb1 * self.K()
        L = self.lamb2 * L(img_data)
        return C + K + L

    def predict_R(self, O1, O2):
        """  UNUSED
        R* = argmax_R V(R, Z | <O_1, O_2>) f(R,W)

        """
        N = rel_feats.shape[1]
        K = obj_probs.shape[1]

        Rs = [(i,j,k) for i in range(N) for j in range(N) for k in range(K)]
        M = sorted(Rs, key=lambda R: -V(R,O1,O2) * f(R,W,b))
        return M[-1]
