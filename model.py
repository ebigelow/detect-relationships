
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


    # -------------------------------------------------------------------------------------------------------

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
        rel_id = objs_to_reluid(O1, O2)

        P_i = self.obj_probs[O1][i]
        P_j = self.obj_probs[O2][j]
        cnn = self.rel_feats[rel_id]
        P_k = np.dot(Z[k], cnn) + s[k]
        return P_i * P_j * P_k


    def predict_Rs(self, O1, O2):
        """
        Full list of predictions `R`, sorted by confidence

        """
        N,K,W,b,Z,s = (self.n, self.k, self.W, self.b, self.Z, self.s)
        Rs = [(i,j,k) for i in range(N) for j in range(N) for k in range(K)]
        M = sorted(Rs, key=lambda R: -self.V(R,O1,O2,Z,s) * self.f(R,W,b))
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
        test_data = [(self.predict_Rs(O1, O2), R) for R, O1, O2 in GT]

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

    def SGD(self, D, save_file='data/models/vrd_weights.npy'):
        """
        Perform SGD over eqs 5 (L) 6 (C)

        """
        obj_probs, rel_feats, w2v = (self.obj_probs, self.rel_feats, self.w2v)
        V, f, d = (self.V, self.f, self.d)
        W, b, Z, s = (self.W, self.b, self.Z, self.s)
        cost_prev = 0.0

        for epoch in range(self.max_iters):

            # Use to get change in cost (mc = mean cost)
            mc = 0.0

            # Iterate over data points (stochastically)
            for R, O1, O2 in tqdm(D):

                # Get (R, O1, O2) that maximizes term in equation 6
                D_ = [(R_,O1_,O2_) for R_,O1_,O2_ in D if (R_ != R) and (O1_ != O1 or O2_ != O2)]
                M = sorted(D_, key=lambda (R,O1,O2): V(R,O1,O2,Z,s) * f(R,W,b))
                R_,O1_,O2_ = M[0]

                i,j,k = R
                i_,j_,k_ = R_

                # Compute value for ` max{cost, 0} ` in equation 6
                cost = max(1. - V(R,O1,O2,Z,s) * f(R,W,b) + V(R_,O1_,O2_,Z,s) * f(R_,W,b), 0.)
                mc  += cost / len(D)


                # Even epochs --> update W,b
                if epoch % 2 == 0:
                    # Equation 5
                    for R_2, O1_2, O2_2 in D:
                        i_2,j_2,k_2 = R_2
                        if f(R_2, W, b) - f(R, W, b) > 0:
                            W[k]   -= self.learning_rate * self.lamb1 * np.concatenate((w2v[i],  w2v[j]))
                            b[k]   -= self.learning_rate * self.lamb1
                            W[k_2] += self.learning_rate * self.lamb1 * np.concatenate((w2v[i_2], w2v[j_2]))
                            b[k_2] += self.learning_rate * self.lamb1

                    # Equation 6
                    if cost > 0:
                        v = V(R,O1,O2,Z,s)
                        W[k]  -= self.learning_rate * v * np.concatenate((w2v[i], w2v[j]))
                        b[k]  -= self.learning_rate * v
                        v_ = V(R_,O1_,O2_,Z,s)
                        W[k_] += self.learning_rate * v_ * np.concatenate((w2v[i_], w2v[j_]))
                        b[k_] += self.learning_rate * v_


                # Odd epochs --> update Z,s
                else:
                    # Equation 6
                    if cost > 0:
                        id_rel  = objs_to_reluid(O1, O2)
                        id_rel_ = objs_to_reluid(O1_, O2_)

                        Z[k]  -= self.learning_rate * f(R,W,b) * obj_probs[O1][i] * obj_probs[O2][j] * rel_feats[id_rel]
                        s[k]  -= self.learning_rate * f(R,W,b) * obj_probs[O1][i] * obj_probs[O2][j]
                        Z[k_] += self.learning_rate * f(R_,W,b)  * obj_probs[O1_][i_] * obj_probs[O2_][j_] * rel_feats[id_rel_]
                        s[k_] += self.learning_rate * f(R_,W,b)  * obj_probs[O1_][i_] * obj_probs[O2_][j_]

            # # Equation 4
            # if epoch % 2 == 0:
            #     dKfun = sum(    (2. / d(R,R_,W,b)) *
            #                     (f(R,W,b) - f(R_,W,b)) *
            #                     (self.word_vec(*R[:-1]) - self.word_vec(*R_[:-1]))
            #                 for R,R_ in self.R_samples )
            #
            #     Kfun = lambda R, R_: (f(R,W,b) - f(R_,W,b))**2 / d(R,R_,W,b)
            #     Ksum = sum(Kfun(R,R_) for R, R_ in self.R_samples)
            #     nr = self.num_samples
            #     dK_dW = ((2.0 - nr) / nr) * Ksum * dKfun
            #
            #     # TODO: separate `W[k] += ...` and `W[k_] -= ...`  ???
            #     # TODO: if (1 - v + v' > 0) : update
            #
            #     W += self.learning_rate * dK_dW * self.lamb2

            self.W, self.b, self.Z, self.s = (W, b, Z, s)
            self.save_weights(save_file)

            final_obj = cost + self.lamb1 * self.L(D)
            print '\tit {} | change in cost: {}'.format(epoch, final_obj - cost_prev)
            cost_prev = final_obj

            accuracy = self.compute_accuracy(D[:100])
            print '\taccuracy {}'.format(accuracy)

        return W, b, Z, s




    def SGD_parallel(self, D, n_proc=20):
        """
        Perform SGD over eqs 5 (L) 6 (C)

        """
        from SimpleMPI.MPI_map import MPI_map

        obj_probs, rel_feats, w2v = (self.obj_probs, self.rel_feats, self.w2v)
        V, f, d = (self.V, self.f, self.d)
        W, b, Z, s = (self.W, self.b, self.Z, self.s)

        for epoch in range(self.max_iters):

            # Shuffle data
            #D = sorted(D, key=lambda x: np.random.rand())

            # Use to get change in cost
            mc      = 0.0
            mc_prev = 1.0

            # Iterate over data points (stochastically)
            MPI_map(self.SGD_update, D)

            # Equation 4
            if epoch % 2 == 0:
                dKfun = sum(    (2. / d(R,R_,W,b)) *
                                (f(R,W,b) - f(R_,W,b)) *
                                (self.word_vec(*R[:-1]) - self.word_vec(*R_[:-1]))
                            for R,R_ in self.R_samples )

                Kfun = lambda R, R_: (f(R,W,b) - f(R_,W,b))**2 / d(R,R_,W,b)
                Ksum = sum(Kfun(R,R_) for R, R_ in self.R_samples)
                nr = self.num_samples
                dK_dW = ((2.0 - nr) / nr) * Ksum * dKfun

                W += self.learning_rate * dK_dW

            print '\tit {} | change in cost: {}'.format(epoch, mc_prev - mc)
            mc_prev = mc

        self.W, self.b, self.Z, self.s = (W, b, Z, s)
        return W, b, Z, s


    def SGD_update(self, R, O1, O2):
        obj_probs, rel_feats, w2v = (self.obj_probs, self.rel_feats, self.w2v)
        V, f, d = (self.V, self.f, self.d)
        W, b, Z, s = (self.W, self.b, self.Z, self.s)

        # Get 2nd data point that maximizes equation ??? TODO
        D_ = [(R_,O1_,O2_) for R_,O1_,O2_ in D if (R_ != R) and (O1_ != O1 or O2_ != O2)]
        M = sorted(D_, key=lambda (R,O1,O2): V(R,O1,O2,Z,s) * f(R,W,b))
        R_,O1_,O2_ = M[0]

        i,j,k = R
        i_,j_,k_ = R_

        # Compute value for ` max{cost, 0} ` in equation 6
        cost = 1 - V(R,O1,O2,Z,s) * f(R,W,b) + V(R_,O1_,O2_,Z,s) * f(R_,W,b)
        mc  += cost / len(D)


        # Even epochs --> update W,b
        if epoch % 2 == 0:
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


        # Odd epochs --> update Z,s
        else:
            # Equation 6
            if cost > 0:
                id_rel  = objs_to_reluid(O1, O2)
                id_rel_ = objs_to_reluid(O1_, O2_)

                Z[k]  -= self.learning_rate * f(R,W,b) * obj_probs[O1][i] * obj_probs[O2][j] * rel_feats[id_rel]
                s[k]  -= self.learning_rate * f(R,W,b) * obj_probs[O1][i] * obj_probs[O2][j]
                Z[k_] += self.learning_rate * f(R_,W,b)  * obj_probs[O1_][i_] * obj_probs[O2_][j_] * rel_feats[id_rel_]
                s[k_] += self.learning_rate * f(R_,W,b)  * obj_probs[O1_][i_] * obj_probs[O2_][j_]

        self.W, self.b, self.Z, self.s = (W, b, Z, s)




    # -------------------------------------------------------------------------------------------------------
    # Helper Methods

    def word_vec(self, i, j):
        return np.concatenate((self.w2v[i], self.w2v[j]))



    # -------------------------------------------------------------------------------------------------------
    # Unused

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
        fn = lambda R1, R2: max(self.f(R1, self.W, self.b) - self.f(R2, self.W, self.b) + 1, 0)
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
