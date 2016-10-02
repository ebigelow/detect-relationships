# from utils import *
import numpy as np
from numpy.random import randint
from scipy.spatial.distance import cosine
from tqdm import tqdm, trange
import tensorflow as tf


def repeat(A, n_repeats):
    #
    # TODO: fix, test
    #
    # http://stackoverflow.com/a/35367161/4248948
    vectorized = tf.reshape(A, [-1, 1])                     # Convert to a len(yp) x 1 matrix.
    tiled      = tf.tile(vectorized, [n_repeats, 1, 1])     # Create multiple columns.
    reshaped   = tf.TODO_SWAP_AXES(tiled, (0,1))            # Convert back to a vector.
    return reshaped

def tile(A, batch_size=None):
    if batch_size is None: batch_size = self.batch_size
    tf.tile(tf.expand_dims(A, axis=0), n=batch_size, axis=0)




class Model:
    """
    w2v
    obj_probs : obj_probs[obj_uid] = (100,) final layer cnn output
    rel_feats : rel_feats[rel_uid] = (4096,) fc7 feature (PRESUMABLY this was used . . .)
    num_samples : number of random R samples to compute for equation 4
    noise : weight initialization variance (gaussian)
    lamb1 : weight of eq 5 in eq 7
    lamb2 : weight of eq 4 in eq 7
    n : number of objects
    k : number of predicates
    W,b : language model weight & bias
    Z,s : visual model weight & bias
    R_samples : list of random (R, O1, O2) samples generated for equation 4



        #
        # TODO TODO TODO
        # input should be two images for each data point, so batch_size for
        #  CNN should actually be  num_objects * 2  -- then we need to
        #  re-map these a little bit later down the line!
        #
        # TODO: (i,j,k) should be 3 one-hot vector stacks, each --> (70 x batch_size)

        # TODO: self.obj_w2v and rel_w2v




        # self.obj_probs: (2 x num_objects x batch_size)
        # self.rel_feats: (4096 x batch_size)


    """
    def __init__(self, obj_w2v, rel_w2v,
                 init_noise=0.05, lamb1=0.05, lamb2=0.001):
        self.obj_w2v       = obj_w2v
        self.rel_w2v       = rel_w2v
        self.noise         = init_noise
        self.lamb1         = lamb1
        self.lamb2         = lamb2
        self.init_weights()

    def save_npy(self, sess, file_path='./vgg.npy'):
        assert isinstance(sess, tf.Session)
        vars_ = [self.W, 'b':self.b, 'Z': self.Z, 's':self.s}
        var_out = sess.run(vars_)
        data_dict = {}

        np.save(file_path, data_dict)
        print 'file saved: {}'.format(file_path)


    def save_weights(self, filename):
        # TODO

    def load_weights(self, filename):
        # TODO

    def w2v_dist(self, Rs1, Rs2):
        I,J,K = Rs1
        I_,J_,K_ = Rs1
        return tf.cosine_dist(tf.matmul(self.obj_w2v, I), tf.matmul(self.obj_w2v, I_)) +  \
               tf.cosine_dist(tf.matmul(self.obj_w2v, J), tf.matmul(self.obj_w2v, J_)) +  \
               tf.cosine_dist(tf.matmul(self.rel_w2v, K), tf.matmul(self.rel_w2v, K_))

    def d(self, Rs1, Rs2):
        d_rel = self.f(Rs1) - self.f(Rs2)
        d_obj = self.w2v_dist(Rs1, Rs2)
        d = tf.div(tf.pow(d_rel, 2), d_obj)
        return tf.max(d, 0.0) # TODO: why is this here?


    def rand_var(self, var_name, shape, mean=0.01, stddev=0.1):
        return tf.Variable(tf.truncated_normal(shape, mean, stddev), name=var_name)

    def init_weights(self):
        cnn_dim = self.rel_feats.get_shape().as_list()[1]
        w2v_dim = self.w2v.get_shape().as_list()[1]
        v, k = (self.noise, self.k)

        self.W = self.rand_var('W', (k, word_dim), stddev=v)
        self.b = self.rand_var('b', (k, 1),        stddev=v)
        self.Z = self.rand_var('Z', (k, nfeats),   stddev=v)
        self.s = self.rand_var('s', (k, 1),        stddev=v)

    def f(self, Rs):
        I, J, K = Rs
        wvec = tf.matmul(tf.concat([self.obj_w2v, self.obj_w2v], axis=1),
                         tf.concat(I, J))
        W_k = tf.mul(self.W, K)
        b_k = tf.mul(self.b, K)
        with tf.variable_scope('f'):
            return tf.nn.bias_add(tf.matmul(W_k, wvec), b_k)

    def V(self, Rs, obj_probs, rel_feats):
        """
        Reduce relationship <i,j,k> to scalar visual space.

        """
        I, J, K = Rs

        P_i = tf.matmul(self.obj_probs[0], I)
        P_j = tf.matmul(self.obj_probs[1], J)

        cnn = self.rel_feats
        Z_k = tf.mul(self.Z, self.K)
        s_k = tf.mul(self.s, self.K)
        # (num_rels x 4096) X (4096 x batch_size)
        P_k = tf.nn.bias_add(tf.matmul(Z_k, cnn), s_k)

        # (1 x batch_size)
        with tf.variable_scope('f'):
            return tf.mul(P_i, P_j, P_k)


    def predict_preds(self, D, topn=20):
        """
        Predict predicate given object labels.
        """
        (I,J,K), obj_probs, rel_feats = D
        preds = [k_ for k_ in range(self.k)]
        preds = sorted(preds, key=lambda x: -self.V((i,j,x),O1,O2) * self.f((i,j,x)))
        return preds[:topn]

    def compute_accuracy2(self, D, topn=20):
        """
        Compute accuracy, predicting predicates only.
        """
        predictions = [(self.predict_preds(R, O1, O2, topn), R[2]) for R, O1, O2 in D]
        accuracy = np.mean([int(truth in p) for p,truth in predictions])
        return accuracy

    # Equation (4)
    def K(self, R_rand):
        Rs1, Rs2 = (repeat(Rs), tile(Rs))
        dists = tf.pow(self.f(Rs1) - self.f(Rs2), 2)
        normed = tf.div(dists, self.d(Rs1, Rs2))
        var = tf.nn.moments(normed, axes=0, name='K')       # (shift=None, keep_dims=False)
        return var

    # Equation (5)
    def L(self, R_full):
        """
        Likelihood of relationships

        TODO: note to self
        > batch_size in this file is really the number of relationships for 1 image
        > really, batch_size should be how many data points we run at a single time
        > (this number should be much smaller, at CNN level at least)

        """
        # repeat = lambda A: tf_repeat(A, n_repeats=batch_size)
        # tile   = lambda A: tf.tile(tf.expand_dims(A, axis=0), n=batch_size, axis=0)

        I,J,K = R_full
        Rs1 = (repeat(I), repeat(J), repeat(K))    # inner sum    shape: (b, b, n, 1)
        Rs2 = (  tile(I),   tile(J),   tile(K))    # outer sum    shape: (b, b, n, 1)

        rank = self.f(Rs2) - self.f(Rs1) + 1.0
        relu = tf.reduce_max(rank, 0.0, axis=0)
        return tf.reduce_sum(relu)

    # Equation (6)
    def C(self, D):
        """
        Rank loss function

        """
        Rs, obj_probs, rel_feats = D
        Vs = self.V(Rs, obj_probs, rel_feats)
        Fs = self.f(Rs)

        Vs2 = tile(Vs)  # shape: (b, b, 100)
        Fs2 = tile(Fs)

        # Zero out diagonal entries -- this is for the second max R', O1', O2'
        #  This looks weird because R is a list (I,J,K), instead of a 3-... tensor
        #  because we have different numbers of objects / relations
        for i in range(batch_size):
            # TODO: will this work?
            Vs2[i,i,:] = Vs2[i,i,:] - Vs2[i,i,:]
            Fs2[i,i,:] = Fs2[i,i,:] - Fs2[i,i,:]

        V_max   = tf.reduce_max(tf.mul(Vs2, Fs2), axis=0)   # shape: (b, b, 1) -> (b, 1)
        V_truth = tf.mul(Vs, Fs)                            # shape: (b, 1)

        rank_loss = tf.max(1.0 - V_truth + V_max, 0.0)
        return tf.reduce_sum(rank_loss)

    # Equation (7)
    def loss(self, D, R_full):
        """
        Final objective loss function.

        D: list of (Rs, obj_probs, rel_feats) for each image
        R_full: used by `L`;   I,J,K for full dataset

        """
        C = self.C(D)
        L = tf.mul(self.lamb1, self.L(R_full))
        K = tf.mul(self.lamb2, self.K())
        return C + L + K

    # Equation (8)
    def predict_Rs(self, obj_probs, rel_feats, topn=100):
        """
        Full list of predictions `R`, sorted by confidence

        """
        N,K = (self.n, self.k)
        Rs = [(i,j,k) for i in range(N) for j in range(N) for k in range(K)]
        M = sorted(Rs, key=lambda R: -self.V(R,O1,O2) * self.f(R))
        return M[:topn]
