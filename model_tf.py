# from utils import *
import numpy as np
from numpy.random import randint
from scipy.spatial.distance import cosine
from tqdm import tqdm, trange
import tensorflow as tf
import subprocess


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# TF helper functions


def repeat(X, n_repeats):
    X = tf.tile(X[...,None], [1, n_repeats])    # flatten(X) = 123123123
    X = tf.transpose(X)                         # flatten(X) = 111222333
    return tf.reshape(X, [-1])



# TODO: save somewhere else -- this is a useful function!
# def matrix_repeat(A, n_repeats=None):
#     dims = A.get_shape()
#     n_repeats = dims[0] if n_repeats is None else n_repeats
#     X = tf.reshape(A, [-1])         # (tensor -> vector)
#     X = tf.tile(X, [n_repeats, 1])  # (vector -> matrix)    # Tile along new axis
#     X = tf.reshape(X, [-1])         # (matrix -> vector)
#     dims[0] *= n_repeats                                    # Return tensor dimensions
#     return tf.reshape(X, dims)      # (vector -> tensor)

def tile(A):
    return tf.tile(A, [int(A.get_shape()[0]), 1])

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor.
          https://www.tensorflow.org/versions/r0.11/how_tos/summaries_and_tensorboard/index.html
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

def cosine_dist(V1, V2):
    prod = tf.reduce_sum(V1 * V2, 1)   # inner product across batch
    norm = tf.reduce_sum(tf.sqrt(V1), 1) * tf.reduce_sum(tf.sqrt(V2), 1)
    return prod / norm

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


class Model:
    """
    w2v

    n : number of objects
    k : number of predicates

    lamb1 : weight of eq 5 in eq 7
    lamb2 : weight of eq 4 in eq 7
    noise : weight initialization variance (gaussian)

    W,b : language model weight & bias
    Z,s : visual model weight & bias
    R_samples : list of random (R, O1, O2) samples generated for equation 4

    obj_probs: (2 x num_objects x batch_size)
    rel_feats: (4096 x batch_size)


    TODO
    ----
    - how to allow for either vggnet plugin OR load table??

    """
    def __init__(self, w2v, cnn_dim=4096,
                 lamb1=0.05, lamb2=0.001, init_noise=0.05):
        n, w2v_o = w2v['obj'].shape
        k, w2v_r = w2v['rel'].shape
        w2v_dim = w2v_o + w2v_r
        # Add extra dimension for batch-padding data
        w2v['obj'] = np.concatenate([w2v['obj'], np.zeros((1, w2v_o), dtype=np.float32)], axis=0)
        w2v['rel'] = np.concatenate([w2v['rel'], np.zeros((1, w2v_r), dtype=np.float32)], axis=0)
        self.w2v = w2v
        self.n = n+1
        self.k = k+1

        self.cnn_dim = cnn_dim
        self.w2v_dim = w2v_dim
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.init_weights(init_noise)

    def save_weights(self, sess, file_path='./model_tf.npy', upload_path=None):
        assert isinstance(sess, tf.Session)
        vars_ = {'W':self.W, 'b':self.b, 'Z': self.Z, 's':self.s}
        var_out = sess.run(vars_)
        data_dict = {}

        np.save(file_path, data_dict)
        if upload_path is not None:
            subprocess.call(['skicka','-no-browser-auth','upload',upload_path])
            print '- Saved to: {}\n- Uploaded to: {}'.format(file_path, upload_path)
        else:
            print '- Saved to: {}\n- Not uploaded to drive.'.format(file_path)

    def load_weights(self, filename):
        # TODO
        return


    def w2v_dist(self, Rs1, Rs2):
        I1,J1,K1 = Rs1
        I2,J2,K2 = Rs2
        return cosine_dist( tf.gather(self.w2v['obj'], I1),  tf.gather(self.w2v['obj'], I2) ) +  \
               cosine_dist( tf.gather(self.w2v['obj'], J1),  tf.gather(self.w2v['obj'], J2) ) +  \
               cosine_dist( tf.gather(self.w2v['rel'], K1),  tf.gather(self.w2v['rel'], K2) )

    def d(self, Rs1, Rs2):
        d_rel = self.f(*Rs1) - self.f(*Rs2)
        d_obj = self.w2v_dist(Rs1, Rs2)
        return tf.div(tf.pow(d_rel, 2), d_obj)


    def rand_var(self, var_name, shape, mean=0.01, stddev=0.1):
        var = tf.Variable(tf.truncated_normal(shape, mean, stddev), name=var_name)
        variable_summaries(var, var_name)
        return var

    def init_weights(self, noise):
        cnn_dim, w2v_dim, k = self.cnn_dim, self.w2v_dim, self.k
        with tf.variable_scope('lang-module'):
            self.W = self.rand_var('W', (k, w2v_dim),  stddev=noise)
            self.b = self.rand_var('b', (k, 1),        stddev=noise)
        with tf.variable_scope('vis-module'):
            self.Z = self.rand_var('Z', (k, cnn_dim),  stddev=noise)
            self.s = self.rand_var('s', (k, 1),        stddev=noise)

    def f(self, I, J, K):
        wvec = tf.concat(1, [tf.gather(self.w2v['obj'], I),
                             tf.gather(self.w2v['obj'], J)])
        W_k = tf.gather(self.W, K)
        b_k = tf.gather(self.b, K)
        return tf.reduce_sum(W_k * wvec, 1) + b_k[:,0]

    def V(self, I, J, K, obj_probs, rel_feats):
        cnn = rel_feats
        Z_k = tf.gather(self.Z, K)
        s_k = tf.gather(self.s, K)

        P_i = tf.diag_part(tf.gather(tf.transpose(obj_probs[0]), I))   # TODO will this work ???
        P_j = tf.diag_part(tf.gather(tf.transpose(obj_probs[1]), J))
        P_k = tf.reduce_sum(Z_k * cnn, 1) + s_k[:,0]        # (num_rels, 4096) x (4096, batch_size)
        return P_i * P_j * P_k                              # (1, batch_size)

    # Equation (4)
    def K(self, R_random):
        """
        Minimize variance of ratio from word2vec distance to our function embedding distance.

        """
        R1, R2 = R_random
        dists = tf.pow(self.f(*R1) - self.f(*R2), 2)
        normed = tf.div(dists, self.w2v_dist(R1, R2))
        return tf.nn.moments(normed, axes=0, name='K')       # (shift=None, keep_dims=False)

    # Equation (5)
    def L(self, D, R_full):
        """
        Likelihood of relationships

        """
        I,J,K,_,_ = D
        I2,J2,K2  = R_full
        F1 = self.f(I, J, K)
        F2 = self.f(I2, J2, K2)

        rank = F2[None, ...] - F1[..., None] + 1
        relu = tf.nn.relu(rank)
        with tf.variable_scope('L'):
            L = tf.reduce_sum(relu)
        return L

    # Equation (6)
    def C(self, D):
        """
        Rank loss function

        """
        I, J, K, obj_probs, rel_feats = D
        Vs = self.V(I, J, K, obj_probs, rel_feats)
        Fs = self.f(I, J, K)

        b = int(Vs.get_shape()[0])
        Vs2 = tf.tile(Vs[None, ...], [b, 1])    # shape: (b, b,)
        Fs2 = tf.tile(Fs[None, ...], [b, 1])    # shape: (b, b,)

        # Zero out diagonal entries for the max R', O1', O2'
        eye  = tf.reshape(np.eye(b), Vs2.get_shape())
        diag = tf.ones_like(Vs2) - tf.to_float(eye)
        Vs2 *= diag
        Fs2 *= diag

        val_max = tf.reduce_max(Vs2 * Fs2, 0)   # shape: (b, b,) -> (b,)
        val_gt  = Vs * Fs                       # shape: (b, 1)

        # Will the `1.0 - ...` here work??? -> TODO: test in isolation! (TODO interactive session)
        rank_loss = tf.maximum(tf.ones_like(val_gt) - val_gt + val_max, tf.zeros_like(val_gt))
        with tf.variable_scope('C'):
            C = tf.reduce_sum(rank_loss)   # collapse across batches
        return C

    # Equation (7)
    def loss(self, ground_truth):
        """
        Final objective loss function.

        D: list of (Rs, obj_probs, rel_feats) for each image

        """
        D = [ground_truth[key] for key in ('I','J','K','obj_probs','rel_feats')]
        R_full = (ground_truth['I_full'], ground_truth['J_full'], ground_truth['K_full'])

        C = self.C(D)
        L = self.lamb1 * self.L(D, R_full)
        # K = self.lamb2 * self.K()
        with tf.variable_scope('loss'):
            loss = C + L # + K
            tf.scalar_summary('loss', loss)
            tf.scalar_summary('C', C)
            tf.scalar_summary('L', L)
            # tf.scalar_summary('K', K)
        return loss



    def compute_accuracy(self, ground_truth, top_k=20):
        """
        Compute accuracy, predicting predicates only.
        """
        I, J, K = (ground_truth['I'], ground_truth['J'], ground_truth['K'])
        obj_probs, rel_feats = (ground_truth['obj_probs'], ground_truth['rel_feats'])

        predicts = self.predict_preds(I, J, obj_probs, rel_feats, top_k)
        predictions = predicts.indices[:K.get_shape()[0], :]
        gt = tf.tile(K[..., None], [1, top_k])
        hits = tf.reduce_sum(tf.to_float(tf.equal(gt, predictions)), 1)

        with tf.variable_scope('accuracies'):
            accuracy = tf.reduce_mean(hits)
            tf.scalar_summary('accuracy_top{}'.format(top_k), accuracy)
        return accuracy

    def predict_preds(self, I, J, obj_probs, rel_feats, top_k=20):
        # Visual module
        P_i = tf.diag_part(tf.gather(tf.transpose(obj_probs[0]), I))
        P_j = tf.diag_part(tf.gather(tf.transpose(obj_probs[1]), J))
        P_K = tf.reduce_sum(tf.matmul(rel_feats, self.Z, transpose_b=True), 1) + self.s
        V = P_i * P_j * P_K

        # Language module
        w2v = self.w2v['obj']
        wvec = tf.concat(1, [tf.gather(w2v, I), tf.gather(w2v, J)])
        F = tf.matmul(self.W, wvec, transpose_b=True) + self.b

        # Top N
        with tf.variable_scope('predictions_top{}'.format(top_k)):
            return tf.nn.top_k(tf.transpose(tf.mul(V, F)), k=top_k)


    def predict_IJK(self, obj_probs, rel_feats, top_k=20):
        # Visual module
        P_I = tf.transpose(obj_probs[0])
        P_J = obj_probs[1]
        P_K = tf.reduce_sum(tf.matmul(rel_feats, self.Z), 1) + self.s
        # (100 x batch) * (100 x batch) * (70 x batch)   ->   (100 x 100 x 70)
        V = tf.add([ tf.matmul(  tf.matmul(P_I[:,b], P_J[b,:])[...,None],  P_K[b,:]  )
                     for b in range(P_J.get_shape()[0])])

        # Language module
        w2v_obj = self.w2v['obj']
        w2v_s = np.concatenate([w2v_obj, np.zeros_like(w2v_obj)], axis=1).T   # TODO: make sure transpose works
        w2v_o = np.concatenate([np.zeros_like(w2v_obj), w2v_obj], axis=1).T   # new dim should be 600 x 100
        # (600 x 100) * (600 x 100) * (600 x 70)   ->   (100 x 100 x 70)
        F = tf.add([ tf.matmul(  tf.matmul(self.W[:,q], w2v_o[q,:])[...,None],  w2v_s[q,:]  )
                     for q in range(w2v_s.get_shape()[0])])
        F += self.b[None, None, ...]   # TODO: tile b??

        # Top N
        return tf.nn.top_k(tf.mul(V, F), top_k=top_k)


    def get_ground_truth(self, batch_size, n_rels):
        n, k, cnn_dim = (self.n, self.k, self.cnn_dim)
        with tf.variable_scope('ground_truth'):
            ground_truth = {'I':         tf.placeholder(tf.int32, shape=(batch_size)),
                            'J':         tf.placeholder(tf.int32, shape=(batch_size)),
                            'K':         tf.placeholder(tf.int32, shape=(batch_size)),
                            'I_full':    tf.placeholder(tf.int32, shape=(n_rels)),
                            'J_full':    tf.placeholder(tf.int32, shape=(n_rels)),
                            'K_full':    tf.placeholder(tf.int32, shape=(n_rels)),
                            'obj_probs': tf.placeholder(tf.float32, shape=(2, batch_size, n)),
                            'rel_feats': tf.placeholder(tf.float32, shape=(batch_size, cnn_dim))}
        return ground_truth
