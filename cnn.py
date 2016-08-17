"""


Parameters
----------
cnn_classes : path to python classes output from caffe-tensorflow ;
              should have `RelationNet` and `ObjectNet` as class names
obj_weights : path to numpy weights file for ObjectNet
rel_weights : path to numpy weights file for RelationNet

obj_dir : directory with class.py, weights.npy, mean.npy

batch_size : process this many images at a time
crop_size  : crop images to square with sides of this length


"""

import sys, os
from cv2 import imread
from utils import *
import numpy as np
import tensorflow as tf
from numpy.random import randint
    from scipy.spatial.distance import cosine
import pickle
sys.path.append('/u/ebigelow/lib/caffe-tensorflow')



def load_cnn(cnn_dir,
             batch_size=10, new_layer=None,
             crop_size=224, train=False):
    sys.path.append(cnn_dir)
    cnn = __import__('net')
    tf.reset_default_graph()
    if train:
        images_var = tf.placeholder(tf.float32, [batch_size, crop_size, crop_size, 3])
        images_batch = tf.reshape(images_var, [-1, crop_size, crop_size, 3])
    else:
        images_var = tf.placeholder(tf.float32, [crop_size, crop_size, 3])
        images_batch = tf.reshape(images_var, [-1, crop_size, crop_size, 3])
    net = cnn.CaffeNet({'data': images_batch}, trainable=train)
    graph = tf.get_default_graph()
    prob = make_prob(graph, new_layer)
    return prob, graph, net, images_var

def make_prob(graph, new_layer):
    if new_layer is not None:
        fc7 = graph.get_tensor_by_name('fc7/fc7:0')
        with tf.variable_scope('fc8_new'):
            fc8 = make_fc8(fc7, new_layer)
        with tf.variable_scope('prob'):
            prob = tf.nn.softmax(fc8)
    else:
        prob = graph.get_tensor_by_name('prob')
    return prob

def make_fc8(fc7, layer_size):
    init = tf.random_normal_initializer()
    fc8W = tf.get_variable('weights', [4096, layer_size], initializer=init)
    fc8b = tf.get_variable('biases',  [layer_size],       initializer=init)
    with tf.variable_scope('fc8'):
        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
    return fc8



def train_cnn(cnn_dir, data,
              batch_size=10, learning_rate=0.001, new_layer=None,
              ckpt_file='model.ckpt', init_weights=None):
    prob, graph, net, images_var = load_cnn(cnn_dir, new_layer=new_layer, train=True)
    ground_truth = tf.placeholder(tf.float32, shape=[batch_size, prob.get_shape()[1]])

    cost = tf.nn.sigmoid_cross_entropy_with_logits(prob, ground_truth)
    # TODO: use other optimizers
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    saver = tf.train.Saver()
    ckpt_path = os.path.join(cnn_dir, ckpt_file)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        if os.path.exists(ckpt_path):
            saver.restore(sess, ckpt_path)
        elif init_weights is not None:
           net.load(init_weights, sess)

        for e, (batch_imgs, batch_labels) in enumerate(data):
            train_dict = {images_var:batch_imgs, ground_truth:batch_labels}
            sess.run(train_op, feed_dict=train_dict)
            save_path = saver.save(sess, ckpt_path)
            if e % 20 == 0:
                save_path = saver.save(sess, ckpt_path + '.' + str(e))
                print('Model saved: {}   Batch: {}'.format(save_path, e))


def run_cnn(images, cnn_dir, ckpt_file,
            layer='prob', new_layer=None):
    prob, graph, net, images_var = load_cnn(cnn_dir, new_layer=new_layer, batch_size=1)
    # epochs = int(np.ceil(float(len(data)) / batch_size))
    graph_layer = prob if layer == 'prob' else graph.get_tensor_by_name(layer)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver.restore(sess, os.path.join(cnn_dir, ckpt_file))

        layer_out  = []
        for img in images:
            feed = {images_var: img}
            batch_prob = sess.run(graph_layer, feed_dict=feed)[0]   # TODO should this be the 0th index?
            layer_out.append(batch_prob)

    return np.vstack(layer_out)

def test_cnn(data, cnn_dir, ckpt_file='model.ckpt', new_layer=None):
    images, labels = zip(*data)
    output = run_cnn(images, cnn_dir, ckpt_file, new_layer=new_layer)

    predictions = output.argmax(axis=1)
    N = float(len(data))
    accuracy = sum(np.array(labels).argmax(axis=1) == predictions) / N
    return accuracy
