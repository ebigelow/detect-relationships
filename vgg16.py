import inspect
import os
import time
import numpy as np
import tensorflow as tf
import skimage
from utils import tf_rgb2bgr


VGG_MEAN = [103.939, 116.779, 123.68]

# From: https://github.com/machrisaa/tensorflow-vgg/issues/7
class CustomVgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, 'vgg.npy')
            vgg16_npy_path = path
            print vgg16_npy_path

        self.data_dict = np.load(vgg16_npy_path).item()
        self.var_dict = {}
        print 'npy file loaded'

    def build(self, rgb_images_var, train=False, output_size=100):
        rgb_scaled = rgb_images_var * 255.0
        bgr_images_var = tf_rgb2bgr(rgb_scaled)

        self.conv1_1 = self.conv_layer(bgr_images_var, 'conv1_1', train)
        self.conv1_2 = self.conv_layer(self.conv1_1,   'conv1_2', train)
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1,   'conv2_1', train)
        self.conv2_2 = self.conv_layer(self.conv2_1, 'conv2_2', train)
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2,   'conv3_1', train)
        self.conv3_2 = self.conv_layer(self.conv3_1, 'conv3_2', train)
        self.conv3_3 = self.conv_layer(self.conv3_2, 'conv3_3', train)
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3,   'conv4_1', train)
        self.conv4_2 = self.conv_layer(self.conv4_1, 'conv4_2', train)
        self.conv4_3 = self.conv_layer(self.conv4_2, 'conv4_3', train)
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4,   'conv5_1', train)
        self.conv5_2 = self.conv_layer(self.conv5_1, 'conv5_2', train)
        self.conv5_3 = self.conv_layer(self.conv5_2, 'conv5_3', train)
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        # if train:
        #     with tf.device('/cpu:0'):
        #         self.fc6 = self.fc_layer(self.pool5, 'fc6', train)
        # else:
        self.fc6 = self.fc_layer(self.pool5, 'fc6', train)

        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)
        if train:
            self.relu6 = tf.nn.dropout(self.relu6, 0.5)

        self.fc7 = self.fc_layer(self.relu6, 'fc7', train)
        self.relu7 = tf.nn.relu(self.fc7)
        if train:
            self.relu7 = tf.nn.dropout(self.relu7, 0.5)

        # replace this one with our own layer of result
        # self.fc8 = self.fc_layer(self.relu7, 'fc8', train)
        # self.prob = tf.nn.softmax(self.fc8, name='prob')

        self.fc8 = self.fc_layer(self.relu7, 'fc8_custom', train,
                                       w_init_shape=[4096, output_size],
                                       b_init_shape=[output_size])
        self.prob = tf.nn.softmax(self.fc8, name='prob')
        tf.histogram_summary('prob', self.prob)

        self.data_dict = None

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def conv_layer(self, bottom, name, train=False):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name, train)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name, train)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name, train=False, w_init_shape=None, b_init_shape=None):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name, train, w_init_shape)
            biases = self.get_bias(name, train, b_init_shape)

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            tf.histogram_summary(name, fc)
            return fc

    def get_conv_filter(self, layer_name, train=False, init_shape=None):
        return self.get_var(layer_name, 0, 'filter', train, init_shape)

    def get_bias(self, layer_name, train=False, init_shape=None):
        return self.get_var(layer_name, 1, 'biases', train, init_shape)

    def get_fc_weight(self, layer_name, train=False, init_shape=None):
        return self.get_var(layer_name, 0, 'weights', train, init_shape)

    def get_var(self, layer_name, idx, var_name, train, init_shape):
        if train:
            if layer_name in self.data_dict:
                data = self.data_dict[layer_name][idx]
                var = tf.Variable(data, name=var_name)
                print '{}:{}/{} loaded from npy'.format(layer_name, idx, var_name)
            else:
                var = tf.Variable(tf.truncated_normal(init_shape, 0.01, 0.1), name=var_name)
                print '{}:{}/{} randomly initialized'.format(layer_name, idx, var_name)
        else:
            data = self.data_dict[layer_name][idx]
            var = tf.constant(data, name=var_name)
            print '{}:{}/{} loaded from npy'.format(layer_name, idx, var_name)

        if layer_name not in self.var_dict:
            self.var_dict[layer_name] = [None] * 2
        self.var_dict[layer_name][idx] = var

        return var

    def get_accuracy(self, GTs):
        with tf.variable_scope('accuracy'):
            with tf.variable_scope('correct_prediction'):
                correct_predictions = tf.equal(tf.argmax(self.prob, 1), tf.argmax(GTs, 1))

            with tf.variable_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

            tf.histogram_summary('correct_prediction', correct_prediction)
            tf.scalar_summary('accuracy', accuracy)
        return accuracy

    def get_all_var(self):
        D = self.var_dict
        var_list = [(var, name, idx) for name in D for idx in D[name]]
        return zip(*var_list)

    def save_npy(self, sess, file_path='./vgg.npy'):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        vars_, names, idxs = self.get_all_var()
        var_out = sess.run(vars_)

        for i in xrange(len(names)):
            if names[i] not in data_dict:
                data_dict[names[i]] = [None] * 2
            data_dict[names[i]][idxs[i]] = var_out[i]

        np.save(file_path, data_dict)
        print 'file saved: {}'.format(file_path)

    def get_train_op(self, learning_rate=0.001):
        with tf.variable_scope('ground_truth'):
            ground_truth = tf.placeholder(tf.float32, shape=[batch_size, self.prob.get_shape()[1]])

        with tf.variable_scope('cost'):
            cost = tf.nn.sigmoid_cross_entropy_with_logits(self.prob, ground_truth)

        # TODO: different training operations?
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        tf.scalar_summary('cost', cost)
        return ground_truth, cost, train_op
