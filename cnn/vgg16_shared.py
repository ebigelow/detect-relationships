import inspect
import os
import time
import numpy as np
import tensorflow as tf
import subprocess

# From: https://github.com/machrisaa/tensorflow-vgg/issues/7
class CustomVgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(CustomVgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, 'vgg.npy')
            vgg16_npy_path = path
            print vgg16_npy_path

        self.data_dict = np.load(vgg16_npy_path).item()
        self.var_dict = {}
        print 'npy file loaded from', vgg16_npy_path

    def build(self, bgr_images_var, train=False, n_objs=100, n_rels=70):
        # rgb_scaled = rgb_images_var * 255.0
        # bgr_images_var = tf_rgb2bgr(rgb_scaled)

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
        self.fc6_obj = self.fc_layer(self.pool5, 'fc6_obj', train)  # TODO put fc6_obj/rel as vgg16 initially
        self.fc6_rel = self.fc_layer(self.pool5, 'fc6_rel', train)
        self.relu6_obj = tf.nn.relu(self.fc6_obj)
        self.relu6_rel = tf.nn.relu(self.fc6_rel)
        if train:
            self.relu6_obj = tf.nn.dropout(self.relu6_obj, 0.5)
            self.relu6_rel = tf.nn.dropout(self.relu6_rel, 0.5)

        self.fc7_obj = self.fc_layer(self.relu6_obj, 'fc7_obj', train)
        self.fc7_rel = self.fc_layer(self.relu6_rel, 'fc7_rel', train)
        self.relu7_obj = tf.nn.relu(self.fc7_obj)
        self.relu7_rel = tf.nn.relu(self.fc7_rel)
        if train:
            self.relu7_obj = tf.nn.dropout(self.relu7_obj, 0.5)
            self.relu7_rel = tf.nn.dropout(self.relu7_rel, 0.5)

        self.fc8_obj = self.fc_layer(self.relu7_obj, 'fc8_obj', train,
                                       w_init_shape=[4096, n_objs],
                                       b_init_shape=[n_objs])
        self.fc8_rel = self.fc_layer(self.relu7_rel, 'fc8_rel', train,
                                       w_init_shape=[4096, n_rels],
                                       b_init_shape=[n_rels])
        self.prob_obj = tf.nn.softmax(self.fc8_obj, name='prob_obj')
        self.prob_rel = tf.nn.softmax(self.fc8_rel, name='prob_rel')
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
        # TODO update for shared weights
        with tf.variable_scope('accuracy'):
            with tf.variable_scope('correct_prediction'):
                correct_predictions = tf.equal(tf.argmax(self.prob, 1), tf.argmax(GTs, 1))

            with tf.variable_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

            #tf.histogram_summary('correct_prediction', correct_prediction)
            #tf.scalar_summary('accuracy', accuracy)
        return accuracy

    def get_all_var(self):
        D = self.var_dict
        var_list = [(var, name, D[name].index(var)) for name in D for var in D[name]]
        return zip(*var_list)

    def save_npy(self, sess, save_path='vgg.npy', upload_path=None):
        assert isinstance(sess, tf.Session)
        data_dict = {}

        vars_, names, idxs = self.get_all_var()
        var_out = sess.run(vars_)

        for i in xrange(len(names)):
            if names[i] not in data_dict:
                data_dict[names[i]] = [None] * 2
            data_dict[names[i]][idxs[i]] = var_out[i]

        np.save(save_path, data_dict)
        if upload_path:
            subprocess.call(['skicka','-no-browser-auth','upload',upload_path])
            print 'saved to: {}\nuploaded to: {}'.format(save_path, upload_path)
        else:
            print 'saved to: {}\nnot uploaded.'.format(save_path)

    def get_train_op(self, learning_rate=0.005):
        # TODO update for shared weights
        with tf.variable_scope('ground_truth'):
            ground_truth = tf.placeholder(tf.float32, shape=self.prob.get_shape())

        with tf.variable_scope('cost'):
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.prob, ground_truth))

        # TODO: different training operations?
        #train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        #tf.scalar_summary('cost', cost)
        return ground_truth, cost, train_op
