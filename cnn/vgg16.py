import inspect
import os
import time
import numpy as np
import tensorflow as tf
import subprocess


# From: https://www.tensorflow.org/get_started/summaries_and_tensorboard
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)



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

    def build(self, bgr_images_var, train=False, output_size=100):
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

        self.fc6 = self.fc_layer(self.pool5, 'fc6', train)

        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)
        if train:
            self.relu6 = tf.nn.dropout(self.relu6, 0.5)

        self.fc7 = self.fc_layer(self.relu6, 'fc7', train)
        self.relu7 = tf.nn.relu(self.fc7)
        if train:
            self.relu7 = tf.nn.dropout(self.relu7, 0.5)

        self.fc8 = self.fc_layer(self.relu7, 'fc8', train,
                                       w_init_shape=[4096, output_size],
                                       b_init_shape=[output_size])
        # with tf.name_scope('prob'):
        self.prob = tf.nn.softmax(self.fc8, name='prob')
        # variable_summaries(self.prob)

        self.data_dict = None

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def conv_layer(self, bottom, name, train=False):
        with tf.name_scope(name):
            filt = self.get_conv_filter(name, train)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name, train)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name, train=False, w_init_shape=None, b_init_shape=None):
        with tf.name_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name, train, w_init_shape)
            biases = self.get_bias(name, train, b_init_shape)

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            variable_summaries(fc)

            return fc

    def get_conv_filter(self, layer_name, train=False, init_shape=None):
        return self.get_var(layer_name, 0, 'filter', train, init_shape)

    def get_bias(self, layer_name, train=False, init_shape=None):
        var = self.get_var(layer_name, 1, 'biases', train, init_shape)
        return var

    def get_fc_weight(self, layer_name, train=False, init_shape=None):
        var = self.get_var(layer_name, 0, 'weights', train, init_shape)
        return var

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
        with tf.name_scope('accuracy'):
            with tf.name_scope('hits'):
                hits = tf.equal(tf.argmax(self.prob, 1), tf.argmax(GTs, 1))
                hits = tf.cast(hits, tf.float32)
                variable_summaries(hits)

            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(hits)
            tf.summary.scalar('accuracy', accuracy)

        return accuracy

    def get_all_var(self):
        D = self.var_dict
        var_list = [(var, name, D[name].index(var)) for name in D for var in D[name]]
        return zip(*var_list)

    def save_npy(self, sess, save_path='vgg.npy'):
        assert isinstance(sess, tf.Session)
        data_dict = {}

        vars_, names, idxs = self.get_all_var()
        var_out = sess.run(vars_)

        for i in xrange(len(names)):
            if names[i] not in data_dict:
                data_dict[names[i]] = [None] * 2
            data_dict[names[i]][idxs[i]] = var_out[i]

        np.save(save_path, data_dict)


    def get_train_op(self, optimizer='gradient', opt_params={}):
        with tf.name_scope('ground_truth'):
            ground_truth = tf.placeholder(tf.float32, shape=self.prob.get_shape())

        with tf.name_scope('cost'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prob, labels=ground_truth)
            variable_summaries(cross_entropy)
            with tf.name_scope('mean'):
                cost = tf.reduce_mean(cross_entropy)

        opt_dict = { 'gradient': tf.train.GradientDescentOptimizer,
                     'rmsprop' : tf.train.RMSPropOptimizer,
                     'adagrad' : tf.train.AdagradOptimizer,
                     'adadelta': tf.train.AdadeltaOptimizer,
                     'adam'    : tf.train.AdamOptimizer     }
        Optimizer = opt_dict[optimizer]
        with tf.name_scope('train'):
            train_op = Optimizer(**opt_params).minimize(cost)

        return ground_truth, cost, train_op
