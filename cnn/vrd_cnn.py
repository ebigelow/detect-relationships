from tqdm import tqdm
import numpy as np
import sys; sys.path.append('..')
from utils import load_vrd_batcher
from vgg16 import CustomVgg16
import tensorflow as tf

tf.app.flags.DEFINE_string('which_net',    'relnet', '')
tf.app.flags.DEFINE_integer('output_size', 70,      '')

tf.app.flags.DEFINE_string('weights',   '../data/vrd/models/objnet_run1/vrd_trained_8.npy', 'Load weights from this file')
tf.app.flags.DEFINE_string('save_file', '../data/vrd/models/objnet_run1/out_train.npy',  'Save layer output here')

tf.app.flags.DEFINE_integer('batch_size',  10, '')
tf.app.flags.DEFINE_integer('data_epochs', 20, '')

tf.app.flags.DEFINE_string('obj_list',  '../data/vrd/mat/objectListN.mat',      '')
tf.app.flags.DEFINE_string('rel_list',  '../data/vrd/mat/predicate.mat',        '')
tf.app.flags.DEFINE_string('mat_file',  '../data/vrd/mat/annotation_train.mat', '')
tf.app.flags.DEFINE_string('img_dir',   '../data/vrd/images/train/',        '')
tf.app.flags.DEFINE_string('mean_file', '../data/vrd/images/mean_train.npy', '')

FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':

    train_batcher = load_vrd_batcher(FLAGS.mat_file, FLAGS.obj_list, FLAGS.rel_list, FLAGS.batch_size,
                                     FLAGS.data_epochs, FLAGS.which_net, FLAGS.img_dir, FLAGS.mean_file )

    images_var = tf.placeholder('float', [FLAGS.batch_size, 224, 224, 3])
    net = CustomVgg16(FLAGS.weights)
    net.build(images_var, train=False, output_size=FLAGS.output_size)

    ground_truth = tf.placeholder(tf.float32, shape=net.prob.get_shape())
    accuracy = net.get_accuracy(ground_truth)

    outputs = {}
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        for train_batch in train_batcher:
            for images, labels, uids in train_batch:
                b_fc7, b_prob, b_acc = sess.run([net.fc7, net.prob, accuracy],
                                                feed_dict={images_var: images, ground_truth: labels})
                for q in range(len(uids)):
                    D = {}
                    if FLAGS.which_net == 'rel':
                        D['fc7']   = b_fc7[q]
                    D['prob']  = b_prob[q]
                    D['label'] = labels[q].argmax()
                    outputs[uids[q]] = D

    np.save(FLAGS.save_file, outputs)
    print '\nFile saved to:{}\n'.format(FLAGS.save_file)
