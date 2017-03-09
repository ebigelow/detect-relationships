import tensorflow as tf
import numpy as np
import sys; sys.path.append('..')
from utils import load_vrd_batcher
from vgg16 import CustomVgg16
from tqdm import trange, tqdm

# tf.app.flags.DEFINE_float('gpu_mem_fraction', 0.9, '')

tf.app.flags.DEFINE_integer('output_size', 100, 'Size of final output layer')
tf.app.flags.DEFINE_string('which_net', 'obj', '')

tf.app.flags.DEFINE_string('init_path', 'data/models/vgg16.npy', 'Initial weights')
tf.app.flags.DEFINE_string('save_dir',  'data/models/objnet/run1/', 'Save weights to this file')
tf.app.flags.DEFINE_integer('test_freq',  10, 'Compute test accuracy every N data epochs.')

tf.app.flags.DEFINE_integer('batch_size',  10, '')
tf.app.flags.DEFINE_integer('data_epochs', 20, '')
tf.app.flags.DEFINE_integer('meta_epochs', 20, '')

tf.app.flags.DEFINE_string('obj_list',  'data/vrd/objectListN.mat',      '')
tf.app.flags.DEFINE_string('rel_list',  'data/vrd/predicate.mat',        '')
tf.app.flags.DEFINE_string('train_mat', 'data/vrd/annotation_train.mat', '')
tf.app.flags.DEFINE_string('test_mat',  'data/vrd/annotation_test.mat',  '')

tf.app.flags.DEFINE_string('train_imgs', 'data/vrd/images/train/', '')
tf.app.flags.DEFINE_string('test_imgs',  'data/vrd/images/test/',  '')
tf.app.flags.DEFINE_string('mean_file',  'data/vrd/images/mean_train.npy',      '')

# All optimizers
tf.app.flags.DEFINE_string('optimizer', 'gradient', 'gradient | adagrad | adadelta | adam | rmsprop')
tf.app.flags.DEFINE_boolean('use_locking', False, 'If True use locks for update operations.')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'How fast do we descend?')    # 0.001
# Adagrad
tf.app.flags.DEFINE_float('initial_accumulator_value', 0.1, 'Starting value for the accumulators, must be positive.')
# Adadelta
tf.app.flags.DEFINE_float('rho', 0.95, 'The decay rate.')
tf.app.flags.DEFINE_float('epsilon', 1e-08, 'A constant epsilon used to better conditioning the grad update.')
# Adam
tf.app.flags.DEFINE_float('beta1', 0.9, 'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float('beta2', 0.999, 'The exponential decay rate for the 2nd moment estimates.')
##tf.app.flags.DEFINE_float('epsilon', 1e-08, 'A small constant for numerical stability.')
# RMS
tf.app.flags.DEFINE_float('decay', 0.9, 'Discounting factor for the history/coming gradient')
tf.app.flags.DEFINE_float('momentum', 0.0, 'A scalar tensor')
##tf.app.flags.DEFINE_float('epsilon', 1e-10, 'Small value to avoid zero denominator')
tf.app.flags.DEFINE_boolean('centered', False, 'Gradients normalized by their estimated varience; may help, but slow')

FLAGS = tf.app.flags.FLAGS

def get_data_batcher(mat_path, img_path, data_epochs):
    return load_vrd_batcher(mat_path, FLAGS.obj_list, FLAGS.rel_list, FLAGS.batch_size,
                            data_epochs, FLAGS.which_net, img_path, FLAGS.mean_file )

def get_test_images(test_batcher):
    test_data = test_batcher.next()
    while len(test_data) == 0:
        test_batcher.next()
    images, labels, uids = test_data[0]
    return images, labels


if __name__ == '__main__':

    # Optimizer parameters
    optimizer = FLAGS.optimizer.lower()

    keeps = ['use_locking', 'learning_rate']
    param_dict = {
        'gradient': [],
        'adagrad' : ['initial_accumulator_value'],
        'adadelta': ['rho', 'epsilon'],
        'adam'    : ['beta1', 'beta2', 'epsilon'],
        'rmsprop' : ['decay', 'momentum', 'epsilon', 'centered']
    }
    keeps += param_dict[optimizer]

    removes = ['initial_accumulator_value', 'rho', 'epsilon',
               'beta1', 'beta2', 'decay', 'momentum', 'centered']

    # Dict of optimizer params
    meta_data = FLAGS.__dict__['__flags']
    opt_params = {k:meta_data[k] for k in keeps}

    # Save metadata for this run
    for r in removes: del meta_data[r]
    meta_data['optimizer'] = opt_params
    np.save(FLAGS.save_dir + 'meta_data.npy', meta_data)

    # Initialize net and feeding TF variables
    images_var = tf.placeholder('float', [FLAGS.batch_size, 224, 224, 3])
    net = CustomVgg16(FLAGS.init_path)
    net.build(images_var, train=True, output_size=FLAGS.output_size)

    ground_truth, cost, train_op = net.get_train_op(optimizer, opt_params)
    accuracy = net.get_accuracy(ground_truth)
    merged = tf.summary.merge_all()


    i = 0
    init_op = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto()) as sess:
        sess.run(init_op)
        train_writer = tf.summary.FileWriter(FLAGS.save_dir + 'summaries/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.save_dir + 'summaries/test')

        for e in trange(FLAGS.meta_epochs):

            train_batcher = get_data_batcher(FLAGS.train_mat, FLAGS.train_imgs, FLAGS.data_epochs)
            test_batcher =  get_data_batcher(FLAGS.test_mat,  FLAGS.test_imgs, 1000)

            for db, train_batch in tqdm(enumerate(train_batcher)):

                for b, (images, labels, uids) in tqdm(enumerate(train_batch)):
                    train_feed = {ground_truth: labels,
                                  images_var: images}

                    # Train and record summaries
                    summary, _ = sess.run([merged, train_op], feed_dict=train_feed)
                    train_writer.add_summary(summary, i)

                    if i % FLAGS.test_freq == 0:
                        test_images, test_labels = get_test_images(test_batcher)
                        test_feed = {ground_truth: test_labels, images_var: test_images}

                        # Test on this training batch
                        summary, train_acc = sess.run([merged, accuracy], feed_dict=train_feed)
                        train_writer.add_summary(summary, i)

                        # Test on a batch from the test set
                        summary, test_acc = sess.run([merged, accuracy], feed_dict=test_feed)
                        test_writer.add_summary(summary, i)

                        print('Step {}:{}-{}-{} | train: {}, test: {}'.format(i, e, db, b, train_acc, test_acc))
                    i += 1

            net.save_npy(sess, save_path=FLAGS.save_dir + 'vrd_trained_{}.npy'.format(e))
