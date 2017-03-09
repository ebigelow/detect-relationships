
import tensorflow as tf
import numpy as np
import sys; sys.path.append('..')
from utils import load_mini_batcher, flatten, batchify_data
from vgg16 import CustomVgg16
from tqdm import trange, tqdm
from collections import defaultdict


tf.app.flags.DEFINE_integer('output_size', 100, 'Size of final output layer')
tf.app.flags.DEFINE_string('which_net',  'obj', 'obj | rel')

tf.app.flags.DEFINE_string('init_path', 'data/models/vgg16.npy', 'Initial weights')
tf.app.flags.DEFINE_string('save_dir',  'data/models/objnet/run1/', 'Save weights to this file')

tf.app.flags.DEFINE_integer('save_freq',    1000, 'Save every N batches.')
tf.app.flags.DEFINE_integer('test_freq',    10, 'Compute test accuracy every N batches.')
tf.app.flags.DEFINE_integer('test_samples', 100, 'Sample this many batches for testing.')

tf.app.flags.DEFINE_integer('rel_cap',     2000, 'limit the number of training data for each relation label to this.')
tf.app.flags.DEFINE_integer('batch_size',  10,   '')
tf.app.flags.DEFINE_integer('data_epochs', 20,   '')
tf.app.flags.DEFINE_integer('meta_epochs', 20,   '')

tf.app.flags.DEFINE_string('train_data', 'data/mini/vg_train.npy', '')
tf.app.flags.DEFINE_string('test_data',  'data/mini/vg_test.npy',  '')

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


def tenumerate(ls):
    return enumerate(tqdm(ls))


def sample_data(data, num_samples):
    idxs = np.random.choice(range(len(data)), num_samples, replace=False)
    return [data[i] for i in idxs]

def cap_data(data, cap=2000):
    """
    TODO: describe

    """
    D = defaultdict(lambda: list())

    for d in data:
        label = d[2]
        D[label].append(d)

    for k, d in D.items():
        if len(d) > cap:
            idxs = np.random.choice(range(len(d)), cap, replace=False)
            D[k] = [d[i] for i in idxs]
        else:
            D[k] = d

    return flatten(D.values())




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

    meta_data['optimizer'] = opt_params
    for r in removes: del meta_data[r]

    # Save metadata for this run
    np.save(FLAGS.save_dir + 'meta_data.npy', meta_data)

    # Initialize net and feeding TF variables
    images_var = tf.placeholder('float', [FLAGS.batch_size, 224, 224, 3])
    net = CustomVgg16(FLAGS.init_path)
    net.build(images_var, train=True, output_size=FLAGS.output_size)

    ground_truth, cost, train_op = net.get_train_op(optimizer, opt_params)
    accuracy = net.get_accuracy(ground_truth)
    merged = tf.summary.merge_all()

    img_mean = np.load(FLAGS.mean_file)

    train_data = np.load(FLAGS.train_data).item()
    test_data  = np.load(FLAGS.test_data).item()

    load_batcher = lambda td, im: load_mini_batcher(td[FLAGS.which_net], img_mean,
        FLAGS.batch_size, FLAGS.data_epochs, FLAGS.output_size, im)

    i = 0
    init_op = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto()) as sess:
        sess.run(init_op)
        train_writer = tf.summary.FileWriter(FLAGS.save_dir + 'summaries/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.save_dir + 'summaries/test')

        for e in trange(FLAGS.meta_epochs):
            D_train = {'obj' : cap_data(train_data['obj'], FLAGS.rel_cap),
                       'rel' : cap_data(train_data['rel'], FLAGS.rel_cap) }
            D_test  = {'obj' : cap_data(test_data['obj'], FLAGS.rel_cap/4),
                       'rel' : cap_data(test_data['rel'], FLAGS.rel_cap/4) }

            train_batcher = load_batcher(D_train, FLAGS.train_imgs)

            for db, D_batch in enumerate(train_batcher):
                for b, (images, labels, uids) in tenumerate(D_batch):

                    # Train and record summaries
                    train_feed = {ground_truth : labels,  images_var : images}
                    summary, _ = sess.run([merged, train_op], feed_dict=train_feed)
                    train_writer.add_summary(summary, i)

                    # Test and record accuracies
                    test_acc = {'train': 0.0, 'test': 0.0}

                    if i % FLAGS.test_freq == 0:
                        s = FLAGS.test_samples

                        summary, acc_ = sess.run([merged, accuracy], feed_dict=train_feed)
                        # train_writer.add_summary(summary, i)

                        # Sample test data from training set
                        test1 = sample_data(D_train[FLAGS.which_net], s)
                        test1 = batchify_data(test1, img_mean, FLAGS.batch_size,
                                                FLAGS.train_imgs, FLAGS.output_size)
                        for images, labels, _ in tqdm(test1):
                            summary, acc = sess.run([merged, accuracy], feed_dict=train_feed)
                            test_acc['train'] += acc / s
                        train_writer.add_summary(summary, i)

                        # Sample test data from test set
                        test2 = sample_data(D_test[FLAGS.which_net], s)
                        test2 = batchify_data(test2, img_mean, FLAGS.batch_size,
                                                FLAGS.test_imgs, FLAGS.output_size)
                        for images, labels, _ in tqdm(test2):
                            test_feed = {ground_truth : labels,  images_var : images}
                            summary, acc = sess.run([merged, accuracy], feed_dict=test_feed)
                            test_acc['test'] += acc / s
                        test_writer.add_summary(summary, i)

                        print('Step {}:{}-{}-{}\t| train: {}\ttest: {}\tsimple: {}'.format(
                                    i, e, db, b, test_acc['train'], test_acc['test'], acc_  ))

                    if i % FLAGS.save_freq == 0:
                        net.save_npy(sess, save_path=FLAGS.save_dir + 'vgg16_{}.npy'.format(i))
                    i += 1
