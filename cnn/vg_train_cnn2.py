import tensorflow as tf
import numpy as np
import sys; sys.path.append('..')
from utils import load_sg_batcher, load_vrd_batcher
from vgg16 import CustomVgg16
from tqdm import trange, tqdm

tf.app.flags.DEFINE_integer('output_size', 96, 'Size of final output layer')
tf.app.flags.DEFINE_string('which_net', 'obj', '')

tf.app.flags.DEFINE_string('init_path', 'data/models/vgg16.npy', 'Initial weights')
tf.app.flags.DEFINE_string('save_dir',  'data/mini/models/cnn/objnet_vg1/', 'Save weights to this file')
tf.app.flags.DEFINE_integer('save_freq',  500, 'Save weights every N data epochs.')
tf.app.flags.DEFINE_integer('test_freq',  20, 'Compute test accuracy every N data epochs.')

tf.app.flags.DEFINE_integer('batch_size',  25, '')
tf.app.flags.DEFINE_integer('data_epochs', 30, '')
tf.app.flags.DEFINE_integer('meta_epochs', 10, '')

tf.app.flags.DEFINE_string('label_dict', '../data/vg/vg_short/label_dict.npy', '')
tf.app.flags.DEFINE_string('json_dir',   '../data/vg/vg_short/', '')

tf.app.flags.DEFINE_string('img_dir',   '../data/vg/images/', '')
tf.app.flags.DEFINE_string('mean_file', '../data/vrd/images/mean_train.npy',      '')

tf.app.flags.DEFINE_float('learning_rate', 0.1, 'How fast do we descend?')    # 0.001

# All optimizers
tf.app.flags.DEFINE_string('optimizer', 'gradient', 'gradient | adagrad | adadelta | adam | rmsprop')
tf.app.flags.DEFINE_boolean('use_locking', False, 'If True use locks for update operations.')

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

def get_test_images(test_batcher):
    test_data = test_batcher.next()
    while len(test_data) == 0:
        test_batcher.next()
    images, labels, uids = test_data[0]
    return images, labels


if __name__ == '__main__':



    label_dict = np.load(FLAGS.label_dict).item()
    img_mean = np.load(FLAGS.mean_file)

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

            data_batcher = load_sg_batcher(FLAGS.json_dir, FLAGS.json_dir+'by-id/',
                label_dict, img_mean, 0, -1, FLAGS.batch_size, FLAGS.data_epochs,
                FLAGS.which_net, FLAGS.output_size, FLAGS.img_dir)

            for db, train_batch in tqdm(enumerate(data_batcher)):

                for b, (images, labels, uids) in tqdm(enumerate(train_batch)):
                    train_feed = {ground_truth: labels,
                                  images_var: images}

                    # Train and record summaries
                    summary, _ = sess.run([merged, train_op], feed_dict=train_feed)
                    train_writer.add_summary(summary, i)

                    if i % FLAGS.test_freq == 0:

                        test_acc = {'train': [], 'test': []}

                        test1 = load_sg_batcher(FLAGS.json_dir, FLAGS.json_dir+'by-id/',
                            label_dict, img_mean, 0, -1, FLAGS.batch_size, FLAGS.data_epochs,
                            FLAGS.which_net, FLAGS.output_size, FLAGS.img_dir)

                        Z = 5

                        # Sample Z data from train set
                        for s in range(Z):
                            test_batch = test1.next()
                            for images, labels, uids in test_batch:
                                summary, acc = sess.run([merged, accuracy], feed_dict=train_feed)
                                test_acc['train'].append(acc)

                        train_writer.add_summary(summary, i)



                        data_dir = '/home/eric/data/vrd/'
                        test2 = load_vrd_batcher(data_dir+'mat/annotation_test.mat',
                            data_dir+'mat/objectListN.mat', data_dir+'mat/predicate.mat',
                            FLAGS.batch_size, FLAGS.data_epochs, FLAGS.which_net,
                            data_dir+'images/test/', FLAGS.mean_file )

                        # Sample Z data from test set
                        for s in range(Z):
                            test_batch = test2.next()
                            for images, labels, _ in test_batch:
                                test_feed = {ground_truth : labels,  images_var : images}
                                summary, acc = sess.run([merged, accuracy], feed_dict=test_feed)
                                test_acc['test'].append(acc)
                        test_writer.add_summary(summary, i)

                        test_acc['test']  = np.mean(test_acc['test'])
                        test_acc['train'] = np.mean(test_acc['train'])

                        print('Step {}:{}-{}-{}\t | train: {}\ttest: {}'.format(
                                    i, e, db, b, test_acc['train'], test_acc['test']))

                    if i % FLAGS.save_freq == 0:
                        net.save_npy(sess, save_path=FLAGS.save_dir + 'weights_{}.npy'.format(i))
                    i += 1
