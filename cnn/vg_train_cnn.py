import tensorflow as tf
import numpy as np
from tqdm import tqdm

import sys; sys.path.append('..')
from utils import load_sg_batcher

from vgg16 import CustomVgg16

tf.app.flags.DEFINE_float('gpu_mem_fraction', 0.95, '')

tf.app.flags.DEFINE_string('which_net',   'objnet', 'either (objnet | relnet)')
tf.app.flags.DEFINE_string('init_path',   'data/models/vgg16.npy', 'Initial weights')
tf.app.flags.DEFINE_string('save_path',   'data/models/objnet/vgg16_vg_trained.npy', 'Save weights to this file')
tf.app.flags.DEFINE_string('upload_path', 'detect-relationships/models/objnet/vgg16_vg_trained_{}.npy', 'Save weights to this file')

tf.app.flags.DEFINE_float('learning_rate', 0.01,   '')
tf.app.flags.DEFINE_integer('batch_size',  10,    '')
tf.app.flags.DEFINE_integer('data_epochs', 20,    '')
tf.app.flags.DEFINE_integer('meta_epochs', 20,    '')
tf.app.flags.DEFINE_integer('train_idx',   90000, '')

tf.app.flags.DEFINE_string('img_dir',     'data/vg/images/', '')
tf.app.flags.DEFINE_string('img_mean',    'mean.npy',        '')
tf.app.flags.DEFINE_string('label_dict',  'data/vg/json/vg_short/label_dict.npy', '')
tf.app.flags.DEFINE_string('json_dir',    'data/vg/json/vg_short/',         '')
tf.app.flags.DEFINE_string('json_id_dir', 'data/vg/json/vg_short/by-id/',   '')

FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':

    label_dict = np.load(FLAGS.label_dict).item()
    n, k = (len(label_dict['obj']), len(label_dict['rel']))

    images_var = tf.placeholder('float', [FLAGS.batch_size, 224, 224, 3])
    net = CustomVgg16(FLAGS.init_path)

    output_size = n if FLAGS.which_net == 'objnet' else k
    net.build(images_var, train=True, output_size=output_size)

    ground_truth, cost, train_op = net.get_train_op(learning_rate=FLAGS.learning_rate)
    accuracy = net.get_accuracy(ground_truth)

    best_acc = 0.0

    train_params = {
        'data_dir':    FLAGS.json_dir,
        'data_id_dir': FLAGS.json_id_dir,
        'start_idx':   0,
        'end_idx':     FLAGS.train_idx,
        'label_dict':  label_dict,
        'batch_size':  FLAGS.batch_size,
        'data_epochs': FLAGS.data_epochs,
        'which_net':   FLAGS.which_net,
        'img_dir':     FLAGS.img_dir,
        'img_mean':    np.load(FLAGS.img_mean)
    }
    test_params = train_params.copy()
    test_params['start_idx'] = FLAGS.train_idx
    test_params['end_idx']   = FLAGS.train_idx + 1000

    gpu_fraction = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_mem_fraction)
    session_init = lambda: tf.Session(config=tf.ConfigProto(gpu_options=(gpu_fraction)))

    gpu_fraction = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_mem_fraction)
    session_init = lambda: tf.Session(config=tf.ConfigProto(gpu_options=(gpu_fraction)))

    with session_init() as sess:
        tf.initialize_all_variables().run()

        for e in range(FLAGS.meta_epochs):
            print 'Beginning epoch {}'.format(e)
            data_batcher = load_sg_batcher(**train_params)
            for db, data_batch in tqdm(enumerate(data_batcher)):

                for b, (images, labels) in tqdm(enumerate(data_batch)):
                    feed_dict = {ground_truth: labels, images_var: images}
                    sess.run(train_op, feed_dict=feed_dict)

            print '\tBegin testing!'
            test_batcher = load_sg_batcher(**test_params)
            accs = []
            for test_batch in test_batcher:
                for images, labels in test_batch:
                    feed_dict = {ground_truth: labels, images_var: images}
                    batch_acc = sess.run(accuracy, feed_dict=feed_dict)
                    accs.append(batch_acc)

            acc = np.mean(accs)
            print ' => epoch {} acurracy: {}'.format(db, acc)
            if acc > best_acc:
                best_acc = acc
                net.save_npy(sess, save_path=FLAGS.save_path, upload_path=FLAGS.upload_path)
