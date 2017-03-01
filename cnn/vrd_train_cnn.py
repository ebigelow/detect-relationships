import tensorflow as tf
import sys; sys.path.append('..')
from utils import load_data_batcher
from vgg16 import CustomVgg16
from tqdm import trange, tqdm

# tf.app.flags.DEFINE_float('gpu_mem_fraction', 0.9, '')

tf.app.flags.DEFINE_integer('output_size', 100, 'Size of final output layer')
tf.app.flags.DEFINE_string('which_net', 'objnet', '')

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
tf.app.flags.DEFINE_string('mean_file',  'data/vrd/images/mean.npy',      '')

FLAGS = tf.app.flags.FLAGS

def get_data_batcher(mat_path, img_path):
    return load_data_batcher(mat_path, FLAGS.obj_list, FLAGS.rel_list, FLAGS.batch_size,
                            FLAGS.data_epochs, FLAGS.which_net, img_path, FLAGS.mean_file )

def get_test_images(test_batcher, N_test):
    images, labels = test_batcher.next()[:N_test]
    return images, labels


if __name__ == '__main__':

    images_var = tf.placeholder('float', [FLAGS.batch_size, 224, 224, 3])
    net = CustomVgg16(FLAGS.init_path)
    net.build(images_var, train=True, output_size=FLAGS.output_size)

    ground_truth, cost, train_op = net.get_train_op()
    accuracy = net.get_accuracy(ground_truth)
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(FLAGS.save_dir + 'summaries/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.save_dir + 'summaries/test')
        tf.global_variables_initializer()

        for e in trange(FLAGS.meta_epochs):

            train_batcher = get_data_batcher(FLAGS.train_mat, FLAGS.train_imgs)
            test_batcher =  get_data_batcher(FLAGS.test_mat,  FLAGS.test_imgs)

            for db, train_batch in tqdm(enumerate(train_batcher)):

                for b, (images, labels) in enumerate(train_batch):
                    train_feed = {ground_truth: labels,
                                  images_var: images}

                    # Record train set summaries, and train
                    summary, _ = sess.run([merged, train_op], feed_dict=train_feed)
                    train_writer.add_summary(summary, '{}:{}:{}'.format(e, db, b))

                    # Record summaries and test-set accuracy
                    if (db % FLAGS.summaries_freq) and (b == 0) == 0:
                        test_labels, test_images = get_test_images(test_batcher, N_test=1000)
                        test_feed = {ground_truth: test_labels, images_var: test_images}

                        summary, acc = sess.run([merged, accuracy], feed_dict=test_feed)
                        test_writer.add_summary(summary, db)
                        print('Accuracy at step %s: %s' % (db, acc))

            net.save_npy(sess, file_path=FLAGS.save_dir + 'vrd_trained_{}.npy'.format(e))
