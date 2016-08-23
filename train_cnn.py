import tensorflow as tf
from utils import load_data_batcher
from vgg16 import CustomVgg16
from numpy import mean




tf.app.flags.DEFINE_float('gpu_mem_fraction', 0.9, '')

tf.app.flags.DEFINE_integer('output_size', 100, 'Size of final output layer')
tf.app.flags.DEFINE_string('which_net', 'objnet', '')

tf.app.flags.DEFINE_string('init_path', 'data/models/vgg16.npy', 'Initial weights')
tf.app.flags.DEFINE_string('save_path', 'data/models/objnet/vgg16_trained.npy', 'Save weights to this file')

tf.app.flags.DEFINE_integer('batch_size',  10, '')
tf.app.flags.DEFINE_integer('save_freq',   1, '')
tf.app.flags.DEFINE_integer('meta_epochs', 20, '')

tf.app.flags.DEFINE_string('obj_list',  'data/vrd/objectListN.mat',      '')
tf.app.flags.DEFINE_string('rel_list',  'data/vrd/predicate.mat',        '')
tf.app.flags.DEFINE_string('train_mat', 'data/vrd/annotation_train.mat', '')
tf.app.flags.DEFINE_string('test_mat',  'data/vrd/annotation_test.mat',  '')

FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    data_batcher = load_data_batcher(FLAGS.train_mat, FLAGS.obj_list, FLAGS.rel_list, 
                                     FLAGS.batch_size, FLAGS.meta_epochs, 'data/vrd/images/train/', FLAGS.which_net)
    test_batcher = load_data_batcher(FLAGS.test_mat, FLAGS.obj_list, FLAGS.rel_list,
                                     FLAGS.batch_size, 1, 'data/vrd/images/test/', FLAGS.which_net).next()

    images_var = tf.placeholder('float', [FLAGS.batch_size, 224, 224, 3])
    net = CustomVgg16(FLAGS.init_path)
    net.build(images_var, train=True, output_size=FLAGS.output_size)

    ground_truth, cost, train_op = net.get_train_op()
    accuracy = net.get_accuracy(ground_truth)
    #merged = tf.merge_all_summaries()

    gpu_fraction = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_mem_fraction)
    session_init = lambda: tf.Session(config=tf.ConfigProto(gpu_options=(gpu_fraction)))

    with session_init() as sess:
        tf.initialize_all_variables().run()

        for mb, data_batch in enumerate(data_batcher):

            for b, (images, labels) in enumerate(data_batch):
                feed_dict = {ground_truth: labels,
                             images_var: images}
                sess.run(train_op, feed_dict=feed_dict)
                #sess.run([merged, train_op], feed_dict=feed_dict)

            if mb % FLAGS.save_freq == 0:
                accs = []
                for test_images, test_labels in test_batcher:
                    feed_dict = {ground_truth: test_labels,
                                 images_var: test_images}
                    batch_acc = sess.run(accuracy, feed_dict=feed_dict)
                    accs.append(batch_acc)

                print '\tmbatch {} acurracy: {}'.format(mb, mean(accs))
                # net.save_npy(sess, file_path=FLAGS.save_path+'.checkpoint-{}-{}'.format(mb,b))

        net.save_npy(sess, file_path=FLAGS.save_path)
