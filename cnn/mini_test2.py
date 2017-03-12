
import tensorflow as tf
import numpy as np
import sys; sys.path.append('..')
from utils import load_mini_batcher
from vgg16 import CustomVgg16
from tqdm import tqdm


tf.app.flags.DEFINE_integer('output_size', 96,    'Size of final output layer')
tf.app.flags.DEFINE_string('which_net',    'obj', 'obj | rel')

tf.app.flags.DEFINE_string('weights',   'data/mini/obj_vrd1/vgg16_vrd_5000.npy', 'Initial weights')
tf.app.flags.DEFINE_string('save_file', 'data/mini/obj_vrd1/npy/TODO.npy',                   'Save weights to this file')

tf.app.flags.DEFINE_integer('batch_size',   10,  '')
tf.app.flags.DEFINE_integer('data_epochs',  20,  '')

tf.app.flags.DEFINE_string('data_file',  'data/mini/vrd_test.npy',         '')
tf.app.flags.DEFINE_string('img_dir',    'data/vrd/images/test/',          '')
tf.app.flags.DEFINE_string('mean_file',  'data/vrd/images/mean_train.npy', '')


FLAGS = tf.app.flags.FLAGS




if __name__ == '__main__':

    # Load data
    img_mean = np.load(FLAGS.mean_file)
    data = np.load(FLAGS.data_file).item()

    data_batcher = load_mini_batcher(data[FLAGS.which_net], img_mean,
        FLAGS.batch_size, FLAGS.data_epochs, FLAGS.output_size, FLAGS.img_dir)

    # Set up TF variables and CNN
    images_var = tf.placeholder('float', [FLAGS.batch_size, 224, 224, 3])
    net = CustomVgg16(FLAGS.weights)
    net.build(images_var, train=False, output_size=FLAGS.output_size)

    ground_truth = tf.placeholder(tf.float32, shape=net.prob.get_shape())
    accuracy = net.get_accuracy(ground_truth)

    # Initilize and go
    outputs = {}
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        for data_batch in tqdm(data_batcher):
            for images, labels, uids in tqdm(data_batch):
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
