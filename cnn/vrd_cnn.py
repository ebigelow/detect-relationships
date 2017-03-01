from utils import loadmat, mat_to_imdata, batch_mats
from vgg16 import CustomVgg16
import os
import sys
from tqdm import tqdm
import numpy as np
import tensorflow as tf


tf.app.flags.DEFINE_float('gpu_mem_fraction', 0.9, '')

tf.app.flags.DEFINE_string('which_net',    'relnet', '')
tf.app.flags.DEFINE_integer('output_size', 70,      '')
#tf.app.flags.DEFINE_string('which_net',    'objnet', '')
#tf.app.flags.DEFINE_integer('output_size', 100,      '')
tf.app.flags.DEFINE_integer('batch_size',  10,       '')

tf.app.flags.DEFINE_string('weights',   'data/relnet_vrd.npy', 'Load weights from this file')
tf.app.flags.DEFINE_string('save_file', 'data/rel_feats-vrd_test.npy',  'Save layer output here')
#tf.app.flags.DEFINE_string('weights',   'data/objnet_vrd.npy', 'Load weights from this file')
#tf.app.flags.DEFINE_string('save_file', 'data/obj_probs-vrd_test.npy',  'Save layer output here')
tf.app.flags.DEFINE_string('layer',     'fc7', '')

# tf.app.flags.DEFINE_string('obj_list', 'data/vrd/objectListN.mat', '')
# tf.app.flags.DEFINE_string('rel_list', 'data/vrd/predicate.mat',   '')

tf.app.flags.DEFINE_string('mat_file', 'data/vrd/annotation_test.mat', '')
tf.app.flags.DEFINE_string('img_dir',  'data/vrd/images/test/',        '')

tf.app.flags.DEFINE_string('mean_file', 'mean.npy', '')

FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':

    #gpu_fraction = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_mem_fraction)
    #session_init = lambda: tf.Session(config=tf.ConfigProto(gpu_options=(gpu_fraction)))
    session_init = lambda: tf.Session(config=tf.ConfigProto())

    mat                = loadmat(FLAGS.mat_file)[FLAGS.mat_file.split('/')[-1].split('.')[0]]
    obj_data, rel_data = mat_to_imdata(mat)
    imdata             = obj_data if FLAGS.which_net == 'objnet' else rel_data

    mean = np.load(FLAGS.mean_file)
    img_batcher = batch_mats(imdata, FLAGS.img_dir, mean, batch_len=FLAGS.batch_size)

    images_var = tf.placeholder('float', [FLAGS.batch_size, 224, 224, 3])
    net = CustomVgg16(FLAGS.weights)
    net.build(images_var, train=False, output_size=FLAGS.output_size)

    ground_truth = tf.placeholder(tf.float32, shape=net.prob.get_shape())
    accuracy = net.get_accuracy(ground_truth)
    layer = getattr(net, FLAGS.layer)

    feature_dict = {}
    outputs = {'fc7':[], 'prob':[], 'label':[], 'uid':[] }

    with session_init() as sess:
        tf.initialize_all_variables().run()

        for idx2uid, batch_imgs in tqdm(img_batcher):
            batch_out = {'fc7':[], 'prob':[], 'label':[], 'uid':[] }

            for uids, images, labels in tqdm(test_batch):       # loop over `batch_size`
                b_fc7, b_prob, b_acc = sess.run([net.fc7, net.prob, accuracy],
                                                feed_dict={ground_truth: labels, images_var: images})
                for q in range(len(uids)):
                    batch_out['fc7'].append(b_fc7[q])
                    batch_out['prob'].append(b_prob[q])
                    batch_out['label'].append(labels[q])
                    batch_out['uid'].append(uids[q])

            for key, item in batch_out.items():
                if key == 'uid':
                    outputs[key].append(item)
                else:
                    outputs[key].append(np.vstack(item))

    np.save(FLAGS.save_file, outputs)
    print '\nFile saved to:{}\n'.format(FLAGS.save_file)
