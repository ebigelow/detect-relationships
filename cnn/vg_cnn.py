import tensorflow as tf
import numpy as np
from tqdm import tqdm
from vgg16 import CustomVgg16

import sys; sys.path.append('..')
from utils import load_sg_batcher


data_dir = '../data/'

tf.app.flags.DEFINE_bool( 'use_gpu',          True,  '')
tf.app.flags.DEFINE_float('gpu_mem_fraction', 0.95,   '')

tf.app.flags.DEFINE_string('which_net',   'objnet',                                           'either (objnet | relnet)')
tf.app.flags.DEFINE_string('weights',     data_dir + 'models/objnet/vgg16_vg_trained_9.npy',  'Load weights from this file')
tf.app.flags.DEFINE_string('save_file',   data_dir + 'models/out/obj_probs.npy',              'Save layer output here')
# tf.app.flags.DEFINE_string('which_net',   'relnet',                                     'either (objnet | relnet)')
#tf.app.flags.DEFINE_string('weights',      'data/vg/models/relnet/vgg16_vg_trained_8.npy',   'Load weights from this file')
#tf.app.flags.DEFINE_string('save_file',    'data/vg/models/out/rel_feats.npy',               'Save layer output here')

tf.app.flags.DEFINE_integer('start_idx',   90000, '')
tf.app.flags.DEFINE_integer('end_idx',     90010, '')
tf.app.flags.DEFINE_integer('batch_size',  100,   '')
tf.app.flags.DEFINE_integer('data_epochs', 2,     '')

tf.app.flags.DEFINE_string('img_dir',     data_dir + 'images/',    ' ')
tf.app.flags.DEFINE_string('img_mean',    data_dir + 'mean.npy',   ' ')
tf.app.flags.DEFINE_string('json_dir',    data_dir + 'vg_short/',               '')
tf.app.flags.DEFINE_string('json_id_dir', data_dir + 'vg_short/by-id/',         '')
tf.app.flags.DEFINE_string('label_dict',  data_dir + 'vg_short/label_dict.npy', '')

FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':

    # Initialize GPU stuff
    gpu_fraction = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_mem_fraction)
    # session_gpu = lambda: tf.Session(config=tf.ConfigProto(gpu_options=(gpu_fraction)))
    session_gpu = lambda: tf.Session(config=tf.ConfigProto())
    session_init = session_gpu if FLAGS.use_gpu else lambda: tf.Session()


    # Set up CNN
    images_var = tf.placeholder('float', [FLAGS.batch_size, 224, 224, 3])
    output_size = 637 if FLAGS.which_net == 'objnet' else 83

    net = CustomVgg16(FLAGS.weights)
    net.build(images_var, train=False, output_size=output_size)

    ground_truth = tf.placeholder(tf.float32, shape=net.prob.get_shape())
    accuracy = net.get_accuracy(ground_truth)

    import ipdb; ipdb.set_trace()

    # Load data batcher
    data_params = {
        'data_dir':    FLAGS.json_dir,
        'data_id_dir': FLAGS.json_id_dir,
        'start_idx':   FLAGS.start_idx,
        'end_idx':     FLAGS.end_idx,
        'label_dict':  np.load(FLAGS.label_dict).item(),
        'batch_size':  FLAGS.batch_size,
        'data_epochs': FLAGS.data_epochs,
        'which_net':   FLAGS.which_net,
        'output_size': output_size,
        'img_dir':     FLAGS.img_dir,
        'img_mean':    np.load(FLAGS.img_mean)
    }
    test_batcher = load_sg_batcher(**data_params)


    # Run on images & save outputs
    outputs = {'fc7':[], 'prob':[], 'label':[], 'uid':[] }

    with session_init() as sess:
        tf.global_variables_initializer()

        for test_batch in tqdm(test_batcher):                   # loop over `data_epochs`
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
