import tensorflow as tf
import numpy as np
from tqdm import tqdm
from vgg16 import CustomVgg16

import sys; sys.path.append('..')
from utils import get_sg_data, batch_mats, rel_coords, vg


tf.app.flags.DEFINE_bool( 'use_gpu',          False,  '')
tf.app.flags.DEFINE_float('gpu_mem_fraction', 0.95,   '')

tf.app.flags.DEFINE_string('which_net',   'objnet',                                'either (objnet | relnet)')
tf.app.flags.DEFINE_string('weights',     'data/vg/models/objnet_vgg16_vg_9.npy',  'Load weights from this file')
tf.app.flags.DEFINE_string('save_file',   'data/vg/models/obj_probs.npy',          'Save layer output here')
# tf.app.flags.DEFINE_string('which_net',   'relnet',                                     'either (objnet | relnet)')
#tf.app.flags.DEFINE_string('weights',      'data/vg/models/relnet_vgg16_vg_8.npy',   'Load weights from this file')
#tf.app.flags.DEFINE_string('save_file',    'data/vg/models/rel_feats.npy',               'Save layer output here')

tf.app.flags.DEFINE_integer('start_idx',   90000, '')
tf.app.flags.DEFINE_integer('end_idx',     90100, '')
tf.app.flags.DEFINE_integer('batch_size',  2,   '')

tf.app.flags.DEFINE_string('img_dir',     'data/vg/images/', ' ')
tf.app.flags.DEFINE_string('img_mean',    'data/mean.npy',   ' ')
tf.app.flags.DEFINE_string('json_dir',    'data/vg/vg_short/',               '')
tf.app.flags.DEFINE_string('json_id_dir', 'data/vg/vg_short/by-id/',         '')
tf.app.flags.DEFINE_string('label_dict',  'data/vg/vg_short/label_dict.npy', '')

FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':

    # Initialize GPU stuff
    gpu_fraction = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_mem_fraction)
    session_gpu = lambda: tf.Session(config=tf.ConfigProto(gpu_options=(gpu_fraction)))
    session_init = session_gpu if FLAGS.use_gpu else lambda: tf.Session()

    # Load scene graph data
    scene_graphs = vg.GetSceneGraphs(FLAGS.start_idx, FLAGS.end_idx, FLAGS.json_dir, FLAGS.json_id_dir)
    scene_graphs = rel_coords(scene_graphs)

    # Load mean and index-to-word label dictionary
    mean = np.load(FLAGS.img_mean)
    label_dict = np.load(FLAGS.label_dict).item()

    # Load image data  TODO -- batcher format for less memory use
    obj_data, rel_data = get_sg_data(scene_graphs, mean, FLAGS.img_dir, label_dict)
    imdata = obj_data if FLAGS.which_net == 'objnet' else rel_data
    img_batcher = batch_mats(imdata, FLAGS.img_dir, mean, batch_len=FLAGS.batch_size)

    # Set up CNN
    images_var = tf.placeholder('float', [FLAGS.batch_size, 224, 224, 3])    
    output_size = 637 if FLAGS.which_net == 'objnet' else 83

    net = CustomVgg16(FLAGS.weights)
    net.build(images_var, train=False, output_size=output_size)

    lay = 'prob' if FLAGS.which_net == 'objnet' else 'fc7'
    layer = getattr(net, lay)

    # Run on images & save outputs (TODO: save output layer for relnet)
    with session_init() as sess:
        tf.initialize_all_variables().run()
        feature_dict = {}

        for idx2uid, batch_imgs in img_batcher:
            feed_dict = {images_var: batch_imgs}
            out = sess.run(layer, feed_dict=feed_dict)

            for idx, uid in idx2uid.items():
                feature_dict[uid] = out[idx]

    np.save(FLAGS.save_file, feature_dict)

