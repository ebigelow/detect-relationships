# from utils import load_sg_batcher
import tensorflow as tf
from vgg16 import CustomVgg16
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/localdisk/ebigelow/lib/visual_genome_python_driver')
import src.local as vg


tf.app.flags.DEFINE_float('gpu_mem_fraction', 0.95, '')

tf.app.flags.DEFINE_string('which_net',   'objnet',                                     'either (objnet | relnet)')
tf.app.flags.DEFINE_string('weights',     'data/vg/models/objnet_vgg16_vg_9.npy.npy',   'Load weights from this file')
tf.app.flags.DEFINE_string('save_file',   'data/vg/models/obj_probs.npy',               'Save layer output here')
#tf.app.flags.DEFINE_string('weights',     'data/vg/models/relnet_vgg16_vg_8.npy.npy',   'Load weights from this file')
#tf.app.flags.DEFINE_string('save_file',   'data/vg/models/rel_feats.npy',               'Save layer output here')

tf.app.flags.DEFINE_integer('train_idx',   90000, '')

tf.app.flags.DEFINE_string('img_dir',     'data/vg/images/')
tf.app.flags.DEFINE_string('img_mean',    'mean.npy')
tf.app.flags.DEFINE_string('label_dict',  'data/vrd/word2idx.npy')
tf.app.flags.DEFINE_string('json_dir',    'data/vg/json/vg_short/')
tf.app.flags.DEFINE_string('json_id_dir', 'data/vg/json/vg_short/by-id/')

FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':

    test_params = {
        'data_dir':    FLAGS.json_dir,
        'data_id_dir': FLAGS.json_id_dir,
        'start_idx':   FLAGS.train_idx,
        'end_idx':     FLAGS.train_idx + 1000,
        'label_dict':  np.load(FLAGS.label_dict).item(),
        'batch_size':  FLAGS.batch_size,
        'data_epochs': FLAGS.data_epochs,
        'which_net':   FLAGS.which_net,
        'img_dir':     FLAGS.img_dir,
        'img_mean':    np.load(FLAGS.img_mean)
    }

    images_var = tf.placeholder('float', [FLAGS.batch_size, 224, 224, 3])
    net = CustomVgg16(FLAGS.weights)
    net.build(images_var, train=False, output_size=FLAGS.output_size)

    gpu_fraction = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_mem_fraction)
    session_init = lambda: tf.Session(config=tf.ConfigProto(gpu_options=(gpu_fraction)))

    mat                = loadmat(FLAGS.mat_file)[FLAGS.mat_file.split('/')[-1].split('.')[0]]
    obj_data, rel_data = mat_to_imdata(mat)
    imdata             = obj_data if FLAGS.which_net == 'objnet' else rel_data

    mean        = np.load(FLAGS.mean_file)
    img_batcher = batch_mats(imdata, FLAGS.img_dir, mean, batch_len=FLAGS.batch_size)

    layer = getattr(net, FLAGS.layer)
    feature_dict = {}

    with session_init() as sess:
        tf.initialize_all_variables().run()
        test_batcher = load_sg_batcher(**test_params)

        for idx2uid, batch_imgs in tqdm(test_batcher):
            feed_dict = {images_var: batch_imgs}
            out = sess.run(layer, feed_dict=feed_dict)

            for idx, uid in idx2uid.items():
                feature_dict[uid] = out[idx]

    np.save(FLAGS.save_file, feature_dict)

