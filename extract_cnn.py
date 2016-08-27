import tensorflow as tf
from utils import get_uid2imdata, batch_images
from vgg16 import CustomVgg16
from numpy import mean
import os
import sys
sys.path.append('/u/ebigelow/lib/visual_genome_python_driver/')
import local as vg


tf.app.flags.DEFINE_float('gpu_mem_fraction', 0.9, '')

tf.app.flags.DEFINE_integer('output_size', 100, '')
tf.app.flags.DEFINE_integer('batch_size',  10,  '')

tf.app.flags.DEFINE_string('weights',    'data/models/objnet/vgg16_trained.npy', 'Load weights from this file')
tf.app.flags.DEFINE_string('save_file',  'data/models/objnet/feature_dict.npy',  'Save layer output here')

tf.app.flags.DEFINE_string('json_file', 'data/vrd/json/test.json', '')
tf.app.flags.DEFINE_string('img_dir',   'data/vrd/images/train/',  '')
tf.app.flags.DEFINE_string('mean_file', 'mean.npy',                '')

FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':

    images_var = tf.placeholder('float', [1, 224, 224, 3])
    net = CustomVgg16(FLAGS.init_path)
    net.build(images_var, train=False, output_size=FLAGS.output_size)

    gpu_fraction = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_mem_fraction)
    session_init = lambda: tf.Session(config=tf.ConfigProto(gpu_options=(gpu_fraction)))

    mean = np.load(FLAGS.mean_file)

    scene_graphs = vg.GetSceneGraphsVRD(json_file=FLAGS.json_file)
    uid2imdata   = get_uid2imdata(scene_graphs)
    img_batcher  = batch_images(uid2imdata, img_dir, mean, batch_len=FLAGS.batch_size)

    layer = getattr(net, FLAGS.layer)
    feature_dict = {}

    with session_init() as sess:
        tf.initialize_all_variables().run()

        for idx2uid, batch_imgs in img_batcher:
            feed_dict = {images_var: batch_imgs}
            out = sess.run(layer, feed_dict=feed_dict)

            for idx, uid in idx2uid.items():
                feature_dict[uid] = out[idx]

    np.save(FLAGS.save_file, feature_dict)
