from utils import load_data_batcher
from vgg16 import CustomVgg16



tf.app.flags.DEFINE_float('gpu_mem_fraction', 0.9)

tf.app.flags.DEFINE_integer('output_size', 'output_size',
    'Size of final output layer')
tf.app.flags.DEFINE_string('init_path', 'data/models/objnet/vgg16a.npy',
    'Initial weights')
tf.app.flags.DEFINE_string('save_path', 'data/models/objnet/vgg16_trained.npy',
    'Save trained weights to this file')

tf.app.flags.DEFINE_integer('batch_size',  10)
tf.app.flags.DEFINE_integer('save_freq',   10)
tf.app.flags.DEFINE_integer('meta_epochs', 20)

tf.app.flags.DEFINE_string('obj_list',  'data/vrd/objectListN.mat')
tf.app.flags.DEFINE_string('rel_list',  'data/vrd/predicate.mat')
tf.app.flags.DEFINE_string('train_mat', 'data/vrd/annotation_train.mat')
tf.app.flags.DEFINE_string('test_mat',  'data/vrd/annotation_test.mat')

FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    data_batcher = load_data_batcher(FLAGS.obj_list_path, FLAGS.rel_list_path,
                                     FLAGS.train_mat_path, FLAGS.test_mat_path,
                                     FLAGS.batch_size, FLAGS.meta_epochs, FLAGS.net)

    images_var = tf.placeholder('float', [FLAGS.batch_size, 224, 224, 3])
    net = CustomVgg16(net=FLAGS.net)
    net.build(images_var, train=True, output_size=FLAGS.output_size)

    ground_truth, cost, train_op = net.get_cost()
    merged = tf.merge_all_summaries()

    gpu_fraction = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_mem_fraction)
    session_init = lambda: tf.Session(config=tf.ConfigProto(gpu_options=(gpu_fraction)))

    with session_init() as sess:
        tf.initialize_all_variables().run()

        for mb, data_batch in enumerate(data_batcher):

            for b, (image, label) in enumerate(data_batch):
                feed_dict = {ground_truth: labels,
                             images_var: batch_images}
                sess.run([merged, train_op], feed_dict=feed_dict)

                if b % FLAGS.save_freq == 0:
                    batch_cost = sess.run(cost, feed_dict=feed_dict)
                    print '\tbatch {}-{} cost: {}'.format(mb, b, batch_cost)
                    vgg.save_npy(sess, file_path=FLAGS.save_path+'.checkpoint-{}-{}'.format(mb,b))

        vgg.save_npy(sess, file_path=FLAGS.save_path)
