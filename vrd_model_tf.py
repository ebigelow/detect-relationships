import tensorflow as tf
import numpy as np
from utils import loadmat, mat_to_tf
from model_tf import Model
from utils import load_sg_batcher

# tf.app.flags.DEFINE_float('gpu_mem_fraction', 0.9, '')

tf.app.flags.DEFINE_integer('epochs',     20,  '')
tf.app.flags.DEFINE_integer('batch_size', 34,  '')
# [max([i.shape for i in x]) for x in zip(*train_data)]

tf.app.flags.DEFINE_float('lamb1', 0.05,  '')
tf.app.flags.DEFINE_float('lamb2', 0.001,  '')
tf.app.flags.DEFINE_float('init_noise', 0.1,  '')
tf.app.flags.DEFINE_float('learning_rate', 0.1,  '')

tf.app.flags.DEFINE_string('obj_mat',   'data/vrd/objectListN.mat',  '')
tf.app.flags.DEFINE_string('rel_mat',   'data/vrd/predicate.mat',    '')
tf.app.flags.DEFINE_string('train_mat', 'data/vrd/annotation_train.mat',  '')
tf.app.flags.DEFINE_string('test_mat',  'data/vrd/annotation_test.mat',   '')

tf.app.flags.DEFINE_string('w2v_file',  'data/vrd/w2v.npy',        '')
tf.app.flags.DEFINE_string('obj_npy',   'data/vrd/obj_probs.npy',  '')
tf.app.flags.DEFINE_string('rel_npy',   'data/vrd/rel_feats.npy',  '')

tf.app.flags.DEFINE_string('summaries_dir',  'data/vrd/summaries/',   '')
tf.app.flags.DEFINE_string('weights_file',   'data/vrd/weights',   '')

FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':

    # Load data
    # ---------

    obj_dict = {r:i for i,r in enumerate(loadmat(FLAGS.obj_mat)['objectListN'])}
    rel_dict = {r:i for i,r in enumerate(loadmat(FLAGS.rel_mat)['predicate'])}

    word2idx = {'obj':obj_dict, 'rel':rel_dict}
    w2v = np.load(FLAGS.w2v_file).item()

    obj_probs = np.load(FLAGS.obj_npy).item()
    rel_feats = np.load(FLAGS.rel_npy).item()

    train_mat = loadmat(FLAGS.train_mat)['annotation_train']
    test_mat  = loadmat(FLAGS.test_mat)['annotation_test']
    # D = mat_to_triplets(mat, word2idx)
    train_data = mat_to_tf(train_mat, word2idx, obj_probs, rel_feats)
    t = 30
    test_data  = [i[:t] for i in train_data]
    train_data = [i[t:] for i in train_data]
    # TODO!
    # test_data  = mat_to_tf(test_mat,  word2idx, obj_probs, rel_feats, FLAGS.batch_size)


    #
    # TODO everthing below here could be in a separate function!!!  (VRD vs. VG run files)
    # maybe... model_train(w2v, R_full, train_data, test_data, epochs,
    #                      batch_size, lamb1, lamb2, init_noise, learning_rate)

    # Setup data for inner sum over all data in L (equation 5)
    I2 = np.concatenate(train_data[0])
    J2 = np.concatenate(train_data[1])
    K2 = np.concatenate(train_data[2])
    n_rels = I2.shape[0]

    test_data  = zip(*test_data)
    train_data = zip(*train_data)

    # Set up model
    # ------------

    model = Model(w2v, lamb1=FLAGS.lamb1, lamb2=FLAGS.lamb2, init_noise=FLAGS.init_noise)
    ground_truth = model.get_ground_truth(FLAGS.batch_size, n_rels)
    #accuracy = model.get_accuracy(ground_truth)

    cost = model.loss(ground_truth)
    with tf.variable_scope('train_op'):
        train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cost)
    #train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Convert data to feed dict format
    def batch_to_feed(I, J, K, op, rf):
        return {  \
            ground_truth['I']: np.squeeze(I),
            ground_truth['J']: np.squeeze(J),
            ground_truth['K']: np.squeeze(K),
            ground_truth['I_full']: np.squeeze(I2),
            ground_truth['J_full']: np.squeeze(J2),
            ground_truth['K_full']: np.squeeze(K2),
            ground_truth['obj_probs']: op,
            ground_truth['rel_feats']: rf
        }

    # gpu_fraction = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_mem_fraction)
    # session_init = lambda: tf.Session(config=tf.ConfigProto(gpu_options=(gpu_fraction)))

    # Train model
    # -----------

    print '##### Begin Training!'
    # with session_init() as sess:
    with tf.Session() as sess:
        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
        tf.initialize_all_variables().run()

        for e in range(FLAGS.epochs):
            print 'Beginning epoch {}'.format(e)

            for db, data_batch in enumerate(train_data):
                print '~~~~~~~~~~~~ DATA BATCH {}  | {}'.format(db, data_batch[0].shape)
                if db % 100 == 0:
                    save_freq = 200    # TODO
                    if db % save_freq == 0:
                        model.save_weights(sess, file_path=FLAGS.weights_file + '_{}-{}.npy'.format(e,db))
                    #     summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
                    #     test_writer.add_summary(summary, i)
                    #     accs = []
                    #     for test_batch in test_data:
                    #         feed_test = batch_to_feed(*test_batch)
                    #         accs.append(sess.run(accuracy, feed_dict=feed_test))
                    #     acc = np.mean(accs)
                    #     print ' => epoch {} batch {}  |  Acurracy: {}'.format(e, db, acc)
                    #     if acc > best_acc:
                    feed_train = batch_to_feed(*data_batch)

                    # Record train set summaries, and train
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, train_op], feed_dict=feed_train,
                                          options=run_options, run_metadata=run_metadata)
                    # TODO look up docs ...
                    train_writer.add_run_metadata(run_metadata, 'epoch {}, batch {}'.format(e, db))
                    print('Adding run metadata for', db)
                    # print '==> DATA BATCH {}, COST {}'.format(db, sess.run(cost, feed_dict=feed_train))
                else:
                    # Record Train
                    feed_train = batch_to_feed(*data_batch)
                    summary, _ = sess.run([merged, train_op], feed_dict=feed_train)

                train_writer.add_summary(summary, db)


        train_writer.close()
        test_writer.close()
