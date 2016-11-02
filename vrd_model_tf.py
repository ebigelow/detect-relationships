import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
from utils import loadmat, mat_to_tf
from model_tf import Model
from utils import load_sg_batcher
from tqdm import tqdm, trange

tf.app.flags.DEFINE_float('gpu_mem_fraction', 0.9, '')

tf.app.flags.DEFINE_integer('epochs',     20,  '')
tf.app.flags.DEFINE_integer('batch_size', 41,  '')
tf.app.flags.DEFINE_integer('save_freq',  2000, '')
# tf.app.flags.DEFINE_integer('top_k',  200, '')
# [max([i.shape for i in x]) for x in zip(*train_data)]

tf.app.flags.DEFINE_float('lamb1', 0.05,  '')
tf.app.flags.DEFINE_float('lamb2', 0.001,  '')
tf.app.flags.DEFINE_float('init_noise', 1e-5,  '')
tf.app.flags.DEFINE_float('learning_rate', 0.1,  '')

tf.app.flags.DEFINE_string('obj_mat',   'data/vrd/objectListN.mat',  '')
tf.app.flags.DEFINE_string('rel_mat',   'data/vrd/predicate.mat',    '')
tf.app.flags.DEFINE_string('train_mat', 'data/vrd/annotation_train.mat',  '')
tf.app.flags.DEFINE_string('test_mat',  'data/vrd/annotation_test.mat',   '')

tf.app.flags.DEFINE_string('w2v_file', 'data/vrd/w2v.npy',        '')
tf.app.flags.DEFINE_string('obj_npy',  'data/vrd/obj_probs.npy',  '')
tf.app.flags.DEFINE_string('rel_npy',  'data/vrd/rel_feats.npy',  '')
tf.app.flags.DEFINE_string('obj_npy_test', 'data/vrd/obj_probs-vrd_test.npy',  '')
tf.app.flags.DEFINE_string('rel_npy_test', 'data/vrd/rel_feats-vrd_test.npy',  '')

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
    train_mat  = loadmat(FLAGS.train_mat)['annotation_train']
    train_data = mat_to_tf(train_mat, word2idx, obj_probs, rel_feats, batch_size=FLAGS.batch_size)

    obj_probs_test = np.load(FLAGS.obj_npy_test).item()
    rel_feats_test = np.load(FLAGS.rel_npy_test).item()
    test_mat  = loadmat(FLAGS.test_mat)['annotation_test']
    test_data = mat_to_tf(test_mat, word2idx, obj_probs_test, rel_feats_test, batch_size=FLAGS.batch_size)

    # TODO everthing below here could be in a separate function  (VRD vs. VG run files)
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
    accuracies = [model.compute_accuracy(ground_truth, top_k=top_k) for top_k in [1,5,10,20]]
    # accuracy = model.compute_accuracy(ground_truth, top_k=10)
    best_acc = 0.0

    cost = model.loss(ground_truth)
    # with tf.variable_scope('train_op'):
    #     train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cost)

    optimizer    = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    train_vars_1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'lang-module')
    train_vars_2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'vis-module')
    train_op_1   = optimizer.minimize(cost, var_list=train_vars_1)
    train_op_2   = optimizer.minimize(cost, var_list=train_vars_2)

    # Convert data to feed dict format
    def batch_to_feed(I, J, K, op, rf, rids):
        return {  \
            ground_truth['I']: np.squeeze(I),
            ground_truth['J']: np.squeeze(J),
            ground_truth['K']: np.squeeze(K),
            ground_truth['I_full']: np.squeeze(I2),
            ground_truth['J_full']: np.squeeze(J2),
            ground_truth['K_full']: np.squeeze(K2),
            ground_truth['obj_probs']: op,
            ground_truth['rel_feats']: rf,
            ground_truth['rel_ids']: rids,
        }


    gpu_fraction = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_mem_fraction)
    session_init = lambda: tf.Session(config=tf.ConfigProto(gpu_options=(gpu_fraction)))

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
            train_op = train_op_1 if (e % 2 == 0) else train_op_2

            for db, data_batch in tqdm(enumerate(train_data)):
                print '~~~~~~~~~~~~ DATA BATCH {}  | {}'.format(db, data_batch[0].shape)

                # Testing
                # -------
                if db % FLAGS.save_freq == 0:
                    accs = []
                    for test_batch in test_data:
                        feed_test = batch_to_feed(*test_batch)
                        accs_ = [sess.run(a, feed_dict=feed_test) for a in accuracies]
                        # sums_, accs_ = zip(*[sess.run([merged, a], feed_dict=feed_test) for a in accuracies])
                        accs.append(accs_)
                        # accs.append(sess.run(accuracy, feed_dict=feed_test))

                    # # TODO -- save accuracy summaries
                    # test_writer.add_summary(sums_[2], db+(e*FLAGS.epochs))

                    final_accs = [np.mean(a) for a in zip(*accs)]
                    print '-> epoch {} batch {}\n\tAcurracies: 1:{}, 5:{}, 10:{}, 20:{}'.format(e, db, *final_accs)
                    # print '-> epoch {} batch {}\n\tAcurracy 20:{}'.format(e, db, np.mean(accs))
                    model.save_weights(sess, file_path=FLAGS.weights_file + '_{}-{}.npy'.format(e,db))

                # Training
                # --------
                if db % 100 == 0:
                    feed_train = batch_to_feed(*data_batch)

                    # Record train set summaries, and train
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, train_op], feed_dict=feed_train,
                                          options=run_options, run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'epoch {}, batch {}'.format(e, db))

                    # Create the Timeline object, and write it to a json
                    if db == 0 and e == 0:
                        tl = timeline.Timeline(run_metadata.step_stats)
                        ctf = tl.generate_chrome_trace_format()
                        with open('data/vrd/summaries/timeline.json', 'w') as f:
                            f.write(ctf)
                    print('Adding run metadata for', db)
                    # print '==> DATA BATCH {}, COST {}'.format(db, sess.run(cost, feed_dict=feed_train))
                else:
                    # Record Train
                    feed_train = batch_to_feed(*data_batch)
                    summary, _ = sess.run([merged, train_op], feed_dict=feed_train)

                train_writer.add_summary(summary, db+(e*FLAGS.epochs))


        train_writer.close()
        test_writer.close()
