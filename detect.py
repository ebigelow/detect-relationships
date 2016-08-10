import sys, os
from cv2 import imread
from utils import square_crop, rel_coords
import numpy as np
import tensorflow as tf
from numpy.random import randint
from scipy.spatial.distance import cosine
import pickle
sys.path.append('/u/ebigelow/lib/caffe-tensorflow')



def id_(a, b): 
    return str(a) + '_' + str(b)


class ConvNets:
    """
    1. load the CNN models
    2. process all relationships and objects
    3. load & crop images to suit
    4. run in batches, agglomerate dictionary of probability-level activation layer
    5. close TF session when done

    TODO
    ----
    - in `parse_scenes`, do something more complicated to select 
      certain scene graphs/relationships (or just pre-process these ...)

    
    Parameters
    ----------
    cnn_classes : path to python classes output from caffe-tensorflow ;
                  should have `RelationNet` and `ObjectNet` as class names
    obj_weights : path to numpy weights file for ObjectNet
    rel_weights : path to numpy weights file for RelationNet

    obj_dir : directory with class.py, weights.npy, mean.npy

    batch_size : process this many images at a time
    crop_size  : crop images to square with sides of this length


    """

    def __init__(self, obj_dir, rel_dir, img_path,
                 batch_size=10, crop_size=224):
        self.obj_dir    = obj_dir
        self.rel_dir    = rel_dir
        self.img_path   = img_path
        self.batch_size = batch_size
        self.crop_size  = crop_size

    def load_cnn(self, cnn_dir, new_layer=None, train=False):
        tf.reset_default_graph()

        batch_size, crop_size = self.batch_size, self.crop_size
        if train:
            images_var = tf.placeholder(tf.float32, [batch_size, crop_size, crop_size, 3])
            images_batch = tf.reshape(images_var, [-1, crop_size, crop_size, 3])
        else:
            images_var = tf.placeholder(tf.float32, [crop_size, crop_size, 3])
            images_batch = tf.reshape(images_var, [-1, crop_size, crop_size, 3])

        # Build cnn models
        sys.path.append(cnn_dir)
        cnn = __import__('net') 
        net = cnn.CaffeNet({'data': images_batch}, trainable=train)
        graph = tf.get_default_graph()

        if new_layer is not None:
            # http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/myalexnet_forward.py
            fc7 = graph.get_tensor_by_name('fc7/fc7:0')
            fc8W = tf.Variable(tf.random_normal([4096, new_layer], stddev=0.01))
            ## fc8 = tf.matmul(fc7, fc8W)     
            fc8b = tf.Variable(tf.random_normal([new_layer], stddev=0.01))
            fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
            prob = tf.nn.softmax(fc8)

        else:
            prob = graph.get_tensor_by_name('prob')

        return prob, graph, net, images_var


    def train_cnn(self, cnn_dir, data, 
                  new_layer=None, ckpt_file='model.ckpt', init_weights=None):
        """
        TODO
        ----
        - use other optimizers?
        - arg for descent rate?
        - prob.shape for ground_truth

        """
        prob, graph, net, images_var = self.load_cnn(cnn_dir, new_layer=new_layer, train=True)
        #import ipdb; ipdb.set_trace()
        ground_truth = tf.placeholder(tf.float32, shape=[self.batch_size, prob.get_shape()[1]])

        cost = tf.nn.sigmoid_cross_entropy_with_logits(prob, ground_truth)
        train_op = tf.train.AdamOptimizer(0.005).minimize(cost)

        saver = tf.train.Saver()
        ckpt_path = os.path.join(cnn_dir, ckpt_file)

        with tf.Session() as sess:
            tf.initialize_all_variables().run()

            if os.path.exists(ckpt_path):
                saver.restore(sess, ckpt_path)
            elif init_weights is not None:
               net.load(init_weights, sess) 


            for e, (batch_imgs, batch_labels) in enumerate(data):
                train_dict = {images_var:batch_imgs, ground_truth:batch_labels}
                sess.run(train_op, feed_dict=train_dict)

                save_path = saver.save(sess, ckpt_path)
                print('Model saved: {}   Batch: {}'.format(save_path, e))



    def train_cnns(self, weights_path=None):
        # TODO
        # 1. Split data into (imgs, labels) sorted by epoch (i.e. minimizing total # images) 
        #   [[ (batch_imgs, batch_labels) for batch in data_epoch ] for data_epoch in data ]
        #### data = TODO 
        self.train_cnn(self.obj_dir, data, init_weights=self.obj_dir+'init_weights.npy')
        self.train_cnn(self.rel_dir, data, init_weights=self.obj_dir+'init_weights.npy')
        return None



    def parse_scenes(self, scene_graphs):
        """
        Function to take scene graphs and return list of (image_id, (x,y,w,h)) pairs

        """
        scene_graphs = rel_coords(scene_graphs)
        r_id = lambda r: (r.subject.id, r.object.id)
        obj_data = {(sg.image.id, o.id, (o.x, o.y, o.width, o.height)) for sg in scene_graphs for o in sg.objects}
        ## TODO: this shouldn't be necessary, right..?
        obj_data.update({(sg.image.id, o.id, (o.x, o.y, o.width, o.height)) for sg in scene_graphs for r in sg.relationships for o in (r.subject, r.object)})
        rel_data = {(sg.image.id, r_id(r), (r.x, r.y, r.width, r.height)) for sg in scene_graphs for r in sg.relationships}
        return list(obj_data), list(rel_data)



    def load_images(self, batch_data, batch=True):
        """
        - load and crop images, return list in same order as given
        - if fewer images than batch size, pad batch with black frames

        TODO 
        ----
        - should we pre-process cropped images?

        """
        if not batch: 
            batch_data = [batch_data]

        img_ids =  set(zip(*batch_data)[0])
        imgs = {img_id:imread(self.img_path + str(img_id) + '.jpg') for img_id in img_ids}

        crops = []
        for img_id, uid, coord in batch_data:
            img = imgs[img_id]
            crop = square_crop(img, self.crop_size, *coord)
####            new_img -= np.load(self.obj_dir + 'mean.npy')
            crops.append(crop)

    def batchify_data(self, data):
        #### data = TODO_SPLIT_BY self.batch_size

        pad_len = self.batch_size - len(batch_data)
        if batch and (pad_len > 0):
            pad_imgs = [np.zeros(226,226,3) for _ in range(pad_len)]
            crops = crops + pad_imgs
            return crops
        else:
            return crop



    def get_dicts(self, obj_data, rel_data):
        """
        Create dictionaries that give indexes for CNN outputs for object & relationship UIDs.

        """
        rid = lambda ii, oi1, oi2: frozenset([id_(ii,oi1), id_(ii,oi2)])
        obj_dict = {id_(img_id, obj_id)   : idx for idx, (img_id, obj_id, coord) in enumerate(obj_data)}
        rel_dict = {rid(img_id, *obj_ids) : idx for idx, (img_id, obj_ids, coord) in enumerate(rel_data)}
        return obj_dict, rel_dict


    def run_cnn(self, data, cnn_dir, layer, ckpt_file='model.ckpt', new_layer=100):
        """
        TODO: update line: `new_layer=100` 
        
        """
        prob, graph, images_var = self.load_cnn(cnn_dir, new_layer=new_layer)
        # epochs = int(np.ceil(float(len(data)) / self.batch_size))
        graph_layer = prob if layer == 'prob' else graph.get_tensor_by_name(layer)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            saver.restore(sess, os.path.join(cnn_dir, ckpt_file))

            layer_out  = []
            for d in data:
                # batch_data = data[e:e+self.batch_size]
                img = self.load_images(d, batch=False)

                feed = {images_var: img}
                batch_prob = sess.run(graph_layer, feed_dict=feed)[0]   # TODO should this be the 0th index?
                layer_out.append(batch_prob)

        return np.vstack(layer_out)



    def run_cnns(self, scene_graphs, weights_path=None):
        """
        Loop for each batch of images, run cnn, aggregate prob values
        
        """
        obj_data, rel_data = self.parse_scenes(scene_graphs)
        obj_dict, rel_dict = self.get_dicts(obj_data, rel_data)

        obj_probs = self.run_cnn(obj_data, self.obj_dir, 'prob', ckpt_file=self.obj_ckpt)
        rel_feats = self.run_cnn(rel_data, self.rel_dir, 'fc7/fc7:0', self.rel_ckpt)

        return obj_probs, rel_feats, obj_dict, rel_dict







class Model:
    """
    Arguments
    ---------
    w2v : w2v = (100 + 70, 300)  numpy vector, 1 for each word, then 1 for each predicate
    obj_probs : obj_probs[rel_index] = (100,) final layer cnn output
    rel_feats : rel_feats[rel_index] = (4096,) fc7 feature (PRESUMABLY this was used . . .)
    obj_dict : match image-specific object id to index in obj_probs
    rel_dict : match image-specific relationship id to index in rel_feats

    init_type : TODO
    num_samples : number of random R samples to compute for equation 4

    noise : initialization noise
    lamb1 : weight of eq 5 in eq 7
    lamb2 : weight of eq 4 in eq 7


    TODO
    ----
    - should we only compute `R_samples` as needed? or for each call to K?


    """
    def __init__(self, obj_probs, rel_feats, obj_dict, rel_dict, 
                 w2v, w2v_dict, n,
                 init_type='TODO', noise=0.05, learning_rate=1.0, max_iters=20,
                 num_samples=500000, lamb1=0.05, lamb2=0.001):
        self.obj_probs     = obj_probs
        self.rel_feats     = rel_feats
        self.obj_dict      = obj_dict
        self.rel_dict      = rel_dict
        self.w2v           = w2v
        self.w2v_dict      = w2v_dict
        self.init_type     = init_type
        self.noise         = noise
        self.learning_rate = learning_rate
        self.max_iters     = max_iters
        self.num_samples   = num_samples
        self.lamb1         = lamb1
        self.lamb2         = lamb2

        self.n = n
        self.k = w2v.shape[0] - n
        self.init_weights()
        self.init_R_samples()

    def init_weights(self):
        ndim = self.rel_feats.shape[1]
        self.W = self.noise * np.random.rand(self.k, 600) 
        self.b = self.noise * np.random.rand(self.k, 1)  
        self.Z = self.noise * np.random.rand(self.k, ndim)  
        self.s = self.noise * np.random.rand(self.k, 1)  


    def init_R_samples(self):
        """
        Draw a number of random (i,j,k) indices: 2 objects and 1 relationship
        
        """
        N = self.n
        K = self.k
        R_rand = lambda: (randint(N), randint(N), randint(K))
        R_pairs = [(R_rand(), R_rand()) for n in range(self.num_samples)]
        self.R_samples = R_pairs

    def convert_rel(self, rel):
        """
        Converting training data `Relationship` instance to `<i,j,k>` format.

        """
        i = self.w2v_dict['obj'][rel.subject.names[0]]
        j = self.w2v_dict['obj'][rel.object.names[0]]
        k = self.w2v_dict['rel'][rel.predicate]
        return i,j,k

    def word_vec(self, i, j):
        return np.concatenate((self.w2v[i], self.w2v[j]))


    def w2v_dist(self, R1, R2):
        N = self.n
        return cosine(self.w2v[R1[0]],   self.w2v[R2[0]]) +  \
               cosine(self.w2v[R1[1]],   self.w2v[R2[1]]) +  \
               cosine(self.w2v[N+R1[2]], self.w2v[N+R2[2]])

    def d(self, R1, R2, W, b):
        """
        Distance between two predicate triplets.
        
        """
        d_rel = self.f(R1, W, b) - self.f(R2, W, b) 
        d_obj = self.w2v_dist(R1, R2)
        d = (d_rel ** 2) / d_obj
        return d if (d > 0) else 1e-10


    def f(self, R, W, b):
        """
        Project relationship `R = <i,j,k>` to scalar space.

        """
        i,j,k = R
        wvec = self.word_vec(i, j)
        return np.dot(W[k].T, wvec) + b[k]

    def V(self, R, O1, O2, Z, s):
        i,j,k = R

        O1_id  = self.obj_dict[O1]
        O2_id  = self.obj_dict[O2]
        rel_id = self.rel_dict[frozenset([O1, O2])]

        P_i = self.obj_probs[O1_id, i]
        P_j = self.obj_probs[O2_id, j]
        cnn = self.rel_feats[rel_id]
        P_k = np.dot(Z[k], cnn) + s[k]
        return P_i * P_j * P_k
    

    def predict_Rs(self, O1, O2):
        """
        Full list of predictions `R`, sorted by confidence

        """
        N, K = self.n, self.k

        Rs = [(i,j,k) for i in range(N) for j in range(N) for k in range(K)]
        M = sorted(Rs, key=lambda R: -V(R,O1,O2, self.W, self.b) * f(R, self.W, self.b))
        return M

    def compute_accuracy(self, GT, k=None):
        """
        TODO
        ----
        input should be list of (R, O1, O2) for each relationship in each image



        - X: list of predictions `x` for each data point, sorted by confidence
        - Ys: list of ground truth labels `y` for each data point
        - each data point should be an `(i,j,k)` tuple

        MAP
        ---
        y, y_: ground truth label for data point `i`
        X: ordered list of label predictions for data point `i`

        Recall @ k
        ----------
        is the correct label within the top k predictions?

        """
        test_data = [[(predict_Rs(O1, O2), R) for R, O1, O2 in datum] for datum in GT]

        # MAP
        if k is None:
            mean_ap = 0.0
            for X, Y in test_data:
                rank = lambda y: float(X.index(y) + 1)
                prec = lambda y: len([y_ for y_ in Y if rank(y_) <= rank(y)]) / rank(y)

                avg_prec = sum(prec(y) for y in Y) / len(Y)
                mean_ap += avg_prec / len(test_data)
            return mean_ap
        # Recall @ k
        else:
            recall = 0.0
            for X, Y in test_data:
                z = float(len(Y) * len(test_data))
                recall += sum((y in X[:k]) for y in Y) / z
            return recall
             
    def SGD(self, D):   
        """
        Perform SGD over eqs 5 (L) 6 (C)

        """
        obj_probs, rel_feats, w2v = (self.obj_probs, self.rel_feats, self.w2v)
        V, f, d = (self.V, self.f, self.d)
        W, b, Z, s = (self.W, self.b, self.Z, self.s)

        for i in range(self.max_iters):
            print 'SGD iteration [{}]'.format(i)
            # TODO: is it worth it to pre-compute these as tables?
            #V = {d:self.V(*d,Z,s) for d in D}
            #F = {R:f(R,W,b) for R in zip(*D)[0]}

            D = sorted(D, key=lambda x: np.random.rand())
            mc = 0.0
            mc_prev = 1.0

            for R, O1, O2 in D:
                # Get max item
                D_ = [(R_,O1_,O2_) for R_,O1_,O2_ in D if (R_ != R) and (O1_ != O1 or O2_ != O2)]
                #D_ = [(R_,O1_,O2_) for R_,O1_,O2_ in D if (R_ != R)]  # TODO: this is the weird max thing in eq 6
                M = sorted(D_, key=lambda (R,O1,O2): V(R,O1,O2,Z,s) * f(R,W,b))
                R_,O1_,O2_ = M[0]
                i,j,k = R
                i_,j_,k_ = R_

                # Compute value for ` max{cost, 0} ` in equation 6
                cost = 1 - V(R,O1,O2,Z,s) * f(R,W,b) + V(R_,O1_,O2_,Z,s) * f(R_,W,b)
                ## print V(R,O1,O2,Z,s), f(R,W,b), V(R_,O1_,O2_,Z,s), f(R_,W,b)
                mc  += cost / len(D)


                # Update W,b
                # ----------
                if i % 2 == 0:
                    ##print 'PART 1'
                    # Equation 5
                    for R_2, O1_2, O2_2 in D:
                        i_2,j_2,k_2 = R_2
                        if f(R_2, W, b) - f(R, W, b) > 0:
                            W[k]   -= self.learning_rate * np.concatenate((w2v[i],  w2v[j])) * self.lamb1
                            b[k]   -= self.learning_rate * self.lamb1
                            W[k_2] += self.learning_rate * np.concatenate((w2v[i_2], w2v[j_2])) * self.lamb1
                            b[k_2] += self.learning_rate * self.lamb1

                    # Equation 6
                    if cost > 0:
                        v = V(R,O1,O2,Z,s)
                        W[k]  -= self.learning_rate * v * np.concatenate((w2v[i], w2v[j]))
                        b[k]  -= self.learning_rate * v
                        v_ = V(R_,O1_,O2_,Z,s)
                        W[k_] += self.learning_rate * v_ * np.concatenate((w2v[i_], w2v[j_]))
                        b[k_] += self.learning_rate * v_


                # Update Z,s
                # ----------
                else:         
                    ##print 'PART 2'
                    # Equation 6    
                    if cost > 0:
                        id_o1  = self.obj_dict[O1]
                        id_o2  = self.obj_dict[O2]
                        id_o1_ = self.obj_dict[O1_]
                        id_o2_ = self.obj_dict[O2_]
                        id_r   = self.rel_dict[frozenset([O1, O2])]
                        id_r_  = self.rel_dict[frozenset([O1_, O2_])]

                        Z[k]  -= self.learning_rate * f(R,W,b) * obj_probs[id_o1,i] * obj_probs[id_o2,j] * rel_feats[id_r]
                        s[k]  -= self.learning_rate * f(R,W,b) * obj_probs[id_o1,i] * obj_probs[id_o2,j]
                        Z[k_] += self.learning_rate * f(R_,W,b)  * obj_probs[id_o1_,i_] * obj_probs[id_o2_,j_] * rel_feats[id_r_]
                        s[k_] += self.learning_rate * f(R_,W,b)  * obj_probs[id_o1_,i_] * obj_probs[id_o2_,j_]

            '''
            # Equation 4
            if i % 2 == 0:
                print 'PART 3'
                dKfun = sum(    (2. / d(R,R_,W,b)) * 
                                (f(R,W,b) - f(R_,W,b)) * 
                                (self.word_vec(*R[:-1]) - self.word_vec(*R_[:-1]))
                            for R,R_ in self.R_samples )

                Kfun = lambda R, R_: (f(R,W,b) - f(R_,W,b))**2 / d(R,R_,W,b)
                # def Kfun (R,R_):
                #     v = (f(R,W,b) - f(R_,W,b))**2 / d(R,R_,W,b)
                #     if np.isnan(v):
                #         print 'KFUN: {} ; {} ; {}; {}'.format(f(R,W,b), f(R_,W,b), d(R,R_,W,b), v)
                #     return (f(R,W,b) - f(R_,W,b))**2 / d(R,R_,W,b)

                Ksum = sum(Kfun(R,R_) for R, R_ in self.R_samples)
                nr = self.num_samples
                dK_dW = ((2.0 - nr) / nr) * Ksum * dKfun
                print '\tKsum', Ksum
                print '\tdKfun', dKfun

                # TODO: separate `W[k] += ...` and `W[k_] -= ...`
                W += self.learning_rate * dK_dW
            '''
            print '\tchange in cost: {}'.format(mc_prev - mc)
            mc_prev = mc

        self.W, self.b, self.Z, self.s = (W, b, Z, s)
        return W, b, Z, s


    def load_data(self, scene_graphs):
        """
        Load a list of data triplets, one for each scene graph: (R, O1, O2)

        """
        obj_names = set([o.names[0] for sg in scene_graphs for o in sg.objects])
        rel_names = set([r.predicate for sg in scene_graphs for r in sg.relationships])
        obj_w2id  = {n:i for i,n in enumerate(obj_names)}
        rel_w2id  = {n:i for i,n in enumerate(rel_names)}

        D = []
        for sg in scene_graphs:
            img_id = sg.image.id
            for rel in sg.relationships:
                R = self.convert_rel(rel)
                O1 = id_(img_id, rel.subject.id)
                O2 = id_(img_id, rel.object.id)
                D.append((R, O1, O2))

        self.obj_names, self.rel_names = (obj_names, rel_names)
        self.obj_w2id,  self.rel_w2id  = (obj_w2id,  rel_w2id)
        return D















# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================
# scraps




    # -------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------
    # UNUSED METHODS IN `VisualModel`
    
    def K(self):
        """  UNUSED
        Eq (4): randomly sample relationship pairs and minimize variance.
        
        """
        D = []
        for R1, R2 in self.R_samples:
            d = dist(R1, R2)
            D.append(d)
        return np.var(D)

    def L(self, D):
        """  UNUSED
        Likelihood of relationships
        
        """
        Rs, O1s, O2s = zip(*D)
        Rs = zip(Is, Js, Ks)
        fn = lambda R1, R2: max(self.f(*R1) - self.f(*R2) + 1, 0)
        return sum(fn(R1, R2) for R1 in Rs for R2 in Rs)
                
    def C(self, img_data):
        """  UNUSED
        Rank loss function
        
        """
        C = 0.0
        for d1 in img_data:
            c = max(V(*d2) * f(*d2[3:]) for d2 in img_data if  \
                    (d2[5] != d2[5]) and ((d1[3] != d2[3]) or (d1[4] != d2[4])))
            C += max(1 + V(*d1) * f(*d1[3:]) * c, 0)
        return C

    def loss(self, img_data):
        """  UNUSED
        Final objective loss function.
        
        """
        C = self.C(img_data)
        K = self.lamb1 * self.K()
        L = self.lamb2 * L(img_data)
        return C + K + L

    def predict_R(self, O1, O2):
        """  UNUSED
        R* = argmax_R V(R, Z | <O_1, O_2>) f(R,W)

        """
        N = rel_feats.shape[1]
        K = obj_probs.shape[1]

        Rs = [(i,j,k) for i in range(N) for j in range(N) for k in range(K)]
        M = sorted(Rs, key=lambda R: -V(R,O1,O2) * f(R,W,b))
        return M[-1]

    # -------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------




'''




    # TODO: move to VisualModel
    # -------------------------
    # def init_rel(self, scene_graphs):
    #     """
    #     Setup `self.rel_dict[frozenset([o1,o2])] = idx`

    #     TODO
    #     ----
    #     - init obj_dict (imageid_objid)   --> idx
    #     - maybe we should output rel_dict inside VisualModel -- otherwise rel_feats will be indexed wrong . . .

    #     """
    #     O = []
    #     obj_pairs = set()
    #     for o1 in O:
    #         for o2 in O:
    #             obj_pairs.add(frozenset([o1, o2]))

    #     self.rel_dict = {r:idx for idx,r in enumerate(obj_pairs)}
    #     self.obj_dict = {r:idx for idx,r in enumerate(obj_pairs)}





# MAP
# ===
#  y, y_: ground truth label for data point `i`
#  x: ordered list of label predictions for data point `i`

# def rank(x, y):
#     return float(x.index(y) + 1)

# def prec(x, y):
#     ls = [y_ for y_ in Y if rank(x, y_) <= rank(x, y)]
#     return len(ls) / rank(x_i, y)




 def SGD(self, D):   
        """
        Perform SGD over eqs 5 (L) 6 (C)

        def SGD(self, obj_probs, rel_feats, w2v,
            D, W, b, Z, s, lamb1, lamb2):

        TODO
        ----
        - O1, O2 should be unique object id's (maybe 'imageid_objectid')
        - P_r should be indexed by (O1, O2)

        """
        obj_probs = self.obj_probs
        rel_feats = self.rel_feats
        w2v = self.w2v
        W = self.W
        b = self.b
        Z = self.Z
        s = self.s
        V = self.V
        f = self.f
        lamb1 = self.lamb1

        for i in range(self.max_iters):
            # TODO: is it worth it to pre-compute these as tables?
            #V = {d:V(*d,Z,s) for d in D}
            #F = {R:f(R,W,b) for R in zip(*D)[0]}

            # Equation 5
            # ----------
            D = sorted(D, key=lambda x: np.random.rand())
            for R, O1, O2 in D:
                i,j,k = R
                for R_, O1_, O2_ in D:
                    i_,j_,k_ = R_

                    if f(R_, W, b) - f(R, W, b) > 0:
                        # Update W,b
                        W[k_] += np.concatenate(w2v[i_], w2v[j_])
                        W[k]  -= np.concatenate(w2v[i],  w2v[j])
                        b[k_] += lamb1
                        b[k]  -= lamb1

            # Equation 6
            # ----------
            D = sorted(D, key=lambda x: np.random.rand())
            for R, O1, O2 in D:
                D_ = [(R_,O1_,O2_) for R_,O1_,O2_ in D_ if (R_ != R) and (O1_ != O1 or O2_ != O2)]
                M = sorted(D_, key=lambda R,O1,O2: V(R,O1,O2) * f(R,W,b))
                R_,O1_,O2_ = M[0]

                id_o1  = self.obj_dict[O1]
                id_o2  = self.obj_dict[O2]
                id_o1_ = self.obj_dict[O1_]
                id_o2_ = self.obj_dict[O2_]
                id_r   = self.rel_dict[frozenset([O1, O2])]
                id_r_  = self.rel_dict[frozenset([O1_, O2_])]
                i,j,k = R
                i_,j_,k_ = R_

                if (1 + V(R,O1,O2,Z,s) * f(R,W,b) - V(R_,O1_,O2_,Z,s) * f(R_,W,b)) > 0:
                    # Update W,b
                    W[k] += V(R,O1,O2,Z,s) * np.concatenate(w2v(i), w2v(j))
                    b[k] += V(R,O1,O2,Z,s)
                    W[k_] -= V(R_,O1_,O2_,Z,s) * np.concatenate(w2v(i_), w2v(j_))
                    b[k_] -= V(R_,O1_,O2_,Z,s)

                    # Update Z,s
                    Z[k] += f(R,W,b) * obj_probs[id_o1,i] * obj_probs[id_o2,j] * P_rel[id_r,k]
                    s[k] += f(R,W,b) * obj_probs[id_o1,i] * obj_probs[id_o2,j]
                    Z[k_] -= f(R_,W,b)  * obj_probs[id_o1_,i_] * obj_probs[id_o2_,j_] * P_rel[id_r_,k_]
                    s[k_] -= f(R_,W,b)  * obj_probs[id_o1_,i_] * obj_probs[id_o2_,j_]

        self.W = W
        self.b = b
        self.Z = Z
        self.s = s
        return W, b, Z, s



    def update_weights(self, TODO):
        """
        This way we don't have to do the dot product of Z with fc7 each iteration...

        """
        # update weights . . .
        self.P_K = np.dot(self.Z.T, self.rel_fc7) + np.tile(self.S, self.rel_fc7.shape[0])



    def V_all(self, img_data):
        """
        Compute V for a list, with an arbitrary sized set of relation triplets for each.

        data should be a list of tuples:
            (rel_id, o1_id, o2_id, Rs)
        
        """
        
        P = {}    
        for d in img_data:
            rel_id, o1_id, o2_id, Rs = d
            ids = (rel_id, o1_id, o2_id)
            P[rel_id] = {R: self.V(*ids, *R) for R in Rs}
        return P



    def F(self, i, j):
        """
        Project relationship `R = <i,j>` to K-dim relationship space.

        """
        word2vec = np.concatenate(self.w2v[i], self.w2v[j])
        return np.dot(self.W.T, word2vec) + self.B


'''
