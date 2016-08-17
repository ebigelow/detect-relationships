
import numpy as np
import cv2













def save_file(fname, s):
    with open(fname) as f:
        f.write(s)



"""
Given a list of scene graphs, output two text files:
   image_fname label

Also, crop images for objects and relationships, save to new file . . .

TODO 
----
??? remove redundant objects


"""
def sg_to_caffe(scene_graphs, O_dict, R_dict, img_dir='images/', train_ratio=0.9):
    O_data = []
    R_data = []

    obj_dir = img_dir+'objs/'
    rel_dir = img_dir+'objs/'

    for sg in scene_graphs:
        sg_id = str(sg.image.id)
        img = cv2.imread(img_dir + sg_id)

        objs = [o for o in sg.objects if o.names[0] in O_dict]
        for o in objs:
            img_obj = img[o.y:o.y+o.height, o.x:o.x+o.width, :]
            fname = obj_dir + sg_id + '_' + o.id
            cv2.imwrite(img_obj, fname)

            w2v_idx = O_dict[o]
            O_data.append((fname, w2v_idx))


        rels = [r for r in sg.objects if r.predicate in R_dict]
        for r in rels:
            img_rel = img[r.y:r.y+r.height, r.x:r.x+r.width, :]
            fname = img_dir+'rels/' + sg_id + '_' + r.id
            cv2.imwrite(img_rel, fname)

            w2v_idx = R_dict[r]
            R_data.append((fname, w2v_idx))

    np.random.shuffle(O_data)
    np.random.shuffle(R_data)
    
    to_str = lambda data: '\n'.join([' '.join(d) for d in data])

    nobj = train_ratio * len(O_data)
    nrel = train_ratio * len(R_data)

    save_file(obj_dir + 'train.txt', to_str( O_data[:nobj] ))
    save_file(obj_dir + 'test.txt',  to_str( O_data[nobj:] ))
    save_file(rel_dir + 'train.txt', to_str( R_data[:nrel] ))
    save_file(rel_dir + 'test.txt',  to_str( R_data[nrel:] ))



