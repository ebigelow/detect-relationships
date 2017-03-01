import numpy as np
import os
from scipy.ndimage import imread
from optparse import OptionParser

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-i", "--in", dest="img_dir",
                      type="string", default="./images",
                      help="Get the mean of all images in this directory.")
    parser.add_option("-o", "--out", dest="out_file",
                      type="string", default="./model/mean.npy",
                      help="Save mean to this file.")
    (options, args) = parser.parse_args()

    img_dir = options.img_dir
    img_files = os.listdir(img_dir)
    N = len(img_files)

    # Load each image, take its mean (BGR), collect in `Mu`
    Mu = np.zeros(3)
    for img_file in img_files:
        rgb = imread(os.path.join(img_dir,img_file)).astype(float)
        bgr = rgb[:,:,::-1]
        Mu += np.mean(bgr, axis=(0,1))

    Mu /= N

    print 'Mean of {} images: {}'.format(N, Mu)
    np.save(options.out_file, Mu)
