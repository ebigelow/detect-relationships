"""
Use this file to extract the mean of a directory of images.

With CNN we subract mean from all images from each image, so on
average each channel has values centered around 0.

Example:
  $ python get_mean.py -i /path/to/images -o /path/to/output/mean.npy

"""
import numpy as np
import cv2
import os
from optparse import OptionParser

if __name__ == '__main__':
    
    parser = OptionParser()
    parser.add_option("-i", "--in", dest="img_dir",
                      type="string", default="./images",
                      help="Images directory.")
    parser.add_option("-o", "--out", dest="out_file",
                      type="string", default="./model/mean.npy",
                      help="Out directory.")
    (options, args) = parser.parse_args()
    img_dir = options.img_dir
    img_files = os.listdir(img_dir)
    N = len(img_files)

    Mu = np.zeros(3)
    for img_file in img_files:
        img = cv2.imread(os.path.join(img_dir,img_file)).astype(float)
        Mu += np.mean(img, axis=(0,1)) / N

    np.save(options.out_file, Mu)
