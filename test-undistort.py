#!/usr/bin/env python
import numpy as np
import cv2
import os
import argparse
import yaml
from glob import glob

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Undistort images based on camera calibration.')
    # parser.add_argument('calibration', help='input video file')
    # parser.add_argument('input_mask', help='input mask')
    # parser.add_argument('out', help='output directory')
    # args = parser.parse_args()

    with open('example_output/calibration.yaml') as fr:
        c = yaml.load(fr)

    for fn in [1]:
        print('processing ...')
        img = cv2.imread('test.jpg')
        if img is None:
            print("Failed to load " + str(fn))
            continue

        K_undistort = np.array(c['camera_matrix'])
        # K_undistort[0:2, 2] = [0., 0.]
        # K_undistort[0, 0] *= 0.3
        # K_undistort[1, 1] *= 0.3

        with open('~/smart_axe/calibration.yaml') as fr:
            c = yaml.load(fr)

        img_und = cv2.undistort(img, np.array(c['camera_matrix']), np.array(c['dist_coefs']),
                                newCameraMatrix=np.array(c['camera_matrix']))
        # name, ext = os.path.splitext(os.path.basename(fn))
        cv2.imwrite('test-und.jpeg',img_und)

        print('ok')


