# -*- coding: utf-8 -*-

import h5py as h5
import fastfilters as ff
import vigra
import os, sys
import argparse
import concurrent.futures
import numpy as np
import vigra
import itertools
from time import sleep

from common import print_progress

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile")
    parser.add_argument('datadset')
    parser.add_argument('predfile')
    parser.add_argument('preddset')
    parser.add_argument('outprefix')
    parser.add_argument('--alpha', default=0.3, type=float)
    args = parser.parse_args()

    overlaycolor = [1,0,0]

    if not os.path.exists(args.outprefix):
        os.mkdir(args.outprefix)

    with h5.File(args.datafile, 'r') as fdata, h5.File(args.predfile) as fpred:
        data = fdata[args.datadset]
        pred = fpred[args.preddset]

        n_slices, n_z, n_y, n_x, n_classes = pred.shape

        assert(n_z == 1)

        for i_slice in range(n_slices):
            for i_class in range(n_classes):
                print_progress(i_slice*n_classes + i_class, n_slices*n_classes)

                img = data[i_slice, 0, :, :, :]
                proba = pred[i_slice, 0, :, :, i_class]

                img /= np.max(img)
                proba /= np.max(proba)

                res = np.zeros((n_y, n_x, 3))
                for i in range(3):
                    res[:,:,i] = (1-args.alpha) * img[:,:,i] + args.alpha*proba*overlaycolor[i]
                vigra.impex.writeImage(res, "%s/proba_%03d_%03d.png" % (args.outprefix, i_slice, i_class))


        print_progress(n_slices*n_classes, n_slices*n_classes)
