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
    parser.add_argument("forest")
    parser.add_argument('datafile')
    parser.add_argument('datadset')
    parser.add_argument('outfile')
    parser.add_argument('outdset')
    args = parser.parse_args()

    print("Loading RF")
    rf = vigra.learning.RandomForest3(args.forest, "/Forest")
    n_classes = rf.labelCount()

    print("Predicting")

    with h5.File(args.datafile, 'r') as fdata, h5.File(args.outfile) as fout:
        data = fdata[args.datadset]

        n_slices, n_z, n_y, n_x, n_features = data.shape
        n_blocks_x = n_x >> 8
        n_blocks_y = n_y >> 8
        n_blocks_total = n_blocks_y*n_blocks_x

        dataout = fout.create_dataset(args.outdset, (n_slices, n_z, n_y, n_x, n_classes), chunks=(1,1,256,256,n_classes))

        for i_slice in range(n_slices):
            print_progress(i_slice, n_slices, bar_length=80)

            i_progress = 0
            for i_blk_x, i_blk_y in itertools.product(range(n_blocks_x), range(n_blocks_y)):
                print_progress(i_slice + i_progress/n_blocks_total, n_slices, suffix='slice: %03d/%03d block: (%03d,%03d)/(%03d,%03d)' % (i_slice, n_slices, i_blk_x, i_blk_y, n_blocks_x, n_blocks_y), bar_length=80)

                i_blk_x  = i_blk_x << 8
                i_blk_y  = i_blk_y << 8
                block = data[i_slice, 0, i_blk_y:i_blk_y+256, i_blk_x:i_blk_x+256, :]
                dataout[i_slice, 0, i_blk_y:i_blk_y+256, i_blk_x:i_blk_x+256, :] = rf.predictProbabilities(block.reshape((-1, n_features)), n_threads=8).reshape((block.shape[0], block.shape[1], n_classes))
                i_progress += 1

        print_progress(n_slices, n_slices, bar_length=80)