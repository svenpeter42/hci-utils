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

n_block = 256

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile')
    parser.add_argument('datadset')
    parser.add_argument('labelfile')
    parser.add_argument('labeldset')
    parser.add_argument('outfile')
    args = parser.parse_args()

    with h5.File(args.datafile, 'r') as fdata, h5.File(args.labelfile, 'r') as flabel:
        print("Loading labels")
        dset_data = fdata[args.datadset]
        dset_label = flabel[args.labeldset][:]

        labels = np.where(dset_label > 0)

        n_train = len(labels[0])

        print("Selecting training data")
        train_x = np.zeros((n_train, dset_data.shape[4]), dtype=np.float32)
        train_y = np.zeros((n_train), dtype=np.uint32)
        offset = 0

        for i in range(dset_data.shape[0]):
            print_progress(i, dset_data.shape[0])
            i_dset_label = dset_label[i,:]
            i_label = np.where(i_dset_label > 0)

            i_label_y = np.array(i_label[1]) & 0xffffff00
            i_label_x = np.array(i_label[2]) & 0xffffff00

            i_label_coord = i_label_y<<16 | i_label_x
            i_blocks = vigra.analysis.unique(i_label_coord)


            j = 0
            for i_block in i_blocks:
                i_bx = i_block >> 16
                i_by = i_block & 0xffff
                idx = np.where(i_label_coord == i_block)

                idx_x = i_label[1][idx]
                idx_y = i_label[2][idx]

                i_blk_data = dset_data[i, 0, i_bx:i_bx+256, i_by:i_by+256, :]
                i_blk_label = dset_label[i, 0, i_bx:i_bx+256, i_by:i_by+256, :]
                train_x[offset:offset+len(idx[0]),:] = i_blk_data[idx_x - i_bx, idx_y - i_by, :]
                train_y[offset:offset+len(idx[0])] = i_blk_label[idx_x - i_bx, idx_y - i_by, :].squeeze()
                offset += len(idx[0])
                print_progress(i + j/len(i_blocks), dset_data.shape[0])
                j += 1
            

        print_progress(dset_data.shape[0], dset_data.shape[0])

        print("Learning random forest")
        rf = vigra.learning.RandomForest3(train_x, train_y, treeCount=100, n_threads=6)

        print("Exporting random forest")
        rf.writeHDF5(args.outfile, "/Forest")