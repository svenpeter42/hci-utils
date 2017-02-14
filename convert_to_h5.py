# -*- coding: utf-8 -*-

import h5py as h5
import fastfilters as ff
import vigra
import os, sys
import argparse
import concurrent.futures
import numpy as np
from time import sleep
from PIL import Image

from common import print_progress


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile')
    parser.add_argument('datadset')
    parser.add_argument('infiles', nargs='+')
    args = parser.parse_args()

    print("Checking image files")
    shapes = [Image.open(i).size for i in args.infiles]
    shapes = np.array(shapes)

    assert len(np.unique(shapes[:,0])) == 1
    assert len(np.unique(shapes[:,1])) == 1

    y,x = shapes[0,:]
    c = 3


    print("Preparing h5 file")
    with h5.File(args.outfile) as f:
        if args.datadset in f:
            dset = f[args.datadset]
            if len(dset.shape) != 5: raise ValueError("shape ndim mismatch.")
            if dset.shape[1:] != (1,y,x,c): raise ValueError("shape yxc mismatch.")
            if dset.maxshape[0] != None: raise ValueError("Cannot extend dset.")
        else:
            f.create_dataset(args.datadset, (0,1,y,x,c), chunks=(1,1,256,256,c), maxshape=(None, 1, y, x, c))

        data = f[args.datadset]

        for i, i_fname in enumerate(args.infiles):
            print_progress(i, len(args.infiles))

            img = vigra.impex.readImage(i_fname)
            data.resize(data.shape[0]+1, axis=0)
            data[data.shape[0]-1,0,:,:,:] = img[:,:,:3]
