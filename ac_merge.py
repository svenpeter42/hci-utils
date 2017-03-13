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
    parser.add_argument('imgfile')
    parser.add_argument('probafile')
    parser.add_argument('outfile')
    args = parser.parse_args()



    with h5.File(args.imgfile, 'r') as fimg, h5.File(args.probafile, 'r') as fproba, h5.File(args.outfile, 'w') as fout:
        indset = fimg[list(fimg)[0]]
        probadset = fproba[list(fproba)[0]]

        outshape = (indset.shape[2], indset.shape[3], indset.shape[4] + probadset.shape[4])
        outdset = fout.create_dataset("data", outshape, chunks=True)


        img = indset[:]
        img /= np.max(img)

        outdset[:,:,:indset.shape[4]] = img
        outdset[:,:,indset.shape[4]:] = probadset[:]