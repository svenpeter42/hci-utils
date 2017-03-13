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
    parser.add_argument('infiles', nargs='+')
    args = parser.parse_args()


    for i, fname in enumerate(args.infiles):
        print_progress(i, len(args.infiles))

        fname, fext = os.path.splitext(fname)

        img = vigra.impex.readImage("%s%s" % (fname, fext)).view5D().transposeToNumpyOrder()

        with h5.File("%s.h5" % fname) as f:
            f.create_dataset("data", data=img, chunks=True, compression='lzf')

    print_progress(i, len(args.infiles))