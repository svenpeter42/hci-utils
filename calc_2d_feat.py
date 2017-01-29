# -*- coding: utf-8 -*-

import h5py as h5
import fastfilters as ff
import vigra
import os, sys
import argparse
import concurrent.futures
import numpy as np
from time import sleep

# https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
# http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def calc_all_features(pool, futures, data, fvec, fvecidx, fvecoffset, scales):
    n_features = 1 + 7*len(scales)

    def calc_features(data, fvec, fvecidx, fvecoffset, i, scale, j):
        # UGLY!
        sleep(1) # hack to at least show the progress bar. yay, GIL.
        if j == 0:
            fvec[fvecidx,0,:,:,1+7*i+0+fvecoffset] = ff.gaussianSmoothing(data, scale)
        elif j == 1:
            fvec[fvecidx,0,:,:,1+7*i+1+fvecoffset] = ff.laplacianOfGaussian(data, scale)
        elif j == 2:
            fvec[fvecidx,0,:,:,1+7*i+2+fvecoffset] = ff.gaussianGradientMagnitude(data, scale)
        elif j == 3:
            fvec[fvecidx,0,:,:,1+7*i+3+fvecoffset:1+7*i+5+fvecoffset] = ff.hessianOfGaussianEigenvalues(data, scale)
        elif j == 4:
            fvec[fvecidx,0,:,:,1+7*i+5+fvecoffset:1+7*i+7+fvecoffset] = ff.structureTensorEigenvalues(data, 0.66*scale, scale)
        else:
            raise ValueError("WTF. j = %d" % j)

        return True

    def copy_data(data, fvec, fvecidx, fvecoffset):
        sleep(1) # hack to at least show the progress bar. yay, GIL.
        fvec[fvecidx,0,:,:,fvecoffset] = data
        return True

    futures.append(pool.submit(copy_data, data, fvec, fvecidx, fvecoffset))
    for i, scale in enumerate(scales):
        for j in range(5):
            futures.append(pool.submit(calc_features, data, fvec, fvecidx, fvecoffset, i, scale, j))

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('outfile')
    parser.add_argument('datadset')
    parser.add_argument('featdset')
    args = parser.parse_args()

    print("Loading image file")
    img = np.array(vigra.impex.readImage(args.infile).transposeToNumpyOrder())

    y,x,c = img.shape

    scales = [0.7, 1.0, 1.6, 3.5, 5, 10]
    n_features = 1 + 7*len(scales)

    print("Preparing h5 file")
    with h5.File(args.outfile) as f:
        if args.datadset in f:
            dset = f[args.datadset]
            if len(dset.shape) != 5: raise ValueError("shape ndim mismatch.")
            if dset.shape[1:] != (1,y,x,c): raise ValueError("shape yxc mismatch.")
            if dset.maxshape[0] != None: raise ValueError("Cannot extend dset.")
        else:
            f.create_dataset(args.datadset, (0,1,y,x,c), chunks=(1,1,256,256,c), maxshape=(None, 1, y, x, c))

        if args.featdset in f:
            dset = f[args.featdset]
            if len(dset.shape) != 5: raise ValueError("shape ndim mismatch.")
            if dset.shape[1:] != (1,y,x,n_features*c): raise ValueError("shape yxc mismatch.")
            if dset.maxshape[0] != None: raise ValueError("Cannot extend dset.")
        else:
            f.create_dataset(args.featdset, (0,1,y,x,n_features*c), chunks=(1,1,256,256,1), maxshape=(None, 1, y, x, n_features*c))



        data = f[args.datadset]
        fvec = f[args.featdset]


        data.resize(data.shape[0]+1, axis=0)
        fvec.resize(fvec.shape[0]+1, axis=0)

        data[data.shape[0]-1,0,:,:,:] = img

        pool = concurrent.futures.ThreadPoolExecutor()
        futures = []

        fvec_mem = np.zeros((1,1,y,x,n_features*c))

        print("Starting calc_all_features now.")
        for i_c in range(c):
            calc_all_features(pool, futures, np.ascontiguousarray(np.array(img[:,:,i_c])), fvec_mem, 0, n_features*i_c, scales)

        print("Waiting for all tasks to finish.")
        done = 0
        print_progress(done, len(futures))
        for x in concurrent.futures.as_completed(futures):
            x.result()
            done += 1
            print_progress(done, len(futures))

        print("Writing to output.")
        fvec[fvec.shape[0]-1,:,:,:,:] = fvec_mem[0,:,:,:,:]

        data.attrs["orig_%04d" % data.shape[0]] = np.string_(args.infile)
        

