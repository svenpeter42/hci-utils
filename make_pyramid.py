import h5py as h5
import numpy as np
import vigra
import argparse

from common import print_progress

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile")
    parser.add_argument('datadset')
    parser.add_argument('pyramidfile')
    parser.add_argument('pyramidgroup')
    parser.add_argument('--levels', default=4, type=int)
    args = parser.parse_args()

    with h5.File(args.datafile) as fdata, h5.File(args.pyramidfile) as fpyramid:
        data = fdata[args.datadset]

        x,y,z,c = data.shape
        assert(c == 1)

        # FIXME: could just calc correct shapes here to save time
        slice_z = data[:,:,0,0].astype(np.float32)
        p = vigra.ImagePyramid(slice_z)
        p.reduce(0, args.levels)


        grp = fpyramid.require_group(args.pyramidgroup)
        dsets = []
        for i_level in range(args.levels):
            lx, ly = p[i_level].shape
            dsets.append(grp.require_dataset("%02d" % i_level, dtype=np.float32, shape=(lx, ly, z, 1), chunks=(256, 256, 1, 1)))


        for i_z in range(z):
            print_progress(i_z, z)

            slice_z = data[:,:,i_z,0].astype(np.float32)
            p = vigra.ImagePyramid(slice_z)
            p.reduce(0, args.levels)

            for i_level, dset in enumerate(dsets):
                dset[:,:,i_z,0] = p[i_level]

        print_progress(z, z)