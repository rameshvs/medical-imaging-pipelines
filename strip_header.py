#!/usr/bin/env python
"""
WARNING: this file assumes the data matrix is rotated 180 degrees
in the xy plane, and hardcodes that assumption in. There are some limited
checks (see assertions), but no guarantees.
"""
from __future__ import print_function
import sys
import shutil

import nibabel as nib
import numpy as np

def unheader(in_file, out_file, permutation, rotate=True):
    shutil.copy(in_file, out_file)
    return
    # load in
    nii = nib.load(in_file)
    header = nii.get_header()
    affine = nii.get_affine()
    data = nii.get_data()
    pixdim = header['pixdim'][1:4]

    # permute axes
    # TODO figure out geometry
    # axis_permutation = np.argmax(np.abs(affine[:3,:3]), 1)
    axis_permutation = permutation
    data = data.transpose(axis_permutation)
    print(data.shape)

    pixdim = pixdim[axis_permutation]
    print(pixdim)

    # rotate data in xy
    if rotate:
        new_data = data[::-1, ::-1, :]
    else:
        new_data = data[::-1, ::-1, ::-1]

    # sanity checks
    direction = np.dot(affine[:3, :3], [1,1,1])
    print(direction)
    # if rotate:
    #     assert np.all(np.sign(direction) == [-1, -1, 1])
    # else:
    #     assert np.all(np.sign(direction) == [1, 1, 1])
    assert np.allclose(np.linalg.det(affine[:3,:3]), np.prod(pixdim))

    # set up the new affine matrix
    new_aff = np.diag(pixdim)
    offset = -header['dim'][1:4]/2
    full_affine = np.zeros([4, 4])
    full_affine[:3, :3] = new_aff
    full_affine[:3, 3] = offset
    full_affine[3,3] = 1

    out = nib.Nifti1Image(new_data, affine=full_affine, header=header)
    out.to_filename(out_file)

if __name__ == '__main__':
    modality = sys.argv[3]
    if modality == 't1':
        #permutation = [1,2,0]
        permutation = [2,0,1]
        rotate = False
    elif modality == 'flair':
        permutation = [0,1,2]
        rotate = True
    else:
        permutation = [0,1,2]
        rotate = False
    unheader(sys.argv[1], sys.argv[2], permutation, rotate)

