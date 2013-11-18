import sys

import nibabel as nib
import numpy as np

def padNii(niiFileName, paddedNiiFileName, maskNiiFileName, padAmountMM='30'):
    """
    pad a nifti and save the nifti and the relevant mask.

    padNii(niiFileName, paddedNiiFileName, maskNiiFileName) pad nifti file
    niiFileName and save the resulting nifti in paddedNiiFileName, and the relevant mask in 
    maskNiiFileName. default padding is 30 mm.

    padNii(niiFileName, paddedNiiFileName, maskNiiFileName, padAmountMM) allows one to
    specify the padding amount in mm via padAmountMM.

    Example arguments:
    niiFileName = '10529_t1.nii.gz'
    paddedNiiFileName = 'padded_10529_t1.nii.gz'
    maskNiiFileName = 'padded_10529_t1_mask.nii.gz'
    padAmountMM = '30'; [default]
    """

    # padding amount in mm
    padAmountMM = int(padAmountMM)

    # load the nifti
    nii = nib.load(niiFileName)

    # get the amount of padding in voxels
    pixdim = nii.get_header()['pixdim'][1:4]
    padAmount = np.ceil(padAmountMM / pixdim)
    dims = nii.get_header()['dim'][1:4]
    assert np.all(dims.shape == padAmount.shape)
    newDims = dims + padAmount * 2

    # compute where the center is for padding
    center = newDims/2
    starts = np.round(center - dims/2)
    ends = starts + dims

    # compute a slice object with the start/end of the center subvolume
    slicer = [slice(start, end) for (start, end) in zip(starts, ends)]

    # set the subvolume in the center of the image w/the padding around it
    vol = np.zeros(newDims)
    vol[slicer] = nii.get_data()
    volMask = np.zeros(newDims)
    volMask[slicer] = np.ones(dims)

    # create niftis
    newNii = nib.Nifti1Image(vol, header=nii.get_header(), affine=nii.get_affine())
    newNiiMask = nib.Nifti1Image(volMask, header=nii.get_header(), affine=nii.get_affine())

    # save niftis
    newNii.to_filename(paddedNiiFileName)
    newNiiMask.to_filename(maskNiiFileName)

if __name__ == '__main__':
    if len(sys.argv) not in [4,5]:
        print("Usage: python padNii niiFileName paddedNiiFileName maskNiiFileName [padAmountMM=30]")
        print(len(sys.argv))
        sys.exit(1)
    padNii(*sys.argv[1:])
