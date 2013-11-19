import sys

import numpy as np
import scipy.cluster.vq as vq
import nibabel as nib

def matchWM(alignedInFile, maskFile, inFile, wmiSrc, outFile):
    """
    heuristic for matching white matter intensities
    niiOut = matchWM(alignedInFile, maskFile, inFile, wmiSrc, outFile) matches the
    white matter intensities in wmiSrc. This is done by taking a file (alignedInFile)
     that is rigid-aligned to the atlas for which we have a mask (maskFile). We
     take the intensities of the alignedInFile within the mask, and match a
     2-component mixture model to the intensities, and take the higher component
     to represent the white matter. We now use the expected white matter
     intensity (wmiSrc) and divide it by the center of the second mixture to get a
     multiplication factor. We then multiply this factor with the voxels in the
     nifti file inFile, the initial (unaligned) image (or whatever you want). The
     output is saved in outFile.

     TODO: Note that the output is capped at uint8 since the atlas was originally
      uint8. This should be fixed depending on the atlas.

      Inputs:
          alignedInFile - input file that's aligned in the b51 framse
              e.g. 11805_t1_img_prep_bcorr_IN_MI_buckner51-.nii.gz
          maskFile - atlas mask file.
              e.g. buckner51_fixed_mask_in_seg.nii.gz
          inFile - the file to alter
              e.g. 11805_t1_img_prep_bcorr.nii.gz
          wmiSrc - the intensity of WM cluster in atlas, or alternatively the b51 file.
              e.g. 121.0683 (faster/preferred) OR buckner51.nii.gz
          outFile - the output file
              e.g. 11805_t1_img_prep_bcorr_mult.nii.gz

    Original author: Adrian Dalca (ported to python by Ramesh Sridharan)
    """
    niimask = nib.load(maskFile)
    mask = niimask.get_data() > 0

    # approximately get the white matter intensity
    if type(wmiSrc) is str:
        nii = nib.load(wmiSrc)
        vox = nii.get_data()[mask]
        (means, error) = vq.kmeans(vox, 2)
        wmIntensity = np.max(means)
    else:
        wmIntensity = wmiSrc

    # get the voxels within the mask
    niiAligned = nib.load(alignedInFile)
    vox = niiAligned.get_data()[mask]

    # get 2 starting points and do k-means
    start = np.percentile(vox, [10, 90])
    (means, error) = vq.kmeans(vox, np.array(start))

    # update the input file's intensities. Note that the output is capped at uint8
    niiIn = nib.load(inFile)
    clipped = (niiIn.get_data() * wmIntensity / np.max(means))
    clipped[clipped > 255] = 255
    niiOut = nib.Nifti1Image(clipped.astype('uint8'),
                             affine=niiIn.get_affine(),
                             header=niiIn.get_header())

    # save output file
    niiOut.to_filename(outFile)
    return niiOut

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage: python matchWM.py alignedInFile maskFile inFile wmiSrc outFile")
        sys.exit(1)
    matchWM(*sys.argv[1:])
