import sys

import numpy as np
import scipy.cluster.vq as vq
import scipy.stats as stats
import nibabel as nib

def matchWM(inFile, maskFile, modeIntensity, outFile):
    """
    heuristic for matching white matter intensities
    niiOut = matchWM(inFile, maskFile, intensity, outFile) scales the intensities
    in inFile so that the mode of intensity (computed within the mask given by maskFile)
    is modeIntensity.

    Currently computes the mode using k-means clustering with k=2; this will
    usually work for matching white matter intensities but may not work for
    other applications.

    Future versions of this python code will use mean-shift clustering.

      Inputs:
          inFile - input file
          maskFile - mask file; only intensities within the mask are used.
          modeIntensity - the desired mode intensity as an integer or string with an integer
          outFile - the output file

    Original author: Adrian Dalca (ported to python by Ramesh Sridharan)
    """
    # TODO: Note that the output is capped at uint8 since the atlas was originally
    # uint8. This should be fixed depending on the atlas.

    niimask = nib.load(maskFile)
    mask = niimask.get_data() > 0

    # approximately get the white matter intensity
    if type(modeIntensity) is str:
        wmIntensity = int(modeIntensity)
    else:
        wmIntensity = modeIntensity

    # get the voxels within the mask
    nii = nib.load(inFile)
    vol = nii.get_data()
    masked_voxels = vol[mask]

    # get 2 starting points and do k-means
    start = np.percentile(masked_voxels, [10, 90])
    (means, error) = vq.kmeans(masked_voxels, np.array(start))

    # find the cluster with the most voxels
    assignments = np.argmin((masked_voxels[:, np.newaxis] - means[np.newaxis, :])**2, 1)
    (biggest_cluster, _) = stats.mode(assignments)

    # update the input file's intensities. Note that the output is capped at uint8
    clipped = (vol * wmIntensity / means[int(biggest_cluster)])
    clipped[clipped > 255] = 255
    niiOut = nib.Nifti1Image(clipped.astype('uint8'),
                             affine=nii.get_affine(),
                             header=nii.get_header())

    # save output file
    niiOut.to_filename(outFile)
    return niiOut

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python matchWM.py alignedInFile maskFile inFile wmiSrc outFile")
        sys.exit(1)
    matchWM(*sys.argv[1:])
