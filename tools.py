# TODO make it easy to generate filenames for experimentation
import os
import numpy as np
import nibabel as nib

# binary arrays for 1-neighbors and 2-neighbors
neighbors_0pix = np.array([[0,0,0], [0,1,0], [0,0,0]])
neighbors_1pix = np.array([[0,1,0], [1,1,1], [0,1,0]])
neighbors_2pix = np.array([[0,0,1,0,0], [0,1,1,1,0], [1,1,1,1,1], [0,1,1,1,0], [0,0,1,0,0]])

def test_stroke_location(stroke_image_file, label_image_file, output_file):
    stroke_image = nib.load(stroke_image_file)
    label_image = nib.load(label_image_file)


    binary_stroke = stroke_image.get_data() > 0
    labels = label_image.get_data()[binary_stroke]
    assert not np.all(labels == 0)
    return labels

def transform_label_map(label_map):
    """
    takes freesurfer label maps with lots of labels, and makes one with only 3 
    labels. 0 = background, 1 = non-ventricle brain, 2 = ventricle
    """

    new_label_map = np.zeros(label_map.shape)

    new_label_map[label_map != 0] = 1
    # ventricles are 4 and 43
    new_label_map[np.logical_or(label_map==4, label_map==43)] = 2

    return new_label_map

def custom_overlay(flair_img, seg_file, wmh_L, wmh_R, stroke, slices, out_png_file_prefix):
    """
    makes images of axial slices with labels overlaid on image
    """
    import ast
    if type(slices) is str:
        slices = ast.literal_eval(slices)
    #out_png_file_prefix = out_png_file_prefix.split('.', 1)[0]
    COLOR = [248, 117, 49] # orange
    import scipy.ndimage as ndimage
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    grid = []
    image = nib.load(flair_img)
    labels = nib.load(seg_file)
    foregrounds = [transform_label_map(labels.get_data()),
                   (nib.load(wmh_L).get_data() + nib.load(wmh_R).get_data()) > .5]
    try:
        stroke_img = nib.load(stroke).get_data() > .5
    except:
        stroke_img = np.zeros(image.get_data().shape)
    foregrounds.append(stroke_img)

    background = image.get_data()
    # intensity normalization
    mini = background.min()
    maxi = background.max()
    #for (image_file, labels_file) in zip(image_files, labels_files):
    for foreground in foregrounds:

        imgs = []
        for slice in slices:
            fg_slice = foreground[:, ::-1, slice].T
            bg_slice = (background[:, ::-1, slice].T - mini) / (maxi - mini) * 255
            fg_neighb_avg = ndimage.filters.generic_filter(fg_slice, np.mean, footprint=neighbors_1pix)
            fg_outline = fg_neighb_avg != fg_slice

            out_img = np.zeros(bg_slice.shape + (3,), dtype='uint8')
            out_img[:,:,:] = bg_slice[:,:,np.newaxis]
            # blending
            out_img[fg_outline] = out_img[fg_outline] * .7 + np.array(COLOR) * .3
            imgs.append(out_img[20:-20, 45:-45, :])
        grid.append(np.hstack(imgs))


    plt.imshow(np.vstack(grid), interpolation='none', aspect='auto')
    plt.axis('off')
    plt.savefig(out_png_file_prefix, bbox_inches='tight')

def orange_overlay(image_file, labels_file, slices, out_png_file_prefix):
    """
    makes images of axial slices with labels overlaid on image
    """
    out_png_file_prefix = out_png_file_prefix.split('.', 1)[0]
    COLOR = [248, 117, 49] # orange
    import scipy.ndimage as ndimage
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    image = nib.load(image_file)
    labels = nib.load(labels_file)

    foreground = transform_label_map(labels.get_data())
    background = image.get_data()
    # intensity normalization
    mini = background.min()
    maxi = background.max()

    for slice in xrange(image.shape[2]):
        fg_slice = foreground[:, ::-1, slice].T
        bg_slice = (background[:, ::-1, slice].T - mini) / (maxi - mini) * 255
        fg_neighb_avg = ndimage.filters.generic_filter(fg_slice, np.mean, footprint=neighbors_1pix)
        fg_outline = fg_neighb_avg != fg_slice

        out_img = np.zeros(bg_slice.shape + (3,), dtype='uint8')
        out_img[:,:,:] = bg_slice[:,:,np.newaxis]
        # blending
        out_img[fg_outline] = out_img[fg_outline] * .5 + COLOR * .5

        plt.imshow(out_img, interpolation='none', aspect='auto')
        plt.axis('off')
        plt.savefig(out_png_file_prefix + str(slice+1) + '.png', bbox_inches='tight')

def main():
    subjects = [x.strip() for x in open('/afs/csail.mit.edu/u/r/rameshvs/site00_subjs').readlines()]
    out = []
    for subj in subjects:
        dwiroi = '/data/vision/polina/projects/stroke/processed_datasets/2013_12_13/site00/{subj}/images/{subj}_dwi_roi_prep_pad.nii.gz'.format(subj=subj)
        labelfile = '/data/vision/polina/projects/stroke/processed_datasets/2013_12_13/site00/{subj}/images/buckner61_seg_IN_NONLINEAR_GAUSS_9000_0200__201x201x201_CC4_RIGID_MI32_MASKED_{subj}_dwi_img_prep_pad-.nii.gz'.format(subj=subj)
        if not (os.path.exists(dwiroi) and os.path.exists(labelfile)):
            continue
        try:
            labels = test_stroke_location(dwiroi, labelfile,'')
        except AssertionError:
            print(subj)
        out.append(labels)
    return out

def freq(arr):
    """ Returns all values in the array with their frequencies """
    (values, indices) = np.unique(arr, return_inverse=True)
    counts = np.bincount(indices)
    sort_idx = np.argsort(counts)[::-1]
    assert counts.sum() == arr.size
    return (values[sort_idx], counts[sort_idx])

    # for (count, value) in zip(counts[sort_idx], values[sort_idx]):
    #     f.

    # wm = np.mean(np.logical_or(labels==41, labels==2))
    # gm = np.mean(np.logical_or(labels==42, labels==3))

    # with open(output_file, 'w') as f:
    #     f.write('{:0.3f}\t{:0.3f}\n'.format(wm, gm))
