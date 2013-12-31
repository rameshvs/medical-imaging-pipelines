#!/usr/bin/env python
# This script implements the registration pipeline described in the paper:
#
# Quantification and Analaysis of Large Multimodal Clinical Image Studies:
# Application to Stroke, by Sridharan, Dalca et al.
#
# For questions, please contact {rameshvs,adalca}@csail.mit.edu.

import pipebuild as pb
import os
import sys
import subprocess
import datetime
import time

ATLAS_MODALITY = 't1'

features_by_modality = {'dwi': ['img', 'roi'], 'flair': ['img', 'wmh_L', 'wmh_R']}
#features_by_modality = {'dwi': ['img'], 'flair': ['img']}
mask_collection = {}
CLOBBER_EXISTING_OUTPUTS = False
cwd = os.path.dirname(os.path.abspath(__file__))

# Where the files live
#BASE =            os.path.join(NFS_ROOT, 'projects/stroke/processed_datasets/2013_12_02/site16')
#BASE =            os.path.join(pb.NFS_ROOT, 'projects/stroke/processed_datasets/2013_05_16')
# Where the atlas is stored
ATLAS_BASE =      os.path.join(pb.NFS_ROOT, 'projects/stroke/work/input/atlases/')

ATLAS_SUBJ_NAME = 'buckner61'

if __name__ == '__main__':

    ########################
    ### Argument parsing ###
    ########################
    USAGE = '%s <subj> <smoothness regularization> <field regularization> <out folder>' % sys.argv[0]
    subj = sys.argv[1]
    # Regularization parameters for ANTS
    regularization = float(sys.argv[2])
    regularization2 = float(sys.argv[3])

    # where the data lives
    data_subfolder = sys.argv[4]

    BASE = os.path.join(pb.NFS_ROOT, 'projects/stroke/processed_datasets', data_subfolder)

    #############################
    ### Set up atlas and data ###
    #############################

    ## Atlas
    atlas = pb.Atlas(ATLAS_SUBJ_NAME, ATLAS_BASE)
    stroke_atlas_files = {
        'histogram': 'Hist_cut.mat',           # histogram for equalization
        'seg': '_seg.nii.gz',                  # full segmentation
        'wmh_mask': '_seg_wm_region.nii.gz',   # periventricular area for WMH
        'wmh_prior': '_wmh_prior_100.nii.gz',  # prior from averaging subjs
        'wmh_L': '_wmh_L_average.nii.gz',      # prior in left hemisphere
        'wmh_R': '_wmh_R_average.nii.gz',      # prior in right hemisphere
        'mask': '_fixed_mask_from_seg_binary.nii.gz'} # brain mask

    for (name, file_suffix) in stroke_atlas_files.iteritems():
        atlas.add_file(name, file_suffix)

    ## Subject data
    dataset = pb.Dataset(BASE, atlas)

    #############################
    ### Registration pipeline ###
    #############################
    t1_modifiers = ''

    ###### Fix problems with T1 images caused by use of Analyze format
    original_t1 = dataset.get_file(subj, 't1', 'img', t1_modifiers + '_prep')
    unheader = pb.InputOutputShellCommand(
                    "Remove header from T1",
                    cmdName=os.path.join(cwd, 'strip_header.py'),
                    input=dataset.get_original_file(subj,'t1', 'raw'),
                    output=original_t1,
                    other='t1'
                    )
    t1_modifiers += '_prep'

    ###### N4 bias correction of T1.
    n4 = pb.N4Command("N4 bias field correction for T1",
                    input=dataset.get_file(subj, 't1', 'img', t1_modifiers),
                    output=dataset.get_file(subj, 't1', 'img', t1_modifiers + '_bcorr'))
    t1_modifiers += '_bcorr'


    ###### Initial rigid registration
    fixed = dataset.get_file(subj, ATLAS_MODALITY, 'img', t1_modifiers)
    moving = atlas.get_file('img')
    first_stage_metric = 'MI'
    forward_reg_init_affine = pb.ANTSCommand("Initial affine step for "
            "forward atlas->%s registration w/%s" % (ATLAS_MODALITY, first_stage_metric),
                                        moving=moving,
                                        fixed=fixed,
                                        metric=first_stage_metric,
                                        regularization='Gauss[4.5,0]',
                                        method='rigid',
                                        mask=False)
    backward_reg_init_affine = pb.ANTSCommand("Initial affine step for "
            "backwards %s->atlas registration w/%s" % (ATLAS_MODALITY, first_stage_metric),
                                        moving=fixed,
                                        fixed=moving,
                                        metric=first_stage_metric,
                                        regularization='Gauss[4.5,0]',
                                        method='rigid',
                                        mask=False)

    init_warp_args = pb.make_warp_kwargs(
                            dataset.get_file(subj, ATLAS_MODALITY, 'img', t1_modifiers),
                            atlas.get_file('img'),
                            [backward_reg_init_affine],
                            ['forward'])

    backward_reg_warp_f = pb.WarpCommand("Warp subject into atlas space using "
            "init affine backwards %s -> atlas registration" % ATLAS_MODALITY,
                                    **init_warp_args)


    init_warp_mask_args = pb.make_warp_kwargs(
                            atlas.get_file('mask'),
                            dataset.get_file(subj, ATLAS_MODALITY, 'img', t1_modifiers),
                            [backward_reg_init_affine],
                            ['inverse'])

    forward_reg_warp_f = pb.WarpCommand("Warp atlas mask into subj space "
            "using init affine backwards %s -> atlas registration" % ATLAS_MODALITY,
                        **init_warp_mask_args)

    ### Intensity matching (can't do histogram equalization)
    # TODO use more robust mean-shift based mode-matching

    match_wm = pb.PyMatchWMCommand("Match white matter intensity values",
                                    alignedInFile=init_warp_args['output'],
                                    maskFile=atlas.get_file('mask'),
                                    inFile=pb.get_file(subj, ATLAS_MODALITY, 'img', t1_modifiers),
                                    wmiSrc=atlas.get_file('img'),
                                    output=pb.get_file(subj, ATLAS_MODALITY, 'img', t1_modifiers + '_matchwm'))

    t1_modifiers += '_matchwm'

    t1_modifiers_this = t1_modifiers

    # moving = dataset.get_file(subj, modality, 'img', modifiers[modality])
    second_stage_metric = 'CC'
    second_stage_radiusBins = 4

    subject_img = dataset.get_file(subj, ATLAS_MODALITY, 'img', t1_modifiers_this)

    ###### Final atlas -> subject registration
    forward_reg_full = pb.ANTSCommand("Forward atlas -> %s registration with "
            "rough mask, w/%s. initialize affine without doing any more affine "
            "steps. " % (ATLAS_MODALITY, second_stage_metric),
                                    moving=atlas.get_file('img'),
                                    fixed=subject_img,
                                    metric=second_stage_metric,
                                    radiusBins=second_stage_radiusBins,
                                    regularization='Gauss[%0.3f,%0.3f]' % (regularization,regularization2),
                                    #regularization='Gauss[%0.1f,%0.1f]' % (regularization,regularization2),
                                    method='201x201x201',
                                    init=forward_reg_init_affine.affine)


    for atlas_feature in ['img', 'mask', 'seg', 'wmh_mask', 'wmh_L', 'wmh_R']:
        warp_forward_args = pb.make_warp_kwargs(atlas.get_file(atlas_feature),
                                    subject_img,
                                    [forward_reg_full],
                                    ['forward'])
        useNN = atlas_feature in ['mask', 'seg']
        backward_reg_warp_mask_b = pb.WarpCommand("Warp atlas %s into subject "
                "space using full forward atlas -> %s registration and "
                "%s NN interpolation" % (atlas_feature, ATLAS_MODALITY, useNN),
                                            useNN=useNN,
                                            **warp_forward_args)
        if atlas_feature == 'mask':
            atlas_mask_in_t1 = warp_forward_args['output']

    warp_backward_args = pb.make_warp_kwargs(subject_img,
                                    atlas.get_file('img'),
                                    [forward_reg_full],
                                    ['inverse'])
    backward_reg_warp_f = pb.WarpCommand("Warp subject into atlas space using"
            "full forward atlas -> %s registration" % ATLAS_MODALITY,
                                    useNN=False,
                                    **warp_backward_args)


    ###### Rigid ANTS registration DWI/FLAIR --> T1
    reg_to_t1 = {} # dict containing the ANTSCommand objects by modality
    for modality in ['flair', 'dwi']:
        if not os.path.exists(dataset.get_original_file(subj, modality, 'img')):
            # Quit if the subject is missing data. TODO support partial script execution
            continue
        for feature in features_by_modality[modality]:
            if feature != 'wmh_LR':
                modifiers = ''
                unheader = pb.InputOutputShellCommand(
                                "Don't remove header from %s %s to fix analyze issues" % (modality,feature),
                                #cmdName=os.path.join(cwd, 'strip_header.py'),
                                cmdName='cp',
                                input=dataset.get_original_file(subj,modality, feature),
                                output=dataset.get_file(subj, modality, feature, modifiers + '_prep'),
                                )
                modifiers = '_prep'

                pad = pb.PyPadCommand("Pad %s %s" % (modality, feature),
                                    input=dataset.get_file(subj, modality, feature, modifiers),
                                    #input=dataset.get_original_file(subj,modality, feature),
                                    output=dataset.get_file(subj, modality, feature, modifiers+'_pad'),
                                    out_mask=dataset.get_file(subj, modality, feature, modifiers+'_padmask_seg'))
        modifiers = '_prep_pad'

        reg_to_t1_init = pb.ANTSCommand("Rigid intrasubject/multimodal "
                "registration of %s to T1: initialize w/o mask" % modality,
                                moving=dataset.get_file(subj, modality, 'img', modifiers),
                                fixed=dataset.get_file(subj, 't1', 'img', t1_modifiers_this),
                                metric='MI',
                                regularization='Gauss[4.5,0]',
                                method='rigid')

        reg_to_t1[modality] = pb.ANTSCommand("Rigid intrasubject/multimodal "
                "registration of %s to T1: continue with mask" % modality,
                                        moving=dataset.get_file(subj, modality, 'img', modifiers),
                                        fixed=dataset.get_file(subj, 't1', 'img', t1_modifiers_this),
                                        metric='MI',
                                        regularization='Gauss[4.5,0]',
                                        method='rigid',
                                        mask=atlas_mask_in_t1,
                                        cont=reg_to_t1_init.affine)

        ### Warp subject stuff into common space for spatial analysis
        for feature in features_by_modality[modality]:
            warp_to_t1_args = pb.make_warp_kwargs(dataset.get_file(subj, modality, feature, modifiers),
                                                        dataset.get_file(subj, 't1', 'img', t1_modifiers_this),
                                                        [reg_to_t1[modality]],
                                                        ['forward'])

            warp_to_t1 = pb.WarpCommand("Warp %s %s into t1 space" % (modality,feature),
                                        useNN=(feature not in 'img'),
                                        **warp_to_t1_args)


            warp_to_atlas_args = pb.make_warp_kwargs(dataset.get_file(subj, modality, feature, modifiers),
                                               atlas.get_file('img'),
                                               [reg_to_t1[modality], forward_reg_full],
                                               ['forward', 'inverse'])

            warp_to_atlas = pb.WarpCommand("Warp %s %s into atlas space" % (modality,feature),
                                     useNN=(feature not in 'img'),
                                     **warp_to_atlas_args)

        ### Warp atlas stuff into subject space for help with segmentation
        for atlas_feature in ['img', 'mask', 'seg', 'wmh_mask', 'wmh_prior', 'wmh_L', 'wmh_R']:

            warp_from_atlas_args = pb.make_warp_kwargs(atlas.get_file(atlas_feature),
                                                    dataset.get_file(subj, modality, 'img', modifiers),
                                                    [forward_reg_full, reg_to_t1[modality]],
                                                    ['forward', 'inverse'])

            warp_from_atlas = pb.WarpCommand("Warp atlas %s into subject "
                    "%s space" % (atlas_feature, modality),
                                          useNN=(atlas_feature == 'seg'),
                                          **warp_from_atlas_args)

            if atlas_feature == 'mask':
                mask_collection[modality] = warp_from_atlas_args['output']

    #############################
    # warping between dwi and flair using the indirect through-t1 registrations
    # Warp dwi roi (stroke) to flair
    if 'dwi' in reg_to_t1 and 'flair' in reg_to_t1:


        for feature in features_by_modality['flair']:
            moving = dataset.get_file(subj, 'flair', feature, '_prep_pad')
            reference = dataset.get_file(subj, 'dwi', 'img', '_prep_pad')

            warp_flair_dwi_args = pb.make_warp_kwargs(moving,
                                                   reference,
                                                   [reg_to_t1['flair'], reg_to_t1['dwi']],
                                                   ['forward', 'inverse'])
            warp_flair_to_dwi = pb.WarpCommand("Warp FLAIR %s into DWI by going through T1 reg" % feature,
                                            useNN=(feature != 'img'),
                                            **warp_flair_dwi_args)

        for feature in features_by_modality['dwi']:
            moving = dataset.get_file(subj, 'dwi', feature, '_prep_pad')
            reference = dataset.get_file(subj, 'flair', 'img', '_prep_pad')

            warp_dwi_flair_args = pb.make_warp_kwargs(moving,
                                                   reference,
                                                   [reg_to_t1['dwi'], reg_to_t1['flair']],
                                                   ['forward', 'inverse'])
            warp_dwi_to_flair = pb.WarpCommand("Warp DWI %s into FLAIR by going through T1 reg" % feature,
                                            useNN=(feature != 'img'),
                                            **warp_dwi_flair_args)

        if 'roi' in features_by_modality['dwi']:
            warp_flair_dwi_args = pb.make_warp_kwargs(dataset.get_file(subj, 'dwi', 'roi', modifiers),
                                                   dataset.get_file(subj, 'flair', 'img', modifiers),
                                                   [reg_to_t1['dwi'], reg_to_t1['flair']],
                                                   ['forward', 'inverse'])
            warp_dwi_to_flair = pb.WarpCommand("Warp DWI ROI (stroke) into FLAIR space",
                                            useNN=False,
                                            **warp_flair_dwi_args)

    # warp segmentations back into atlas space
    if 'flair' in reg_to_t1:
        moving = dataset.get_file(subj, 'flair', 'wmh', '_CALL_prep_pad-MATLAB_WM_corr')
        reference = atlas.get_file('img')

        warp_wmh_back_args = pb.make_warp_kwargs(moving, reference, [reg_to_t1['flair'], forward_reg_full], ['forward', 'inverse'])
        warp_wmh_back = pb.WarpCommand("Warp WMH back into atlas space", useNN=True, **warp_wmh_back_args)

    for path in [os.path.join(BASE,subj,'images'),
            os.path.join(BASE,subj,'images','reg'),
            dataset.get_sge_folder(subj)]:
        try:
            os.mkdir(path)
        except:
            pass

    ### Generate script file and SGE qsub file
    time.sleep(1) # sleep so that timestamps don't clash
    timestamp = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    out_script = os.path.join(dataset.get_sge_folder(subj), 'pipeline.%s.sh' % timestamp)
    pb.Command.generate_code(out_script, clobber_existing_outputs=CLOBBER_EXISTING_OUTPUTS)

    ## Prep for SGE
    out_qsub = out_script + '.qsub'
    os.environ['SGE_LOG_PATH'] = dataset.get_sge_folder(subj)
    with open(out_qsub,'w') as out_qsub_file:
        subprocess.call([pb.QSUB_RUN, '-c', out_script], stdout=out_qsub_file)

    print(out_qsub)
    subprocess.call([pb.QSUB, out_qsub])

