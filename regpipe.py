#!/usr/bin/env python

# This script implements the registration pipeline described in the paper:
#
# Quantification and Analysis of Large Multimodal Clinical Image Studies:
# Application to Stroke, by Sridharan, Dalca et al.
#
# For questions, please contact {rameshvs,adalca}@csail.mit.edu.

import pipebuild as pb

import os
import sys
import subprocess
import datetime
import time

cwd = os.path.dirname(os.path.abspath(__file__))

ATLAS_MODALITY = 't1'

#features_by_modality = {'dwi': ['img', 'roi'], 'flair': ['img', 'wmh_L', 'wmh_R']}
features_by_modality = {'dwi': ['img'], 'flair': ['img']}
CLOBBER_EXISTING_OUTPUTS = False

DATA_ROOT = os.path.join(pb.NFS_ROOT, 'projects/stroke')

ATLAS_BASE = os.path.join(DATA_ROOT, 'work/input/atlases/')
ATLAS_NAME = 'buckner61'

if __name__ == '__main__':

    ########################
    ### Argument parsing ###
    ########################
    USAGE = '%s <subj> <smoothness regularization> <field regularization> <out folder>' % sys.argv[0]

    if len(sys.argv != 5):
        print(USAGE)
        sys.exit(1)

    subj = sys.argv[1]
    # Regularization parameters for ANTS
    regularization = float(sys.argv[2])
    regularization2 = float(sys.argv[3])

    # where the data lives
    data_subfolder = sys.argv[4]

    #############################
    ### Set up atlas and data ###
    #############################

    BASE = os.path.join(DATA_ROOT, 'processed_datasets', data_subfolder)
    ## Atlas
    atlas = pb.Atlas(ATLAS_NAME, ATLAS_BASE)
    stroke_atlas_files = {
        'seg': '_seg.nii.gz',                  # full segmentation
        'wm_region': '_seg_wm_region.nii.gz',  # periventricular area for WMH
        'wmh_prior': '_wmh_prior_100.nii.gz',  # prior from averaging subjs
        'wmh_L': '_wmh_L_average.nii.gz',      # prior in left hemisphere
        'wmh_R': '_wmh_R_average.nii.gz',      # prior in right hemisphere
        'mask': '_fixed_mask_from_seg_binary.nii.gz'} # brain mask

    for (name, file_suffix) in stroke_atlas_files.iteritems():
        atlas.add_file(name, file_suffix)

    ## Subject data
    dataset = pb.Dataset(
                BASE,
                atlas,
                # How are the inputs to the pipeline stored?
                os.path.join(BASE , '{subj}/original/{modality}_1/{subj}_{modality}_{feature}.nii.gz'),
                # How should intermediate files be stored?
                os.path.join(BASE, '{subj}/images/{subj}_{modality}_{feature}{modifiers}.nii.gz'))

    #############################
    ### Registration pipeline ###
    #############################
    t1_modifiers = '' # keeps track of what's been done so far for filenames

    ###### Fix problems with T1 images caused by use of Analyze format
    original_t1 = dataset.get_file(subj, 't1', 'img', t1_modifiers + '_prep')
    pb.InputOutputShellCommand(
                    "Remove header from T1",
                    cmdName=os.path.join(cwd, 'strip_header.py'),
                    input=dataset.get_original_file(subj,'t1', 'raw'),
                    output=original_t1,
                    extra_args='t1'
                    )
    t1_modifiers += '_prep'

    ###### N4 bias correction of T1.
    pb.N4Command("N4 bias field correction for T1",
                    input=dataset.get_file(subj, 't1', 'img', t1_modifiers),
                    output=dataset.get_file(subj, 't1', 'img', t1_modifiers + '_bcorr'))
    t1_modifiers += '_bcorr'


    ###### Initial rigid registration
    initial_affine_reg = pb.ANTSCommand(
            "Initial affine registration step: atlas->subj T1",
            moving=atlas.get_file('img'),
            fixed=dataset.get_file(subj, 't1', 'img', t1_modifiers),
            metric='MI',
            method='rigid')

    mask_warp = pb.ANTSWarpCommand.make_from_registration(
            "Warp atlas mask into subj space using initial"
            "affine subj T1->atlas registration",
            atlas.get_file('mask'),
            dataset.get_file(subj, 't1', 'img', t1_modifiers),
            [initial_affine_reg],
            ['forward'])

    ###### Intensity matching (can't do histogram equalization)
    # TODO use more robust mean-shift based mode-matching
    pb.PyMatchWMCommand("Match white matter intensity values",
                         inFile=dataset.get_file(subj, 't1', 'img', t1_modifiers),
                         maskFile=mask_warp.outfiles[0],
                         intensity='138',
                         output=dataset.get_file(subj, 't1', 'img', t1_modifiers + '_matchwm'))

    # pb.PyMatchWMCommand("Match white matter intensity values",
    #                     alignedInFile=init_warp_args['output'],
    #                     maskFile=atlas.get_file('mask'),
    #                     inFile=dataset.get_file(subj, 't1', 'img', t1_modifiers),
    #                     wmiSrc=atlas.get_file('img'),
    #                     output=dataset.get_file(subj, 't1', 'img', t1_modifiers + '_matchwm'))

    t1_modifiers += '_matchwm'

    subject_img = dataset.get_file(subj, 't1', 'img', t1_modifiers)

    ###### Final atlas -> subject registration
    forward_reg_full = pb.ANTSCommand("Forward atlas->subj T1 registration with "
            "rough mask & CC. initialize affine w/o doing any more affine steps.",
            moving=atlas.get_file('img'),
            fixed=subject_img,
            metric='CC',
            radiusBins=4,
            regularization='Gauss[%0.3f,%0.3f]' % (regularization,regularization2),
            method='201x201x201',
            init=initial_affine_reg.affine)

    ### Warp quantities of interest for visualization and future analysis
    for atlas_feature in ['img', 'mask', 'seg', 'wm_region', 'wmh_L', 'wmh_R']:

        warp_atlas_to_t1 = pb.ANTSWarpCommand.make_from_registration(
                "Warp atlas %s into subject space using full forward"
                "atlas -> t1 registration" % atlas_feature,
                atlas.get_file(atlas_feature),
                subject_img,
                [forward_reg_full],
                ['forward'],
                useNN=(atlas_feature in ['mask', 'seg']))

        if atlas_feature == 'mask':
            atlas_mask_in_t1 = warp_atlas_to_t1.outfiles[0]

    pb.ANTSWarpCommand.make_from_registration(
            "Warp subject into atlas space using"
            "full forward atlas->subj T1 registration",
            subject_img,
            atlas.get_file('img'),
            [forward_reg_full],
            ['inverse'],
            useNN=False)

    ###### Rigid ANTS registration DWI/FLAIR --> T1
    multimodal_t1_registrations = {} # dict containing the ANTSCommand objects by modality

    for modality in ['flair', 'dwi']:
        if not os.path.exists(dataset.get_original_file(subj, modality, 'img')):
            # Quit if the subject is missing data. TODO support partial script execution
            continue
        for feature in features_by_modality[modality]:
            if feature != 'wmh_LR':
                modifiers = ''

                pb.PyPadCommand("Pad %s %s" % (modality, feature),
                                    input=dataset.get_original_file(subj, modality, feature),
                                    output=dataset.get_file(subj, modality, feature, modifiers+'_pad'),
                                    out_mask=dataset.get_file(subj, modality, feature, modifiers+'_padmask_seg'))
        modifiers = '_pad'

        reg_to_t1_init = pb.ANTSCommand("Rigid intrasubject/multimodal "
                "registration of %s to T1: initialize w/o mask" % modality,
                                moving=dataset.get_file(subj, modality, 'img', modifiers),
                                fixed=dataset.get_file(subj, 't1', 'img', t1_modifiers),
                                metric='MI',
                                method='rigid')

        multimodal_t1_registrations[modality] = pb.ANTSCommand("Rigid intrasubject/multimodal "
                "registration of %s to T1: continue with mask" % modality,
                                        moving=dataset.get_file(subj, modality, 'img', modifiers),
                                        fixed=dataset.get_file(subj, 't1', 'img', t1_modifiers),
                                        metric='MI',
                                        method='rigid',
                                        mask=atlas_mask_in_t1,
                                        cont=reg_to_t1_init.affine)

        ### Warp subject stuff into common space for spatial analysis
        for feature in features_by_modality[modality]:

            warp_to_t1 = pb.ANTSWarpCommand.make_from_registration(
                    "Warp {} {} into t1 space".format(modality,feature),
                    dataset.get_file(subj, modality, feature, modifiers),
                    dataset.get_file(subj, 't1', 'img', t1_modifiers),
                    [multimodal_t1_registrations[modality]],
                    ['forward'])

            warp_to_atlas = pb.ANTSWarpCommand.make_from_registration(
                    "Warp {} {} into atlas space".format(modality, feature),
                    dataset.get_file(subj, modality, feature, modifiers),
                    atlas.get_file('img'),
                    [multimodal_t1_registrations[modality], forward_reg_full],
                    ['forward', 'inverse'])



        ### Warp atlas stuff into subject space for help with segmentation
        for atlas_feature in ['img', 'mask', 'seg', 'wm_region', 'wmh_prior', 'wmh_L', 'wmh_R']:

            warp_from_atlas = pb.ANTSWarpCommand.make_from_registration(
                    "Warp atlas {} into subject {} space".format(atlas_feature, modality),
                    moving=atlas.get_file(atlas_feature),
                    reference=dataset.get_file(subj, modality, 'img', modifiers),
                    reg_sequence=[forward_reg_full, multimodal_t1_registrations[modality]],
                    inversion_sequence=['forward', 'inverse'])

    #############################
    # warping between dwi and flair using the indirect through-t1 registrations
    if 'dwi' in multimodal_t1_registrations and 'flair' in multimodal_t1_registrations:

        for feature in features_by_modality['flair']:

           pb.ANTSWarpCommand.make_from_registration(
                    "Warp FLAIR {} into DWI using through-T1 reg".format(feature),
                    moving = dataset.get_file(subj, 'flair', feature, modifiers),
                    reference = dataset.get_file(subj, 'dwi', 'img', modifiers),
                    reg_sequence = [multimodal_t1_registrations['flair'], multimodal_t1_registrations['dwi']],
                    inversion_sequence = ['forward', 'inverse'],
                    useNN = (feature != 'img')) # nearest neighbor for all non-image features

        for feature in features_by_modality['dwi']:

            pb.ANTSWarpCommand.make_from_registration(
                    "Warp DWI {} into FLAIR using through-T1 reg".format(feature),
                    moving = dataset.get_file(subj, 'dwi', feature, modifiers),
                    reference = dataset.get_file(subj, 'flair', 'img', modifiers),
                    reg_sequence = [multimodal_t1_registrations['dwi'], multimodal_t1_registrations['flair']],
                    inversion_sequence = ['forward', 'inverse'],
                    useNN = (feature != 'img')) # nearest neighbor for all non-image features


    # warp segmentations back into atlas space
    if 'flair' in multimodal_t1_registrations:

        warp_wmh_back = pb.ANTSWarpCommand.make_from_registration(
                "Warp WMH back into atlas space",
                dataset.get_file(subj, 'flair', 'wmh', '_CALL_pad-MATLAB_WM_corr'),
                atlas.get_file('img'),
                [multimodal_t1_registrations['flair'], forward_reg_full],
                ['forward', 'inverse'],
                useNN=True)

    for path in [os.path.join(BASE,subj,'images'),
            os.path.join(BASE,subj,'images','reg'),
            dataset.get_log_folder(subj)]:
        try:
            os.mkdir(path)
        except:
            pass

    ### Generate script file and SGE qsub file
    time.sleep(1) # sleep so that timestamps don't clash, SGE isn't overloaded
    timestamp = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    out_script = os.path.join(dataset.get_log_folder(subj), 'pipeline.%s.sh' % timestamp)
    pb.Command.generate_code(out_script, clobber_existing_outputs=CLOBBER_EXISTING_OUTPUTS)

    ## Prep for SGE
    out_qsub = out_script + '.qsub'
    os.environ['SGE_LOG_PATH'] = dataset.get_log_folder(subj)
    with open(out_qsub,'w') as out_qsub_file:
        subprocess.call([pb.QSUB_RUN, '-c', out_script], stdout=out_qsub_file)

    print(out_qsub)
    subprocess.call([pb.QSUB, out_qsub])

