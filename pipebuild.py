# This module provides a general set of tools for implementing medical imaging
# pipelines. In particular, it was used to build the registration pipeline
# described in the paper:
#
# Quantification and Analaysis of Large Multimodal Clinical Image Studies:
# Application to Stroke, by Sridharan, Dalca et al.
#
# For questions, please contact {rameshvs,adalca}@csail.mit.edu.

from __future__ import print_function
import re
import os
import numpy as np

import json

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

##############################################
### System paths (you should change these) ###
##############################################

# On our system, everything lives on an NFS drive.
NFS_ROOT = '/path/to/NFS/drive/'
# ANTS build directory
ANTSPATH =        os.path.join(NFS_ROOT, 'shared_software/ANTS/build/bin/')
# Where MCC binaries are stored
MCC_BINARY_PATH = os.path.join(NFS_ROOT, 'projects/stroke/bin/MCC/')
# Matlab compiler runtime location
MCR =             os.path.join(NFS_ROOT, 'shared_software/MCR/v717/')
# Script to generate a file for submitting with QSUB
QSUB_RUN =        os.path.join(THIS_DIR,'qsub-run')

# Requires SGE to run in batch mode
QSUB = 'qsub'
# image type / file extension: only .nii.gz is currently supported!!
STANDARD_EXTENSION = '.nii.gz'


################################################################################
### Utility I/O stuff (handles all file naming conventions: adjust to taste) ###
################################################################################

def get_filebase(filename):
    """
    Returns a file's base name without an extension:
    e.g., takes '/foo/bar/hello.nii.gz' and returns 'hello'
    """
    base_ext = filename.split(os.path.sep)[-1]
    return base_ext.split('.')[0]

def select_non_atlas(moving, fixed):
    """
    Returns whichever of moving/fixed isn't the atlas image.
    """
    # TODO this is kind of a hack: get rid of this and do things better
    if 'atlases' not in fixed and \
            'atlases' in moving:
        non_atlas = fixed # atlas -> subj
    elif 'atlases' not in moving and \
            'atlases' in fixed:
        non_atlas = moving # subj -> atlas
    elif 'atlases' not in fixed and \
            'atlases' not in moving:
        assert os.path.dirname(fixed) == os.path.dirname(moving), \
                "Found registration between two non-atlas images from"\
                "different folders"
        non_atlas = moving # subject -> subject
    else:
        raise ValueError("Shouldn't have more than one atlas image in a registration")
    return os.path.dirname(non_atlas)

class Dataset(object):
    """
    Class representing your data. Has a range of features from simple to
    elaborate; you can use as much as is useful to you.

    The parameter `feature` refers to a particular image: in our dataset,
    it was usually either "image" or "roi" (region of interest, referring
    to a manual or automatically segmented region).
    """
    def __init__(self, base, atlas):
        self.base = base
        self.atlas = atlas

    def get_sge_folder(self, subj):
        """ Returns the subfolder of the subject with SGE logs """
        return os.path.join(self.base, subj, 'sge_new')

    def get_file(self, subj, modality, feature, modifiers=''):
        spec = {'subj': subj, 'modality': modality, 'feature': feature, 'modifiers': modifiers}

        template = os.path.join(self.base, '{subj}/images/{subj}_{modality}_{feature}{modifiers}.nii.gz')

        return template.format(**spec)

    def get_original_file(self, subj, modality, feature):
        if feature == 'img':
            feature = 'raw'
        template = os.path.join(self.base , '{subj}/original/{modality}_1/{subj}_{modality}_{feature}.nii.gz')
        spec = {'subj': subj, 'modality': modality, 'feature': feature}
        # ORIGINAL_TEMPLATE = os.path.join(self.base , '/%(subj)s/%(modality)s.nii.gz')
        # spec = {'subj': subj, 'modality': modality}

        return template.format(**spec)

class Atlas(object):
    """
    Class representing an atlas/template image. Provides the ability to
    easily track and get files in atlas space.
    """

    suffixes = {'img': '.nii.gz'}

    def __init__(self, atlas_subj, atlas_location):
        self.atlas_subj = atlas_subj
        self.atlas_location = atlas_location

    def add_file(self, name, file_suffix):
        """ Registers a new file for use with get_file """
        self.suffixes[name] = file_suffix

    def get_file(self, name):
        """ Returns the filename for some registered file. """
        return os.path.join(self.atlas_location,
                            '{atlas_subj}{suffix}'.format(
                                suffix=self.suffixes[name], atlas_subj=self.atlas_subj))

def modify_filename(filename, modifier):
    """ Takes ('/a/b/foo.nii.gz', '_pad') and returns '/a/b/foo_pad.nii.gz' """
    assert filename.endswith(STANDARD_EXTENSION)
    return os.path.join(os.path.dirname(filename), get_filebase(filename) + modifier + STANDARD_EXTENSION)

###############################################################################
### Commands (perform tasks, sort of like interfaces in nipype but simpler) ###
###############################################################################
class Command(object):
    """
    Represents a command/task that has to be run.  When creating one, make sure
    to call Command.__init__ with comment and output!

    If your command produces any outputs not given by the output keyword
    argument, you should set outfiles within your command.
    """
    all_commands = [] # Static list of all command objects

    @classmethod
    def reset(cls):
        cls.all_commands = []
    @classmethod
    def generate_code(cls, command_file, clobber_existing_outputs=False,
                      json_file=None):
        """
        Writes code to perform all created commands. Commands are run in the
        order they were created; there are no dependency-based reorderings.

        command_file : a file to write the commands to
        clobber_existing_outputs: whether or not to rerun commands whose
        outputs already exist
        """
        # TODO incorporate dependencies
        with open(command_file, 'w') as f:
            f.write('#!/usr/bin/env bash\n')
            f.write('set -e\n\n')
            for command in cls.all_commands: # order is very important here
                if clobber_existing_outputs or not command.check_outputs():
                    f.write('# ' + command.comment + '\n')
                    f.write(command.cmd + '\n\n\n\n')
                else:
                    f.write('# *** Skipping ' + command.comment + '\n\n\n\n')
        os.chmod(command_file, 0775)



    def __init__(self, comment='', **kwargs):
        if not hasattr(self, 'outfiles'):
            self.outfiles = [kwargs['output']]
        self.comment = comment

        self.cmd = self.cmd % kwargs

        self.command_id = len(Command.all_commands)
        Command.all_commands.append(self) # order is very important here

        self.parameters = kwargs

        # Tracking of input/output relationships between commands.
        # TODO use this to track files, etc.
        inputs = []
        self.inputs = []
        # TODO make this much better
        for (k, v) in self.parameters.items():
            if k == 'cmdName':
                continue
            if type(v) is str:
                # split on non-escaped spaces (since escaped spaces could be
                # filenames)
                inputs.extend(re.split(r'(?<!\\)\s+', v))
            elif type(v) is list or type(v) is tuple:
                inputs.extend(v)
        for v in inputs:
            if type(v) is str and has_valid_path(v):
                self.inputs.append(v)

        # print(self.inputs)
        # print('\n\n')
        self.inputs = set(self.inputs).difference(self.outfiles)

    def check_outputs(self):
        """ Checks if outputs are already there (True if they are). Warning: not thread-safe """
        return len(filter(os.path.exists, self.outfiles)) == len(self.outfiles)

def is_original_file(filename):
    # TODO fix this awful hack
    return ('original' in filename or 'atlas' in filename or 'CALL' in filename) and \
            not (filename.endswith('.sh') or filename.endswith('.py'))

def has_valid_path(filename):
    return os.path.isdir(os.path.dirname(filename))
###################################################
# Commands for using binaries that ship with ANTS #
###################################################
class ANTSCommand(Command):
    def __init__(self, comment='', **kwargs):
        """
        Creates an ANTS registration command.

        Parameters:
        -----------
        comment : a short string describing what your command does

        Keyword arguments:
        ------------------
        fixed : the fixed/target image for registration
        moving : the moving image for registration
        metric : An ANTS similarity metric. Tested/supported options are
                 'CC' (corr. coeff.) and 'MI' (mutual information).
                 Other ANTS metrics might work but are untested.
        regularization : an ANTS regularization type. The tested/supported
                         option is 'Gauss'
        mask : a mask (binary volume file) over which to compute the metric
               for registration
        method : 'affine', 'rigid', or otherwise the number of iterations
                 (e.g. 200x200x200) for non-linear.
        cont : the name of an affine registration file to initialize
               and continue from
        init : the name of an affine registration file to initialize
               from (does NOT perform further affine steps)
        radiusBins : the correlation radius (for CC) or # of bins (for MI)
        other : a string with extra flags for ANTS that aren't listed/handled above.

        """
        self.cmd = ANTSPATH + \
            'ANTS 3 ' + \
            '-m %(metric)s[%(fixed)s,%(moving)s,1,%(radiusBins)d] ' + \
            '-t Syn[0.25] ' + \
            '-r %(regularization)s ' + \
            '-o %(output)s ' + \
            '--number-of-affine-iterations 10000x10000x10000x10000x10000'
        outfiles = ['Affine.txt']
        if 'method' in kwargs:
            if kwargs['method'] in ['affine', 'rigid']:
                self.cmd += ' -i 0'
                if kwargs['method'] == 'rigid':
                    self.cmd += ' --rigid-affine true'
                    self.method_name = 'RIGID'
                else:
                    self.method_name = 'AFFINE'
            else:
                outfiles += ['InverseWarp' + STANDARD_EXTENSION, 'Warp' + STANDARD_EXTENSION]
                self.cmd += ' -i %(method)s';
                # Convert to a form that's appropriate for string naming:
                # "Gauss[4.5,0] becomes "GAUSS_45_0"
                sanitized_regularization = kwargs['regularization'].replace('[','_')\
                    .replace(']','_').replace(',','_').replace('.','').upper()
                self.method_name = 'NONLINEAR_{}_{}'.format(sanitized_regularization, kwargs['method'])
        else:
            raise ValueError("You need to specify a method! (affine, rigid, or nonlinear)")
        if 'mask' in kwargs and kwargs['mask']:
            self.cmd += ' -x %(mask)s'
            mask_string = 'MASKED'
        else:
            mask_string = ''

        if 'cont' in kwargs:
            self.cmd += ' --initial-affine %(cont)s --continue-affine 1'
        if 'init' in kwargs:
            self.cmd += ' --initial-affine %(init)s --continue-affine 0'
        assert not ('cont' in kwargs and 'init' in kwargs), "Should only specify one of init/cont for affine"
        if 'other' in kwargs:
            self.cmd += '%(other)s'

        if 'radiusBins' not in kwargs:
            if kwargs['metric'] == 'CC':
                kwargs['radiusBins'] = 5
            elif kwargs['metric'] == 'MI':
                kwargs['radiusBins'] = 32

        # Set up output name, etc
        base_folder = select_non_atlas(kwargs['fixed'], kwargs['moving'])
        outpath = os.path.join(os.path.dirname(base_folder),'images', 'reg')

        self.transform_infix = '%s_%s%d_%s' % \
                (self.method_name, kwargs['metric'], kwargs['radiusBins'], mask_string)

        # file prefix for ANTS: contains all information about the reg
        # TODO more sophisticated tracking of parameters should happen here
        a = '{moving}_TO_{descr}_{fixed}'.format(
                moving=get_filebase(kwargs['moving']),
                fixed=get_filebase(kwargs['fixed']),
                descr=self.transform_infix)

        kwargs['output'] = self.transform_string = os.path.join(outpath, a)
        Command.__init__(self, comment, **kwargs)

        # outfiles could be EITHER aff+warp+invwarp or aff
        self.outfiles = [self.transform_string + name for name in outfiles]

        # Figure out the strings for warping using this registration's output
        if len(outfiles) == 1:
            assert kwargs['method'] in ('affine', 'rigid')
            (self.affine,) = self.outfiles
            self.forward_warp_string = ' {0} '.format(self.affine)
            self.backward_warp_string = ' -i {0} '.format(self.affine)
            #self.forward_warp_string = ' ' + self.affine + ' '
            #self.backward_warp_string = ' -i ' + self.affine + ' '
        elif len(outfiles) == 3:
            assert kwargs['method'] not in ('affine', 'rigid')
            # Nonlinear registration: we have Warp, InverseWarp, and Affine
            (self.affine, self.inverse_warp, self.warp) = self.outfiles
            self.forward_warp_string = ' {0} {1} '.format(self.warp, self.affine)
            self.backward_warp_string = ' -i {0} {1} '.format(self.affine, self.inverse_warp)
            #self.forward_warp_string = ' ' + self.warp + ' ' + self.affine + ' '
            #self.backward_warp_string = ' -i ' + self.affine + ' ' + self.inverse_warp + ' '
        else:
            raise ValueError("I expected 1 or 3 outputs from ANTS, not %d" % len(outfiles))

def make_warp_kwargs(moving, reference, reg_sequence, invert_sequence):
    """
    Creates a dictionary of arguments for a warp command using the *ordered*
    sequence of registration command objects provided.

    invert_sequence is a list of strings corresponding to the warps in
    reg_sequence: each one should be 'inverse' or 'forward'

    Sample usage:
    -------------
    regA = ANTSCommand("Register 1 to 2", moving=img1, fixed=img2, ...)
    regB = ANTSCommand("Register 2 to 3", moving=img2, fixed=img3, ...)

    # To warp 1 to 3 using these registrations:
    args = make_warp_kwargs(img1, img3, [regA, regB], ['forward', 'forward'])
    warp_1to3 = WarpCommand("Warp 1 to 3", **args)

    # To warp 3 to 1 using these registrations
    args = make_warp_kwargs(img3, img1, [regB, regA], ['inverse', 'inverse'])
    warp_3to1 = WarpCommand("Warp 3 to 1", **args)
    """
    transform_infix = ''
    command_sequence = ''
    for (ants, invert) in zip(reg_sequence, invert_sequence):
        transform_infix += ants.transform_infix

        if invert == 'inverse':
            warp_string = ants.backward_warp_string
        elif invert == 'forward':
            warp_string = ants.forward_warp_string
        command_sequence = warp_string + command_sequence
    base_folder = select_non_atlas(moving, reference)
    a = '{moving}_IN_{descr}_{reference}-.nii.gz'.format(
            moving=get_filebase(moving),
            descr=transform_infix,
            reference=get_filebase(reference))
            #'%(moving)s_IN_%(descr)s_%(reference)s-.nii.gz' % {'moving': get_filebase(moving),
    output = os.path.join(base_folder,a)
    return {'moving': moving, 'reference': reference,
            'output': output, 'transforms': command_sequence}

class WarpCommand(Command):
    def __init__(self, comment='', **kwargs):
        """
        Creates a warping command for ANTS warps.

        Parameters:
        -----------
        comment : a short string describing what your command does

        Keyword arguments:
        ------------------
        moving, output, reference (image filenames)
        transforms : string of transforms as would be passed to WIMT,
        useNN : boolean indicating whether to use nearest-neighbor interp
        """
        self.cmd = ANTSPATH + 'WarpImageMultiTransform 3 %(moving)s %(output)s -R %(reference)s'
        self.cmd += ' ' + kwargs['transforms']

        if 'useNN' in kwargs and kwargs['useNN']:
            self.cmd += ' --use-NN'

        Command.__init__(self, comment, **kwargs)

class N4Command(Command):
    def __init__(self, comment='', **kwargs):
        """
        Creates a command for N4 bias field correction.

        Parameters:
        -----------
        comment : a short string describing what your command does

        Keyword arguments:
        ------------------
        input, output: image filenames
        """
        self.cmd = os.path.join(ANTSPATH, 'N4BiasFieldCorrection') + \
                ' --image-dimension 3 --input-image %(input)s --output %(output)s'
        Command.__init__(self, comment, **kwargs)

###############################
# Commands for simple scripts #
###############################
class InputOutputShellCommand(Command):
    def __init__(self, comment='', **kwargs):
        """
        Runs a simple command of the form <cmdName> <input> <output>

        Parameters:
        -----------
        comment : a short string describing what your command does

        Keyword arguments:
        ------------------
        input, output : filenames
        cmdName : name of command/script (must be on PATH or fully qualified)

        Example:
        converter = InputOutputShellCommand('/path/to/freesurfer/bin/mri_convert',
                                            input='/path/to/data/img1.mgz',
                                            output='/path/to/data/img2.nii.gz')
        """
        self.cmd = "%(cmdName)s %(input)s %(output)s %(other)s"
        if 'other' not in kwargs:
            kwargs['other'] = ''
        Command.__init__(self, comment, **kwargs)

class PyMatchWMCommand(Command):
    def __init__(self, comment='', **kwargs):
        """
        Command to match white matter intensity between two images.
        See matchWM.py documentation for more details.

        Keyword arguments:
        ------------------
        alignedInFile : input image in atlas space
        maskFile : atlas space mask
        inFile : input image in original space
        wmiSrc : atlas file or a numeric string w/the average white matter intensity
        outFile: output
        """
        py = os.path.join(THIS_DIR, 'matchWM.py')
        self.cmd = py + ' %(alignedInFile)s %(maskFile)s %(inFile)s %(wmiSrc)s %(output)s'
        Command.__init__(self, comment, **kwargs)

class PyPadCommand(Command):
    def __init__(self, comment='', **kwargs):
        """
        Command to pad a nifti volume
        See matchWM.py documentation for more details.

        Keyword arguments:
        ------------------
        input, output: input and padded output volumes
        out_mask: output mask (binary image) marking which regions weren't padding
        """
        py = os.path.join(THIS_DIR, 'padNii.py')
        self.cmd = py + ' %(input)s %(output)s %(out_mask)s'
        self.outfiles = [kwargs['output'], kwargs['out_mask']]
        Command.__init__(self, comment, **kwargs)

############################################
# Commands for using MCC-compiled binaries #
############################################
class MCCCommand(Command): # abstract class
    prefix = os.path.join(MCC_BINARY_PATH, 'MCC_%(matlabName)s/run_%(matlabName)s.sh ') + MCR + ' '
    def __init__(self, comment='', **kwargs):
        """ Arguments: matlabName, ... """
        self.cmd = self.prefix
        raise NotImplementedError # Abstract class

class MCCInputOutputCommand(MCCCommand):
    def __init__(self, comment='', **kwargs):
        """ See MCCCommand. Arguments: matlabName, input, output """
        self.cmd = (self.prefix + '%(input)s %(output)s')
        Command.__init__(self, comment, **kwargs)

class MCCPadCommand(MCCCommand):
    def __init__(self, comment='', **kwargs):
        """ See MCCCommand. Arguments: input, output, out_mask """
        kwargs['matlabName'] = 'padNii'
        self.cmd = (self.prefix + '%(input)s %(output)s %(out_mask)s')
        Command.__init__(self, comment, **kwargs)

class MCCHistEqCommand(MCCCommand):
    def __init__(self, comment='', **kwargs):
        """ See MCCCommand. Arguments: input, output, ref_hist. does 'cut' only """
        kwargs['matlabName'] = 'doHistogramEqualization'
        self.cmd = (self.prefix + '%(ref_hist)s %(output)s %(input)s cut')
        Command.__init__(self, comment, **kwargs)

class MCCMatchWMCommand(MCCCommand):
    def __init__(self, comment='', **kwargs):
        kwargs['matlabName'] = 'matchWM'
        self.cmd = self.prefix + '%(inFile)s %(maskFile)s %(intensity)s %(output)s'
        Command.__init__(self, comment, **kwargs)


