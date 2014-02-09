"""
This module provides a general set of tools for implementing medical imaging
pipelines. In particular, it was used to build the registration pipeline
described in the paper:

Quantification and Analysis of Large Multimodal Clinical Image Studies:
Application to Stroke, by Sridharan, Dalca et al.

For questions, please contact {rameshvs,adalca}@csail.mit.edu.
"""

from __future__ import print_function
import re
import os
import warnings

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

##############################################
### System paths (you should change these) ###
##############################################

# On our system, everything lives on an NFS drive.
NFS_ROOT = '/data/vision/polina'
# ANTS build directory
ANTSPATH =        os.path.join(NFS_ROOT, 'shared_software/ANTS/build/bin/')
# Where MCC binaries are stored
MCC_BINARY_PATH = os.path.join(NFS_ROOT, 'projects/stroke/bin/MCC/')
# Matlab compiler runtime location
MCR =             os.path.join(NFS_ROOT, 'shared_software/MCR/v717/')
# Script to generate a file for submitting with SGE QSUB
QSUB_RUN =        os.path.join(THIS_DIR,'qsub-run')

# Requires SGE to run in batch mode
QSUB = 'qsub'
# image type / file extension: only .nii.gz is currently supported!!
# TODO support non-image outputs
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
        # while this is a reasonable thing to want to do, we just don't
        # support it yet
        assert os.path.dirname(fixed) == os.path.dirname(moving), \
                "Found registration between two non-atlas images from"\
                "different folders"
        non_atlas = moving # subject -> subject
    else:
        raise ValueError("Shouldn't have more than one atlas image in a registration")
    return os.path.dirname(non_atlas)

class DatasetFile(object):
    def __init__(self, subj, modality, feature, modifiers, extension, template):

        self.subj = subj
        self.modality = modality
        self.feature = feature
        self.modifiers = modifiers
        self.template = template
        self.extension = extension
        self.pieces = (subj, modality, feature, modifiers, extension, template)

        spec = {'subj': self.subj, 'modality': self.modality,
                'feature': self.feature, 'extension': extension}
        if '{modifiers}' in self.template:
            spec['modifiers'] = self.modifiers

        self.filename = self.template.format(**spec) + extension

    def __eq__(self, other):
        return type(self) == type(other) and self.pieces == other.pieces

    def __hash__(self):
        return hash(self.pieces)

    def get_filebase(self):
        """
        Returns a file's base name without an extension:
        e.g., takes '/foo/bar/hello.nii.gz' and returns 'hello'
        """
        base_ext = self.filename.split(os.path.sep)[-1]
        return base_ext.split('.')[0]


class DatasetMissingFileError(Exception): pass

class Dataset(object):
    """
    Class representing your data. Has a range of features from simple to
    elaborate; you can use as much as is useful to you.

    The parameter `feature` refers to a particular image: in our dataset,
    it was usually either "image" or "roi" (region of interest, referring
    to a manual or automatically segmented region).
    """
    # TODO make an example here with the freesurfer format
    def __init__(self, base, atlas, original_template, processing_template):
        self.base = base
        self.atlas = atlas
        self.original_template = original_template
        self.processing_template = processing_template

        self.mandatory_files = set()
        self.optional_files = {}
        self.optional_inputs = set()
        self.invalid_files = set()

    def get_folder(self, subj):
        return os.path.join(self.base, subj)

    def get_log_folder(self, subj):
        """
        Returns the subfolder of the subject with logs (SGE, metadata, etc)
        """
        return os.path.join(self.base, subj, 'logs')

    def get_original_file(self, subj, modality, feature, extension=STANDARD_EXTENSION):
        """
        Returns the specified file. Meant to be used for files that are inputs to the pipeline.
        """
        if feature == 'img':
            feature = 'raw'
        f = DatasetFile(subj, modality, feature, None, extension, self.original_template)

        ### Tracking optional/mandatory files
        if not os.path.exists(f.filename):
            if self.is_mandatory(f):
                raise DatasetMissingFileError("Missing mandatory file:" + f.filename)
            else:
                self.invalidate_file(f)
        if not self.is_mandatory(f):
            self.add_optional_original(f)

        return f

    def get_file(self, subj, modality, feature, modifiers='', extension=STANDARD_EXTENSION):
        """
        Returns the specified file. Meant to be used for intermediate files in the processing
        pipeline.
        """
        f = DatasetFile(subj, modality, feature, modifiers, extension, self.processing_template)
        return f

    ### Mandatory-ness is shared across all subjects, so we track tuples
    ### rather than DatasetFile objects.
    def is_mandatory(self, f):
        return (f.modality, f.feature, f.extension) in self.mandatory_files

    def add_mandatory_input_file(self, modality, feature, extension=STANDARD_EXTENSION):
        """ Registers a certain file as mandatory """
        self.mandatory_files.add((modality, feature, extension))

    def add_optional_original(self, f):
        self.optional_inputs.add((f.modality, f.feature, f.extension))

    ### Invalid-ness is done on a per-subject basis, so track DatasetFile objects
    def is_invalid(self, f):
        """ Checks if a DatasetFile is invalid: future commands using it won't run. """
        return f in self.invalid_files
    def invalidate_file(self, f):
        """ Invalidates a DatasetFile: future commands using it won't run. """
        self.invalid_files.add(f)

    def invalidate_command_outputs(self, command):
        """
        Invalidates all of a Command's outputs (DatasetFiles) so that future
        commands that depend on them won't run.
        """
        for outp in command.outfiles:
            self.invalidate_file(outp)

    def check_command_input_validity(self, command):
        """ Checks if any inputs to command have been invalidated """
        for inp in command.inputs:
            if type(inp) is DatasetFile and self.is_invalid(inp):
                return False
        return True

class Atlas(object):
    """
    Class representing an atlas/template image. Provides the ability to
    easily track and get files in atlas space.
    """

    suffixes = {'img': STANDARD_EXTENSION}

    def __init__(self, atlas_subj, atlas_location):
        self.atlas_subj = atlas_subj
        self.atlas_location = atlas_location

    def add_file(self, name, file_suffix):
        """ Registers a new file for use with get_file """
        self.suffixes[name] = file_suffix

    def get_file(self, name):
        """ Returns the filename for some registered file. """
        return AtlasFile(os.path.join(self.atlas_location,
                            '{atlas_subj}{suffix}'.format(
                                suffix=self.suffixes[name], atlas_subj=self.atlas_subj)))

class AtlasFile(DatasetFile):
    def __init__(self, filename):
        self.filename = filename
    def __hash__(self):
        return hash(self.filename)
    def __eq__(self, other):
        return type(other) == type(self) and self.filename == other.filename



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
    def generate_code(cls, command_file, dataset=None, clobber_existing_outputs=False,
                      json_file=None):
        """
        Writes code to perform all created commands. Commands are run in the
        order they were created; there are no dependency-based reorderings.

        command_file : a file to write the commands to
        dataset : a dataset object to cross-check commands against
        clobber_existing_outputs : whether or not to rerun commands whose
        outputs already exist
        """
        # TODO incorporate dependencies
        with open(command_file, 'w') as f:
            f.write('#!/usr/bin/env bash\n')
            f.write('set -e\n\n')
            for command in cls.all_commands: # loop in order listed
                # TODO cleaner logic
                if clobber_existing_outputs or not command.check_outputs():
                    if dataset is not None and not dataset.check_command_input_validity(command):
                        f.write('# *** Skipping (due to missing input) ' + command.comment + '\n'*4)
                        dataset.invalidate_command_outputs(command)

                    else:
                        f.write('# ' + command.comment + '\n')
                        f.write(command.cmd + '\n'*4)
                else:
                    f.write('# *** Skipping ' + command.comment + '\n'*4)
        os.chmod(command_file, 0775)



    def __init__(self, comment, **kwargs):
        if not hasattr(self, 'outfiles'):
            if 'output' not in kwargs:
                self.outfiles = []
                warnings.warn("output/outfiles not specified for command; can't track output:\n"+comment,
                        RuntimeWarning)
            else:
                self.outfiles = [kwargs['output']]

        self.parameters = kwargs
        self.comment = comment

        good_kwargs = {}
        for (k, v) in kwargs.iteritems():
            if hasattr(v, 'filename'):
                good_kwargs[k] = v.filename
            else:
                good_kwargs[k] = v
        self.cmd = self.cmd % good_kwargs

        self.command_id = len(Command.all_commands)
        Command.all_commands.append(self) # order is very important here


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
            elif hasattr(v, 'filename'):
                inputs.append(v)
            elif type(v) is list or type(v) is tuple:
                inputs.extend(v)
        for v in inputs:
            if (type(v) is str and has_valid_path(v)) or \
                    hasattr(v, 'filename'):
                self.inputs.append(v)

        self.inputs = set(self.inputs).difference(self.outfiles)

    def check_outputs(self):
        """ Checks if outputs are already there (True if they are). Warning: not thread-safe """
        # TODO move to dataset
        new_outfiles = []
        for f in self.outfiles:
            if hasattr(f, 'filename'):
                new_outfiles.append(f.filename)
            else:
                new_outfiles.append(f)

        return len(self.outfiles) > 0 and \
                all([os.path.exists(f) for f in new_outfiles])

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
    def __init__(self, comment, **kwargs):
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
        output_folder : folder to store output with default naming
        output_prefix : exact prefix for output files (as with ANTS's -o option)
        (specify either output_folder OR output_prefix)
        other : a string with extra flags for ANTS that aren't listed/handled above.

        """
        self.cmd = ANTSPATH + \
            'ANTS 3 ' + \
            '-m %(metric)s[%(fixed)s,%(moving)s,1,%(radiusBins)d] ' + \
            '-t Syn[0.25] ' + \
            '-o %(output)s ' + \
            '--number-of-affine-iterations 100x100x100'
            #'--number-of-affine-iterations 10000x10000x10000x10000x10000'
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
                self.cmd += ' -i %(method)s -r %(regularization)s '
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
        if 'output_folder' in kwargs:
            base_folder = kwargs.pop('output_folder')
            outpath = os.path.join(base_folder, 'images', 'reg')

            self.transform_infix = '%s_%s%d_%s' % \
                    (self.method_name, kwargs['metric'], kwargs['radiusBins'], mask_string)

            # file prefix for ANTS: contains all information about the reg
            # TODO more sophisticated tracking of parameters should happen here
            a = '{moving}_TO_{descr}_{fixed}'.format(
                    moving=kwargs['moving'].get_filebase(),
                    fixed=kwargs['fixed'].get_filebase(),
                    descr=self.transform_infix)

            self.output_prefix = os.path.join(outpath, a)
        elif 'output_prefix' in kwargs:
            self.output_prefix = kwargs.pop('output_prefix')
        kwargs['output'] = self.output_prefix
        Command.__init__(self, comment, **kwargs)

        # outfiles could be EITHER aff+warp+invwarp or aff
        self.outfiles = [self.output_prefix + name for name in outfiles]

        # Figure out the strings for warping using this registration's output
        if len(outfiles) == 1:
            assert kwargs['method'] in ('affine', 'rigid')
            (self.affine,) = self.outfiles
            self.forward_warp_string = ' {0} '.format(self.affine)
            self.backward_warp_string = ' -i {0} '.format(self.affine)
        elif len(outfiles) == 3:
            assert kwargs['method'] not in ('affine', 'rigid')
            # Nonlinear registration: we have Warp, InverseWarp, and Affine
            (self.affine, self.inverse_warp, self.warp) = self.outfiles
            self.forward_warp_string = ' {0} {1} '.format(self.warp, self.affine)
            self.backward_warp_string = ' -i {0} {1} '.format(self.affine, self.inverse_warp)
        else:
            raise ValueError("I expected 1 or 3 outputs from ANTS, not %d" % len(outfiles))


def get_warped_filename(moving, reference, reg_sequence, inversion_sequence):
    """
    Utility function for combining ANTS registrations. See
    `ANTSWarpCommand.make_from_registration_sequence' for more details.
    """
    ## Combine registrations
    transform_infix = ''
    command_sequence = ''
    # construct warps in the format that ANTS wants them
    for (ants, invert) in zip(reg_sequence, inversion_sequence):
        transform_infix += ants.transform_infix

        if invert == 'inverse':
            warp_string = ants.backward_warp_string
        elif invert == 'forward':
            warp_string = ants.forward_warp_string
        command_sequence = warp_string + command_sequence
    base_folder = select_non_atlas(moving.filename, reference.filename)
    output_file = '{moving}_IN_{descr}_{reference}-.nii.gz'.format(
            moving=moving.get_filebase(),
            descr=transform_infix,
            reference=reference.get_filebase())
    output_path = os.path.join(base_folder, output_file)
    return (output_path, command_sequence)

class ANTSWarpCommand(Command):
    """
    Command representing an ANTS warp. Unlike most other commands, the most
    convenient way to make these is using the class methods
    make_from_single_registration and make_from_registration_sequence.
    """

    # Maps (moving, reference) filename pairs to warped image filenames.
    # Warning: not reliable for pairs which have multiple warp paths!
    warp_mapping = {}
    @classmethod
    def make_from_single_registration(cls, comment, moving, reference,
            registration, inversion='forward'):
        """
        Creates a warp command from a single registration. See
        `make_from_registration_sequence' for more details.

        moving and reference are filenames; registration is an ANTSCommand
        object, and inversion is a string ('forward' or 'inverse', default
        'forward') describing which way to warp.
        """
        return cls.make_from_registration_sequence(comment, moving, reference,
                [registration], [inversion])

    @classmethod
    def make_from_registration_sequence(cls, comment, moving, reference,
            reg_sequence, inversion_sequence, **kwargs):
        """
        Creates a warp command using the *ordered* sequence of registration
        command objects provided.

        inversion_sequence is a list of strings corresponding to the
        registration command objects in reg_sequence: each one should be
        'inverse' or 'forward'

        kwargs contains extra arguments to the constructor (such as useNN)

        See make_from_single_registration for a simpler interface when
        multiple warps aren't needed.

        Sample usage:
        -------------
        regA = ANTSCommand("Register 1 to 2", moving=img1, fixed=img2, ...)
        regB = ANTSCommand("Register 2 to 3", moving=img2, fixed=img3, ...)

        # To warp 1 to 3 using these registrations:
        warp_1to3 = ANTSWarpCommand.make_from_registration_sequence(
                        img1, img3, [regA, regB], ['forward', 'forward'])

        # To warp 3 to 1 using these registrations
        warp_3to1 = ANTSWarpCommand.make_from_registration(
                        img3, img1, [regB, regA], ['inverse', 'inverse'])
        """
        (output_path, cmd_sequence) = get_warped_filename(
                moving, reference, reg_sequence, inversion_sequence)

        return cls(comment, moving=moving, reference=reference,
                output=output_path, transforms=cmd_sequence,
                **kwargs)

    def __init__(self, comment, **kwargs):
        """
        Creates a warping command for ANTS warps. For a more convenient
        interface, see make_from_registration.

        Parameters:
        -----------
        comment : a short string describing what your command does

        Keyword arguments:
        ------------------
        moving, output, reference (image filenames)
        transforms : string of transforms with spaces just as you would pass
                     to WarpImageMultiTransform
        useNN : boolean indicating whether to use nearest-neighbor interp
        """
        self.warp_mapping[(kwargs['moving'], kwargs['reference'])] = kwargs['output']
        self.cmd = ANTSPATH + 'WarpImageMultiTransform 3 %(moving)s %(output)s -R %(reference)s'
        self.cmd += ' ' + kwargs['transforms']

        if 'useNN' in kwargs and kwargs['useNN']:
            self.cmd += ' --use-NN'

        Command.__init__(self, comment, **kwargs)

class N4Command(Command):
    def __init__(self, comment, **kwargs):
        """
        Creates a command for N4 bias field correction. Assumes 3D images.

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
    def __init__(self, comment, **kwargs):
        """
        Runs a simple command of the form <cmdName> <input> <output> <extra_args>

        Parameters:
        -----------
        comment : a short string describing what your command does

        Keyword arguments:
        ------------------
        input, output : filenames
        cmdName : name of command/script (must be on PATH or fully qualified)
        extra_args: additional command line arguments (*not* outputs)

        Example:
        converter = InputOutputShellCommand('/path/to/freesurfer/bin/mri_convert',
                                            input='/path/to/data/img1.mgz',
                                            output='/path/to/data/img2.nii.gz')
        """
        self.cmd = "%(cmdName)s %(input)s %(output)s %(extra_args)s"
        if 'extra_args' not in kwargs:
            kwargs['extra_args'] = ''
        Command.__init__(self, comment, **kwargs)

class DumpCommand(Command):
    def __init__(self, comment, **kwargs):
        pass
class PyFunctionCommand(Command):
    def __init__(self, comment, function, args, output_positions=[]):
        """
        Command to run a python function. Function must passed as
        a string of the form 'module.function', and the module must
        be on the current path.

        args is a list of strings with the arguments to the function (right
        now only strings are supported). Strings with quotes of any kind in them
        may not work properly (untested).

        output_positions is an optional list of indices into args that
        say which ones are output files.

        All of the function's parameters and all of the arguments must be
        strings (this limitation will hopefully be removed in a future
        version).
        """
        import sys
        (module, funcname) = function.rsplit('.', 1)
        # TODO trim this to where the module is
        path = sys.path + [os.getcwd()]
        self.cmd =\
            "python -c \"import sys; sys.path.extend(%(path)s);" \
            "import %(module)s; %(module)s.%(func)s(%(arg)s)\""

        self.outfiles = [args[i] for i in output_positions]
        newargs = []
        for arg in args:
            if type(arg) in [DatasetFile, AtlasFile]:
                newargs.append(arg.filename)
            else:
                newargs.append(str(arg))
        Command.__init__(self, comment, path=path, module=module, func=funcname,
                arg=','.join(["'"+arg+"'" for arg in newargs]))



class PyMatchWMCommand(Command):
    def __init__(self, comment, **kwargs):
        """
        Command to match mode of white matter intensity to a reference value.
        See matchWM.py documentation for more details.

        Keyword arguments:
        ------------------
        inFile : input image
        maskFile : mask (in same coordinates as inFile)
        intensity : numeric string w/the average white matter intensity
        outFile: output
        """
        py = os.path.join(THIS_DIR, 'matchWM.py')
        self.cmd = py + '%(inFile)s %(maskFile)s %(intensity)s %(output)s'
        Command.__init__(self, comment, **kwargs)

class PyPadCommand(Command):
    def __init__(self, comment, **kwargs):
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
    def __init__(self, comment, **kwargs):
        """ Arguments: matlabName, ... """
        self.cmd = self.prefix
        raise NotImplementedError # Abstract class

class MCCInputOutputCommand(MCCCommand):
    def __init__(self, comment, **kwargs):
        """ See MCCCommand. Arguments: matlabName, input, output """
        self.cmd = (self.prefix + '%(input)s %(output)s')
        Command.__init__(self, comment, **kwargs)

class MCCPadCommand(MCCCommand):
    def __init__(self, comment, **kwargs):
        """ See MCCCommand. Arguments: input, output, out_mask """
        kwargs['matlabName'] = 'padNii'
        self.cmd = (self.prefix + '%(input)s %(output)s %(out_mask)s')
        Command.__init__(self, comment, **kwargs)

class MCCHistEqCommand(MCCCommand):
    def __init__(self, comment, **kwargs):
        """ See MCCCommand. Arguments: input, output, ref_hist. does 'cut' only """
        kwargs['matlabName'] = 'doHistogramEqualization'
        self.cmd = (self.prefix + '%(ref_hist)s %(output)s %(input)s cut')
        Command.__init__(self, comment, **kwargs)

class MCCMatchWMCommand(MCCCommand):
    def __init__(self, comment, **kwargs):
        kwargs['matlabName'] = 'matchWM'
        self.cmd = self.prefix + '%(inFile)s %(maskFile)s %(intensity)s %(output)s'
        Command.__init__(self, comment, **kwargs)

