# Medical Imaging Pipelines

This code provides a simple framework for building pipelines for medical
image processing tasks.

## Requirements

Using the basic pipeline-building code requires
[python](http://www.python.org/) with [numpy](http://www.numpy.org/).

Using the image registration tools currently available requires
the [ANTS](http://stnava.github.io/ANTs/) image registration toolkit.

Using MATLAB-compiled commands (MCC) requires the [MATLAB Compiler Runtime
(MCR)](http://www.mathworks.com/products/compiler/mcr/).

## Getting started

For an example, see `regpipe.py`, which implements the registration
pipeline described in the paper:

Quantification and Analysis of Large Multimodal Clinical Image Studies:
Application to Stroke by Sridharan et al.

To construct your own pipeline, you can use this script as a baseline
or build your own. Create `Command` objects from pipebuild.py in the
order you want them executed, just like you would with a shell script.

At the end, use `Command.generate_code()` to write a shell script to
a file. You can run this file either from within python using subprocess,
using a cluster with SGE as in `regpipe.py`, or using any other method
you prefer.

## File structure

File structure is dictated by the Dataset class in `pipebuild.py`. The code is
currently built around the assumption that there's a single atlas image, but
this will be generalized in a future update. Files are organized according to
the templates in the `get_file()` method of `Dataset`. The organization of your
original images (before being input to the pipeline) can be specified in the
`get_original_file()` method of `Dataset`.
