# GPU Radio Image Transient Search

## Requirements

In order to use this software, in addition to the requirements in the setup.py file, you will need:

* To be running Linux
* An NVIDIA GPU with CUDA installed
* Bifrost https://github.com/lwa-project/bifrost

Install the software with pip install . 

## Basic Usage

* Make and go into an empty directory

The imdiffsearch.py script does all the heavy lifting and can take either .pims or .oims files as input. See https://lda10g.alliance.unm.edu/ for the LWA archive. 

* Run ```imdiffsearch.py /path/to/obs.oims``` to get plots and a numpy array of sources or ```imdiffsearch.py /path/to/obs.oims --noplot``` to suppress plotting. 

The plotting adds significant overheads and should be avoided for large runs, if possible. The default mode is to search stokes I, linear polarization, total polarization, and Circular polarization. Use ```--maskstokes 1 2 3``` to only search stokes I. 


## What it is and how it works

This software is intended to search LWA all-sky images for transients. It leverages the GPU to search for transients in a massively parallel manner. The steps that the code performs is roughly as follows:

1. Read in the .oims or .pims file and create a numpy array of the images and a numpy array of the difference images
2. Calculate the positions of Jupiter and the Sun, just in case the user wants to automatically mask their positions via ```--maskateam``` (not the default).
3. Compute the pre-calibrated RMS of the pre-calibrated difference images by sliding a box over the image in x and y coordinates and taking the smallest RMS per image.
4. Very roughly calibrate the flux of the images and difference images using Cas A, Cyg A, and Virgo A
