# GPU Radio Image Transient Search

## Requirements

In order to use this software, in addition to the requirements in the setup.py file, you will need:

* To be running Linux
* An NVIDIA GPU with CUDA installed
* Bifrost https://github.com/lwa-project/bifrost

Install the software with ```pip install .``` 

## Basic Usage

* Make and go into an empty directory

The imdiffsearch.py script does all the heavy lifting and can take either .pims or .oims files as input. See https://lda10g.alliance.unm.edu/ for the LWA archive. 

* Run ```imdiffsearch.py /path/to/obs.oims``` to get plots and a numpy array of sources or ```imdiffsearch.py /path/to/obs.oims --noplot``` to suppress plotting.

The first time you run the script it will generate a ```config.ini``` file and then exit. Edit the file to your liking adjusting the field of view, number of sources allowed per image, the detection thresholds, etc and then re-run the script.

The plotting adds significant overheads and should be avoided for large runs, if possible. The default mode is to search stokes I, linear polarization, total polarization, and Circular polarization. Use ```--maskstokes 1 2 3``` to only search stokes I. 


## What it is and how it works

### This software is intended to search LWA all-sky images for transients. It leverages the GPU to search for transients in a massively parallel manner. The steps that the code performs is roughly as follows:

1. Read in the .oims or .pims file and create a numpy array of the images and a numpy array of the difference images
2. Calculate the positions of Jupiter and the Sun, just in case the user wants to automatically mask their positions via ```--maskateam``` (not the default).
3. Primary beam correct
4. Compute the RMS of the pre-calibrated difference images by sliding a box over the image in x and y coordinates and taking the smallest RMS per image. (This step may be eliminated in future versions.)
5. Very roughly calibrate the flux of the images and difference images using Cas A, Cyg A, Virgo A, and Tau A. This step averages over the entire timerange and averages over all the correction factors from each of the calibrator sources to hopefully mitigate the influence of scintillation. These models do not take into account the diffuse emission contribution to the flux.
6. Un-primary beam correct for flat noise across the sky
7. Re-compute RMS noise.

### Now the fun part, generating and labeling source islands:

6. All of the pixels in the difference image that are above the threshold specified in the ```config.ini``` file are noted in an array. For example, if the RMS noise is 50 Jy (good quality for a PASI image), and your ```config.ini``` file says ```nsigma = 5```, then all of the pixels in the difference image that are greater than 250 Jy will be marked. 
7. Each marked pixel is assigned a unique integer value. (Label)
8. Each column of adjacent pixels takes on one of these unique values. (Cluster columns)
9. Each row of adjacent pixels takes on one of these unique values. (Cluster rows)
10. Re-cluster columns
11. The image is rotated by 45 degrees in the image plane. 
12. Cluster the post-rotation columns
13. Cluster the post-rotation rows
14. Re-cluster the post-rotation Columns
15. Un-rotate back to the original orientation
16. Re-cluster columns
17. Re-cluster rows
18. Re-cluster columns
19. Re-rotate, re-cluster, and then un-rotate once more and the clustering is finished.

### Now that the islands are identified, grouped, and each island has a unique integer label, the script performs some quality control and flagging steps. 

20. Perform flagging according to ```config.ini``` settings, including dynamic range cuts and restrictions on the number of sources
21. Get the primary beam corrected flux of each source by taking the brightest pixel for peak flux and the sum of the flux for the island.
22. Optionally plot
23. Save sources into a numpy array for future investigations.

### A note on searching polarization images

There is a hope that by searching the polarization images (Stokes V) etc. that the confusion noise will be lower and that a transient search of such images will yield more transients. The default output of the LWA archival images produces stokes I,Q,U, and V images. The code then turns this into linear polarization, total polarization, and the absolute value of the stokes V, or circular, polarization. These polarization images have not be rigorously calibrated for leakage. We search the polarization images in the same way as the stokes I images: create a difference image and identify positive islands. 

