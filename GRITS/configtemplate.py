configfile = """[Quality Control]
; Max number of sources per image
maxsrcs = 99
; Max standard deviation of diff images
maxstd = 1e6
; Max rms of image
maxrms = 1e6
; Max absolute value of mean of the diff image
maxabsmean = 1e6
; Max dynamic range
DynamicRange = 1e16
; Minimum angular velocity of moving sources in degrees per second
minangvel = 1e-6
; Detection sigma 
nsigma=5
; Acceptable percentage of bright galactic plane to be detected
galpercent=0.10
; Relative tolerance for position angle between sources. For determining if it's a plane.
rtol = 0.2
; Maximum separation from catalog for reporting
catsep = 1.5 
; Maximum separation between sources detected at the same time in different stations
statsep = 1.5
; Minimum source width
minwid = 0
; Minimum fraction of channels
minchanfraction = 0
; Fraction of PB to include, must be less than 0.95
pbpercent = 0.80
[Batch Control]
; Number of Orville files to process at time. PASI files do this numberX4.
chunktime = 400
; Overlap in time between chunks. Helps catch moving objects.
overlaptime = 1"""
