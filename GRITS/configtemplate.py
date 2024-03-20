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
; Initial, low nsigma. Keeping this low improves the qc performance, but takes longer
initialnsigma=2
; Final, higher nsigma. This nsigma will determine what candidates are returned.
finalnsigma=5
; Relative tolerance for position angle between sources. For determining if it's a plane.
rtol = 0.2
; Maximum separation from catalog for reporting
catsep = 5"""
