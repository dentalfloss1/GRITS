import numpy as np
from lsl.astro import MJD_OFFSET, DJD_OFFSET

CYG_A = (299.86815237, 40.7339159)
CAS_A = (350.85, 58.815)
SGR_A = (266.41681663, -29.00782497)
TAU_A = (83.6287,22.0147)
VIRGO_A = (187.70593077,12.39112325)
HERC_A = (252.783945, 4.992588)
floatinfo = np.finfo('f4')
floatmax = floatinfo.max
MJD_TO_DJD = MJD_OFFSET - DJD_OFFSET
