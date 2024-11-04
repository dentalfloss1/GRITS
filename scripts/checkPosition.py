import glob
import numpy as np
from GRITS.imagetools import angsep
from GRITS.catalogs import cat3c, VLSSr50jy
fluxfiles = glob.glob('*fluxfile*.npz')

alldettimes = []
for f in fluxfiles:
    alldettimes.append(np.load(f)['srcarray']['mjd'][0])
fitsrc = []
for f in fluxfiles:
    for fit in np.load(f)['diffFLUX1'][['mjd','ra','dec']]:
        fitsrc.append(tuple(fit))
detsrcout = []
for t in np.unique(alldettimes):
    for row in fitsrc:
        if row[0]==t:
            detsrcout.append(row)
for src in detsrcout:
    min3c = cat3c[angsep(src[1],src[2],cat3c['ra'],cat3c['dec'])==angsep(src[1],src[2],cat3c['ra'],cat3c['dec']).min()]
    minvlssr = VLSSr50jy[angsep(src[1],src[2],VLSSr50jy['ra'],VLSSr50jy['dec'])==angsep(src[1],src[2],VLSSr50jy['ra'],VLSSr50jy['dec']).min()]
    print(minvlssr,min3c,src,angsep(src[1],src[2],cat3c['ra'],cat3c['dec']).min(),angsep(src[1],src[2],VLSSr50jy['ra'],VLSSr50jy['dec']).min())

