#!/usr/bin/env python
import glob
import numpy as np
from GRITS.imagetools import angsep
from GRITS.catalogs import cat3c, VLSSr50jy
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mincounts",type=int,default=1,help="Minimum number of counts to make a histogram")
args = parser.parse_args()

fluxfiles = glob.glob('*fluxfile*.npz')

detsrcout = []
for f in fluxfiles:
    alldettimes = []
    fitsrc = []
    # alldettimes.append(tuple(np.load(f)['srcarray'][['mjd','filename','index']]))
    minrms = np.load(f)['minrms']
    fluxmeas = np.load(f)['diffFLUX1']
    for t,fname,i,c,s in np.load(f)['srcarray'][['mjd','filename','index','channel','stokes']]:
        curRMS = minrms[i,c,s]
        curflux = fluxmeas['f0'][fluxmeas['mjd']==t]
        if curflux.size==0:
            continue
        if (np.abs((curflux/curRMS)) < 10).any():
            continue
        alldettimes.append((t,fname,i))
    for fit in np.load(f)['diffFLUX1'][['mjd','ra','dec']]:
        fitsrc.append(tuple(fit))
    for t,fname,i in alldettimes:
        for row in fitsrc:
            outrow = []
            if row[0]==t:
                for r in row:
                    outrow.append(r)
                outrow.append(fname)
                outrow.append(i)
                detsrcout.append(outrow)
hist3C = []
histvlssr = []
for src in detsrcout:
    min3c = cat3c[angsep(src[1],src[2],cat3c['ra'],cat3c['dec'])==angsep(src[1],src[2],cat3c['ra'],cat3c['dec']).min()]
    minvlssr = VLSSr50jy[angsep(src[1],src[2],VLSSr50jy['ra'],VLSSr50jy['dec'])==angsep(src[1],src[2],VLSSr50jy['ra'],VLSSr50jy['dec']).min()]
    print(src[3],src[4],src[1],src[2],minvlssr['name'][0].replace(" ",""),angsep(src[1],src[2],VLSSr50jy['ra'],VLSSr50jy['dec']).min(),f"3C{min3c['3C'][0]}",angsep(src[1],src[2],cat3c['ra'],cat3c['dec']).min())
    hist3C.append((f"3C{min3c['3C'][0]}",angsep(src[1],src[2],cat3c['ra'],cat3c['dec']).min(),min3c['ra'][0],min3c['dec'][0]))
    histvlssr.append((minvlssr['name'][0].replace(" ",""),angsep(src[1],src[2],VLSSr50jy['ra'],VLSSr50jy['dec']).min(),minvlssr['ra'][0],minvlssr['dec'][0]))
hist3C = np.array(hist3C, dtype=[('name','<U32'),('dist','f8'),('ra','f8'),('dec','f8')])
histvlssr = np.array(histvlssr,dtype=[('name','<U128'),('dist','f8'),('ra','f8'),('dec','f8')])

bins = np.linspace(0.2,2.5,num=20)
for src3c in np.unique(hist3C['name']):
    curdata =hist3C['dist'][hist3C['name']==src3c]
    fig = plt.figure()
    n, bins, patches = plt.hist(curdata,bins=bins)
    if np.sum(n)<args.mincounts:
        plt.close()
        continue
    ax = plt.gca()
    ax.set_xlim(0.2,2.5)
    ax.set_title(f"{src3c} src sep")
    plt.savefig(f"{src3c}_separation.png")
    plt.close()

for srcvlssr in np.unique(histvlssr['name']):
    curdata =histvlssr[histvlssr['name']==srcvlssr]
    ra = curdata['ra'][0]
    dec = curdata['dec'][0]
    # No dupes
    if (angsep(ra,dec,hist3C['ra'],hist3C['dec']) < 0.1).any():
        min3c = cat3c[angsep(ra,dec,cat3c['ra'],cat3c['dec'])==angsep(ra,dec,cat3c['ra'],cat3c['dec']).min()]
        print(srcvlssr,"is a duplicate of",f"3C{min3c['3C'][0]}")
        continue
    fig = plt.figure()
    n,bins,patches = plt.hist(curdata['dist'],bins=bins)
    if np.sum(n)<args.mincounts:
        plt.close()
        continue
    ax = plt.gca()
    ax.set_xlim(0.2,2.5)
    ax.set_title(f"{srcvlssr} src sep")
    plt.savefig(f"{srcvlssr}_separation.png")
    plt.close()
