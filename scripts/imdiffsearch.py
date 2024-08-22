#!/usr/bin/env python

import os
import sys
import datetime
import traceback
start_epoch = datetime.datetime(1858, 11, 17, 00, 00, 00, 00)
import argparse
import sys
from configparser import ConfigParser 
from contextlib import closing

import matplotlib as mpl
font = {'weight' : 'bold',
        'size'   : 22}
mpl.rc('font', **font)
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from lsl.common.mcs import mjdmpm_to_datetime
from lsl.misc import parser as aph
from lsl import astro

from scipy.stats import norm
from scipy.interpolate import interp1d

import numpy as np

from astropy.coordinates import SkyCoord

from bifrost import ndarray as BFArray
from bifrost import map as BFMap 
from bifrost.device import set_devices_no_spin_cpu as BFNoSpinZone
BFNoSpinZone()


from GRITS.ateamtools import getsunjup, getcasa, getcyga, getcrab, getvira
from GRITS.catalogs import cat3c, VLSSr50jy
from GRITS.configtemplate import configfile
from GRITS.constants import floatmax
from GRITS.GPUimagestats import getminrms,fluxcalmask,getmultistat,getdiffstats
from GRITS.GPUgrouping import groupvertcols,grouphorizrows,groupdiagcols,groupdiagrows,labelgroups
from GRITS.imagetools import  angsep,maskhorizon, pbcorr, hdrtopixmask, hdrtoworld, maskislands
from GRITS.plotresults import plotresults
from GRITS.qc import restrictnumsrc,restrictangvel
from GRITS.readimage import readoims,readpims
from tqdm import tqdm

from lsl_toolkits.OrvilleImage import OrvilleImageDB
from lsl_toolkits.PasiImage import PasiImageDB
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("error",category=RuntimeWarning)

def plotislands(islanddata):
    islanddata = islanddata.copy('system')
    fig = plt.figure()
    subdat = islanddata[2,0,0,55:75,65:85]
    plt.imshow(subdat, cmap='coolwarm', vmin=0,vmax=1, origin='upper')
    for label in np.unique(subdat):
        ypix,xpix = np.where(subdat==label)
        for xp,yp in zip(xpix,ypix):
            plt.annotate(str(label), (xp-0.5,yp+0.5))
    plt.show()
    plt.close()

def main(filename,config,vmax=8000,noplot=False,quiet=False):
    uniquesrcid=0
    srclist = []
    obslist = []
    if not quiet:
        print("--------",filename,"------")
    # DATA and HEADER
    
    if ".oims" in filename:
        chunksize = int(config['Batch Control']['chunktime'])
        with closing(OrvilleImageDB(filename, 'r')) as db:
            fullsize = db.nint
            ngrid = db.header.ngrid
            nchan = db.header.nchan
        filetype="oims"
    elif ".pims" in filename:
        chunksize = int(config['Batch Control']['chunktime'])*4
        with closing(PasiImageDB(filename, 'r')) as db:
            fullsize = len(db)
            ngrid = db.header['xSize']
            nchan = 1 
        filetype="pims"
    else:
        print("unrecognized filename")
        exit()
    final = 0
    overlaptime = int(config['Batch Control']['overlaptime'])
    chunkt = 0
    chunkno = 1
    numchunks = np.ceil(fullsize/chunksize)
    allimagedata = np.zeros((fullsize,nchan,4,ngrid,ngrid))
    allislanddata = np.zeros((fullsize,nchan,4,ngrid,ngrid))
    fullhdrlist = []
    allrmsdata = np.zeros((fullsize,nchan))
    startindex = 0
    if fullsize==0:
        raise Exception("file is empty")

    while final<(fullsize-1):
        start = max(chunkt - overlaptime,0)
        final = min(fullsize-1,chunkt + chunksize)
        if ".oims" in filename:
            print("Read oims")
            ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata,station = readoims(filename,startind=start,stopind=final)
        elif ".pims" in filename:
            ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata,station = readpims(filename,startind=start,stopind=final)

        fullhdrlist.extend(hdrlist) 
            
        if not quiet:
            print(chunkno,"out of",numchunks,"chunks")
        chunkno+=1
        for c in range(nchan):
            XX,YY = pbcorr(hdrlist[0],ngrid,c,psize,station)
            imagedata[:,c,0,:,:]/=((XX+YY)/2)
            data[:,c,0,:,:]/=((XX+YY)/2)

            

        # Get sun and jupiter position for every time in the file. 
        # This part of the code could use some optimization. 
        # There is FORTRAN code that is publically available to
        # compute these positions. If it seems necessary,
        # I could attempt to adapt that code to the GPU
        sunpos, juppos = getsunjup(file_start, file_end, ints)

        # Get a mask for the horizon, initially search the whole image, restrict later
        # this is intended to reduce sources that are included due to edge effects
        invalid, circle = maskhorizon(data,hdrlist,ngrid,sRad,pbpercent=0.95)
        data[:,:,:,invalid[0], invalid[1]] = np.nan
        
        minrms = getminrms(sRad, imagedata,ints,nchan,ngrid,station)
        # Flux calibrate data and mask off A-Team
        imagedata, data, corrfluxlist = fluxcalmask(sRad, data, imagedata, hdrlist, minrms, sunpos, juppos, nchan, ngrid, station)
        # UN pbcorrect for a flat sky, we aren't measuring fluxes here anyway
        for c in range(nchan):
            imagedata[:,c,0,:,:]*=((XX+YY)/2)
            data[:,c,0,:,:]*=((XX+YY)/2)
        if not quiet:
            print("corrflux:",corrfluxlist)
        removechan = []
        outchannels=[]
        counter = 0
        for i,c in enumerate(corrfluxlist):
            if c<=0:
                removechan.append(c)
            else:
                outchannels.append(i)
             #    rmsdata[:,counter] = minrms[:,i]*c
                counter+=1
        outchannels = np.array(outchannels)
        allchannels = np.arange(data.shape[1])
        data  = np.copy(data[:,~np.isin(allchannels,removechan),:,:,:])
        imagedata  = np.copy(imagedata[:,~np.isin(allchannels,removechan),:,:,:])
        nchan = imagedata.shape[1]
        rmsdata = getminrms(sRad, data,ints,nchan,ngrid,station)
        
        if not quiet:
            medrms = np.median(rmsdata[1:])
            maxrms = np.amax(rmsdata[1:])
            print("min","max","median")
            print(np.amin(rmsdata[1:]),maxrms,medrms)
        
        islanddata = maskislands(data, rmsdata, ints, nchan, ngrid, nsigma=float(config['Quality Control']['nsigma']))

        islanddata = islanddata.copy("cuda")
    
        # plotislands(islanddata)



        t1 = datetime.datetime.now()
        # LABEL GROUPS 
        labelgroups(islanddata,ngrid,ints,nchan)
        # plotislands(islanddata)
        # GROUPCOLS
        groupvertcols(islanddata,ngrid)
        # plotislands(islanddata)
        # GROUPROWS
        grouphorizrows(islanddata,ngrid)
        # plotislands(islanddata)
        # GROUPCOLS
        groupvertcols(islanddata,ngrid)
        # plotislands(islanddata)
        # ROTATE Island data
        rotateddata = BFArray(np.zeros((ints,nchan,4,2*ngrid-1,2*ngrid-1)),space='cuda',dtype='u16')
        BFMap("""rm(time,chan,stokes,%i-i+j-1,i+j-2) = isldata(time,chan,stokes,i,j);""" % (ngrid,),
            {'isldata':islanddata,'rm':rotateddata},
            axis_names = ('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))
        rotgrid = 2*ngrid-1

        # GROUP DIAG COLS
        groupdiagcols(rotateddata,rotgrid)
        # GROUP DIAG ROWS
        groupdiagrows(rotateddata,rotgrid)
        # GROUP DIAG COLS
        groupdiagcols(rotateddata,rotgrid)
        # UPDATE UNROTATED Island data
        BFMap("""isldata(time,chan,stokes,i,j) = rm(time,chan,stokes,%i-i+j-1,i+j-2);""" % (ngrid,),
            {'isldata':islanddata,'rm':rotateddata},
            axis_names = ('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))
        # GROUPCOLS
        groupvertcols(islanddata,ngrid)
        # GROUPROWS
        grouphorizrows(islanddata,ngrid)
        # GROUPCOLS
        groupvertcols(islanddata,ngrid)
        # ROTATE Island data
        rotateddata = BFArray(np.zeros((ints,nchan,4,2*ngrid-1,2*ngrid-1)),space='cuda',dtype='u16')
        BFMap("""rm(time,chan,stokes,%i-i+j-1,i+j-2) = isldata(time,chan,stokes,i,j);""" % (ngrid,),
            {'isldata':islanddata,'rm':rotateddata},
            axis_names = ('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))
        rotgrid = 2*ngrid-1

        # GROUP DIAG COLS
        groupdiagcols(rotateddata,rotgrid)
        # GROUP DIAG ROWS
        groupdiagrows(rotateddata,rotgrid)
        # GROUP DIAG COLS
        groupdiagcols(rotateddata,rotgrid)
        # UPDATE UNROTATED Island data
        BFMap("""isldata(time,chan,stokes,i,j) = rm(time,chan,stokes,%i-i+j-1,i+j-2);""" % (ngrid,),
            {'isldata':islanddata,'rm':rotateddata},
            axis_names = ('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))

        t2 = datetime.datetime.now()
        if not quiet:
            print("Source Grouping Steps:",t2-t1)
        
        islanddata = islanddata.copy('system')
        rotateddata = rotateddata.copy('system')


        
        data = np.where(data==floatmax,np.nan,data)
        islandcopy = np.copy(islanddata)
        for i in range(ints):
            for c in range(nchan):
                hdr = hdrlist[i]
                for s in [0,3]:
                    for label in np.unique(islanddata[i,c,s,:,:][islanddata[i,c,s,:,:]>0]):
                        srclocs = np.argwhere(islanddata[i,c,s,:,:]==label)
                        avy, avx = np.average(srclocs+0.5,axis=0)
                        radec = hdrtoworld(hdrlist[i],ngrid,sRad,avx,avy,station)
                        # impose restrictions on conditions 
                        if (angsep(radec[0][0],radec[0][1],hdrlist[i]['center_ra'],hdrlist[i]['center_dec']) > float(config['Quality Control']['pbpercent'])*sRad):
                            islandcopy[i,c,s,:,:][islanddata[i,c,s,:,:]==label] = 0
        islanddata = np.copy(islandcopy)
        rmscond  = rmsdata < float(config['Quality Control']['maxrms'])

        numsrccond = restrictnumsrc(islanddata,ints,nchan,maxsrcs=int(config['Quality Control']['maxsrcs']))
        islandcondition = islanddata >0
        firstcut =  rmscond & numsrccond & islandcondition.any(axis=(2,3,4))  
     #    angvelcond = restrictangvel(np.where(firstcut), islanddata, hdrlist, nchan, ngrid,sRad,minangvel=float(config['Quality Control']['minangvel']),rtol=float(config['Quality Control']['rtol']))
        whereislands = np.where(firstcut ) 
        if not quiet:
            print(np.sum(numsrccond),"Images with a valid number of sources")
            print(np.sum(islandcondition.any(axis=(1,2,3,4))),"Images with sources")
            print(np.sum(numsrccond),np.sum(islandcondition.any(axis=(2,3,4))))
        islandfloat = np.zeros(islanddata.shape,dtype=float)
        islandfloat[...] = islanddata
       # islandfloat[:,:,:,circle[0],circle[1]] = np.nan
       #  imagedata[:,:,:,circle[0],circle[1]] = np.nan
        print("Iterating over sources")
        for ind in tqdm(range(len(whereislands[0]))):
            i = whereislands[0][ind]
            c = whereislands[1][ind]
            hdr = hdrlist[i]
            lastlabel=-1
            checkscint = []
            for s in [0,3]:
                for label in np.unique(islanddata[i,c,s,:,:][islanddata[i,c,s,:,:]>0]):
                    srclocs = np.argwhere(islanddata[i,c,s,:,:]==label)
                    if srclocs.size==0:
                        continue
                    avy, avx = np.average(srclocs+0.5,axis=0)
                    if label==lastlabel:
                        continue
                    radec = hdrtoworld(hdrlist[i],ngrid,sRad,avx,avy,station)
                    thesepix,otherchan,otherstokes = np.where(islanddata[i,:,:,srclocs[:,0],srclocs[:,1]]>0)
                    detchan = len(np.unique(otherchan))

                    mjd = int(hdr['start_time'])
                    mpm = int((hdr['start_time'] - mjd)*86400.0*1000.0)
                    # determine if header is in seconds or days
                    if round(hdr['int_len']) > 0:
                        tInt = hdr['int_len']
                    else:
                        tInt = hdr['int_len']*86400.0
                    dateObs = mjdmpm_to_datetime(mjd, mpm)
                    if (angsep(VLSSr50jy['ra'],VLSSr50jy['dec'],radec[0][0],radec[0][1]) < float(config['Quality Control']['catsep'])).any():
                        vlss = True
                    else:
                        vlss = False
                    if (angsep(cat3c['ra'],cat3c['dec'],radec[0][0],radec[0][1]) < float(config['Quality Control']['catsep'])).any():
                        c3 = True
                    else:
                        c3 = False
                    detchan=1
                    checkscint.append(c3 or vlss) 
                    if c3 or vlss:
                        islanddata[i,c,s,:,:][islanddata[i,c,s,:,:]==label] = 0
                    srclist.append((i, radec[0][0], radec[0][1],hdrlist[i]['start_time'],dateObs.isoformat(),c3,vlss,i+startindex,filename,outchannels[c],detchan))
                    lastlabel = label
                    uniquesrcid += 1
            if not noplot:
                if not np.all(checkscint):
                    plotresults(i,c,filename,rmsdata[i,c],islandfloat[:,c,:,:,:],imagedata[:,c,:,:,:],data[:,c,:,:,:],hdrlist[i],sRad,station,catsep=float(config['Quality Control']['catsep']),vmax=vmax,chunkt=startindex)
        for t in range(ints):
            for c in range(nchan):
                if numsrccond[t,c] and (rmsdata[t,c] < float(config['Quality Control']['maxrms'])):
                    if round(hdrlist[t]['int_len']) > 0:
                        tInt = hdrlist[t]['int_len']
                    else:
                        tInt = hdrlist[t]['int_len']*86400.0
                    dtstart = (start_epoch + datetime.timedelta(days=hdrlist[t]['start_time'])).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
                    dtend = (start_epoch + datetime.timedelta(days=hdrlist[t]['start_time']) + datetime.timedelta(seconds=tInt)).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
                    sens = rmsdata[t,c]
                    ra = 0.0
                    dec = 0.0
                    gaps="False"
                    fov=sRad*float(config['Quality Control']['pbpercent'])
                    obslist.append((dtstart,dtend,sens,ra,dec,gaps,fov))
        allimagedata[start:final+1,~np.isin(allchannels,removechan),:,:,:] = imagedata
        allrmsdata[start:final+1,~np.isin(allchannels,removechan)] =rmsdata 
        allislanddata[start:final+1,~np.isin(allchannels,removechan),:,:,:] = islanddata
        chunkt = chunkt + chunksize 
        startindex = chunkt - overlaptime

    if not quiet: 
        print("-------------------")
    return np.array(srclist,dtype=[('id','i8'),('ra','f8'),('dec','f8'),('mjd','f8'),('isodate','<U32'),('3c',"<U9"),('VLSSr50',"<U9"),('index','i8'),('filename',"<U256"),('channel',int),("nchan",int)]), allrmsdata, corrfluxlist, allimagedata, allislanddata, fullhdrlist, station, obslist, sRad

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='convert the images contained in one or more .oims files into FITS images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, nargs='+',
                        help='filename to convert')
    parser.add_argument('-b', '--background',type=float,
                        default = 0,
                        help='Uncorrected background flux')
    parser.add_argument('--configfile', type=str,
                        default='config.ini',
                        help="Configuration file.")
    parser.add_argument('-c', '--corrfac', type=float,
                        default=1,
                        help="Flux correction multiplicative factor")
    parser.add_argument('--channel', type=int,
                        help="Only image this channel")
    parser.add_argument('--noplot', action="store_true")
    parser.add_argument('-q','--quiet', action='store_true')
    parser.add_argument("--vmax", type=float, default=8000, help="controls contrast of plots")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='be verbose during the conversion')
    args = parser.parse_args()
    config = ConfigParser()
    if not os.path.isfile(args.configfile):
        print(f"{args.configfile} does not exist. Creating one and stopping.")
        with open(args.configfile,"w") as f:
            f.write(configfile)
    else:
        config.read(args.configfile)
        fullobslist = []
        outsrc = []
        for i,filename in enumerate(args.filename):
            print(i+1,"out of",len(args.filename))
            try:
                srcarr, minrms, corrfluxlist, allimagedata, allisland, fullhdrlist, station, obslist, sRad = main(filename,config,args.vmax,args.noplot,quiet=args.quiet)
                fullobslist.extend(obslist)
            except Exception as e:
                print(traceback.format_exc())
                print("Issue with ",filename,"skipping.")
            for i in np.unique(srcarr['index']):
                src = srcarr[srcarr['index']==i]
                nsrcleft = len(src)
                srcleft = src
                while nsrcleft > 0:
                    s1 = srcleft[0]
                    seps = np.zeros(len(src),dtype=bool)
                    for j,s in enumerate(src):
                        if s==s1:
                            seps[j]=True
                        else:  
                            seps[j] = angsep(s1['ra'],s1['dec'],s['ra'],s['dec']) < float(config['Quality Control']['catsep'])
                    matching2 = src[seps]
                    outsrc.append((src['isodate'],np.unique(len(matching2['channel']))))
                    srcleft = srcleft[~np.isin(srcleft,matching2)]
                    nsrcleft = srcleft.size
        outobsfile = f"obslist{datetime.datetime.now().timestamp()}.txt"
        with open(outobsfile,"w") as f:
            for o in fullobslist:
                f.write(f"{o[0]},{o[1]},{o[2]},{o[3]},{o[4]},{o[5]},{o[6]}\n")
        print("date","nchans")
        for o in outsrc:
            print(*o)
