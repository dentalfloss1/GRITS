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
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("error",category=RuntimeWarning)

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
from GRITS import OrvilleImageDB
from GRITS.plotresults import plotresults
from GRITS.qc import restrictnumsrc,restrictangvel
from GRITS.readimage import readoims,readpims
from tqdm import tqdm

from lsl_toolkits.PasiImage import PasiImageDB


def main(filename,config,vmax=8000,noplot=False,quiet=False):
    uniquesrcid=0
    srclist = []
    obslist = []
    if not quiet:
        print("--------",filename,"------")
    # DATA and HEADER
    
    if ".oims" in filename:
        chunksize = int(config['Batch Control']['chunktime'])
        with closing(OrvilleImageDB.OrvilleImageDB(filename, 'r')) as db:
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
    fullhdrlist = []
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

        # Get a mask for the parts of the image beyond 60 degrees from the center
        invalid, circle = maskhorizon(data,hdrlist,ngrid,sRad,pbpercent=float(config['Quality Control']['pbpercent']))
        data[:,:,:,invalid[0], invalid[1]] = np.nan
        
        minrms = getminrms(sRad, imagedata,ints,nchan,ngrid)
        print(minrms)
        print(imagedata[0,0,0,:,:].max())
        # Flux calibrate data and mask off A-Team
        imagedata, data, corrfluxlist = fluxcalmask(sRad, data, imagedata, hdrlist, minrms, sunpos, juppos, nchan, ngrid)
        print(corrfluxlist)
        removechan = []
        rmsdata =[]
        for i,c in enumerate(corrfluxlist):
            if c==0:
                removechan.append(c)
            else:
                rmsdata.append(minrms[i]*c)
        allchannels = np.arange(data.shape[1])
        data  = np.copy(data[:,~np.isin(allchannels,removechan),:,:,:])
        imagedata  = np.copy(imagedata[:,~np.isin(allchannels,removechan),:,:,:])
        nchan = imagedata.shape[1]
        

        # Get S/N and RMS from calibrated data
        sndata, _, stdarray, meanarray = getmultistat(sRad, imagedata, ints, nchan, ngrid)
        print(rmsdata)
        
        print(imagedata[0,0,0,:,:].max())
        meanofdiff, stdofdiff = getdiffstats(data,sRad,ints,nchan,ngrid)
        islanddata = maskislands(data, rmsdata, ints, nchan, ngrid, nsigma=float(config['Quality Control']['initialnsigma']))

        stdcondition = np.array(stdofdiff) > float(config['Quality Control']['maxstd'])
        rmscond  = np.amax(rmsdata) > float(config['Quality Control']['maxrms'])
        absmeancond = np.abs(np.array(meanofdiff)) > float(config['Quality Control']['maxabsmean'])
        pregroupconditions = np.where(stdcondition | rmscond | absmeancond)
        islanddata = islanddata.copy("system")
        islanddata[pregroupconditions,...] = 0

        islanddata = islanddata.copy("cuda")




        t1 = datetime.datetime.now()
        # LABEL GROUPS 
        labelgroups(islanddata,ngrid,ints,nchan)
        # GROUPCOLS
        groupvertcols(islanddata,ngrid)
        # GROUPROWS
        grouphorizrows(islanddata,ngrid)
        # GROUPCOLS
        groupvertcols(islanddata,ngrid)
        # ROTATE Island data
        rotateddata = BFArray(np.zeros((ints,nchan,4,2*ngrid-1,2*ngrid-1)),space='cuda',dtype='i8')
        BFMap("""rm(time,chan,stokes,i+j-2,%i-i+j-1) = isldata(time,chan,stokes,i,j);""" % (ngrid,),
            {'isldata':islanddata,'rm':rotateddata},
            axis_names = ('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))
        rotgrid = 2*ngrid-1

        # GROUP DIAG COLS
        groupdiagcols(rotateddata,rotgrid)
        # GROUPROWS
        groupdiagrows(rotateddata,rotgrid)
        # GROUPCOLS
        groupdiagcols(rotateddata,rotgrid)
        # UPDATE UNROTATED Island data
        BFMap("""isldata(time,chan,stokes,i,j) = rm(time,chan,stokes,i+j-2,%i-i+j-1);""" % (ngrid,),
            {'isldata':islanddata,'rm':rotateddata},
            axis_names = ('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))
        # GROUPCOLS
        groupvertcols(islanddata,ngrid)
        # GROUPROWS
        grouphorizrows(islanddata,ngrid)
        # GROUPCOLS
        groupvertcols(islanddata,ngrid)
        # UPDATE ROTATED Island data
        BFMap("""rm(time,chan,stokes,i+j-2,%i-i+j-1) = isldata(time,chan,stokes,i,j);""" % (ngrid,),
            {'isldata':islanddata,'rm':rotateddata},
            axis_names = ('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))

        t2 = datetime.datetime.now()
        if not quiet:
            print("Source Grouping Steps:",t2-t1)
        
        islanddata = islanddata.copy('system')
        rotateddata = rotateddata.copy('system')


        
        data = np.where(data==floatmax,np.nan,data)

        
        # impose restrictions on conditions 
        numsrccond = restrictnumsrc(islanddata,ints,nchan,maxsrcs=int(config['Quality Control']['maxsrcs']))
        print(np.sum(numsrccond),"Images with a valid number of sources")
        islandcondition = islanddata >0
        print(np.sum(islandcondition.any(axis=(1,2,3,4))),"Images with sources")
        drcondition = np.array(sndata) < float(config['Quality Control']['DynamicRange'])
        print(np.sum(drcondition),"Images with correct dynamic range")
        whereislands = np.where( numsrccond & islandcondition.any(axis=(1,2,3,4))  &  drcondition)

        angvelcond = restrictangvel(whereislands, islanddata, hdrlist, nchan, ngrid,sRad,minangvel=float(config['Quality Control']['minangvel']),rtol=float(config['Quality Control']['rtol']))
        nsigmacondition = np.zeros(ints,dtype=bool)
        islandloc = np.where(islandcondition) 
        maxrms = np.amax(rmsdata)
        for i in np.unique(islandloc[0]):
            islchan, islstokes, islx, isly = np.where((islanddata[i] > 0))
            brightpix = np.nanmax(data[i,islchan,islstokes,islx,isly])/maxrms
            nsigmacondition[i] = brightpix > float(config['Quality Control']['finalnsigma'])
        if not quiet:
            print(ints - np.sum(angvelcond),"slow moving sources rejected")
            print(np.sum(nsigmacondition), "passed nsigma condition")
        print(np.sum(nsigmacondition),np.sum(angvelcond),np.sum(numsrccond),np.sum(islandcondition.any(axis=(1,2,3,4))),np.sum(drcondition))
        whereislands = list(np.where(nsigmacondition & angvelcond & numsrccond & islandcondition.any(axis=(1,2,3,4)) &  drcondition )[0])
        islandfloat = np.zeros(islanddata.shape,dtype=float)
        islandfloat[...] = islanddata
        islandfloat[:,:,:,circle[0],circle[1]] = np.nan
       #  imagedata[:,:,:,circle[0],circle[1]] = np.nan
        for i in np.unique(whereislands):
            hdr = hdrlist[i]
            if not noplot:
                plotresults(i,filename,meanofdiff[i],stdofdiff[i],maxrms,sndata[i],islandfloat,imagedata,data,hdrlist[i],sRad,catsep=float(config['Quality Control']['catsep']),vmax=vmax,chunkt=startindex)
            lastlabel=-1
            for label in range(1,np.amax(islanddata[i,:,:,:,:])+1):
                for c in range(nchan):
                    for s in [0,3]:
                        srclocs = np.argwhere(islanddata[i,c,s,:,:]==label)
                        if srclocs.size==0:
                            continue
                        avx, avy = np.average(srclocs+0.5,axis=0)
                        if label==lastlabel:
                            continue
                        radec = hdrtoworld(hdrlist[i],ngrid,sRad,avx,avy)
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

                        srclist.append((i, radec[0][0], radec[0][1],hdrlist[i]['start_time'],dateObs.isoformat(),c3,vlss,i+startindex,filename))
                        lastlabel = label
                        uniquesrcid += 1
      #   for t in range(ints):
      #       dtstart = (start_epoch + datetime.timedelta(days=file_start)+
      #                  datetime.timedelta(seconds=(t*(file_end-file_start)/ints)*24.0*3600.0)).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
      #       dtend = (start_epoch + datetime.timedelta(days=file_start)+
      #                  datetime.timedelta(seconds=(t+1)*(((file_end-file_start)/ints)*24.0*3600.0))).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
      #       sens = rmsdata[i] 
      #       ra = hdrlist[i]['center_ra']
      #       dec = hdrlist[i]['center_dec']
      #       gaps="False"
      #       fov=sRad*float(config['Quality Control']['pbpercent'])
      #       obslist.append((dtstart,dtend,sens,ra,dec,gaps,fov))
        allimagedata[start:final+1,~np.isin(allchannels,removechan),:,:,:] = imagedata
        chunkt = chunkt + chunksize 
        startindex = chunkt - overlaptime

   #  print(obslist)
    if not quiet: 
        print("-------------------")
    return np.array(srclist,dtype=[('id','i8'),('ra','f8'),('dec','f8'),('mjd','f8'),('isodate','<U32'),('3c',"<U9"),('VLSSr50',"<U9"),('index','i8'),('filename',"<U256")]), minrms, corrfluxlist, allimagedata, fullhdrlist, station

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
    parser.add_argument('-c', '--corrfac', type=float,
                        default=1,
                        help="Flux correction multiplicative factor")
    parser.add_argument('--channel', type=int,
                        help="Only image this channel")
    parser.add_argument('-d', '--diff', action='store_true',
                        help='Generate diff images')
    parser.add_argument('-f', '--force', action='store_true',
                        help='force overwriting of FITS files')
    parser.add_argument('-i', '--index', type=int,
                        help='Only output this index')
    parser.add_argument('--noplot', action="store_true")
    parser.add_argument('-p', '--pbcorr', action='store_true',
                        help='Perform primary beam correction on Stokes I')
    parser.add_argument('-q','--quiet', action='store_true')
    parser.add_argument("--vmax", type=float, default=8000, help="controls contrast of plots")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='be verbose during the conversion')
    args = parser.parse_args()
    config = ConfigParser()
    if not os.path.isfile("config.ini"):
        print("config.ini does not exist. Creating one and stopping.")
        with open("config.ini","w") as f:
            f.write(configfile)
    else:
        config.read("config.ini")
        for i,filename in enumerate(args.filename):
            print(i+1,"out of",len(args.filename))
            try:
                main(filename,config,args.vmax,args.noplot,quiet=args.quiet)
            except Exception as e:
                print(traceback.format_exc())
                print("Issue with ",filename,"skipping.")
