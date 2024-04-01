#!/usr/bin/env python

import os
import sys
import datetime
import argparse
from configparser import ConfigParser 

import matplotlib as mpl
font = {'weight' : 'bold',
        'size'   : 22}
mpl.rc('font', **font)
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from lsl.misc import parser as aph
from lsl import astro

from scipy.stats import norm
from scipy.interpolate import interp1d

import numpy as np

from astropy.coordinates import SkyCoord

from bifrost import ndarray as BFArray
from bifrost import map as BFMap 

from GRITS.ateamtools import getsunjup, getcasa, getcyga, getcrab, getvira
from GRITS.configtemplate import configfile
from GRITS.constants import floatmax
from GRITS.GPUimagestats import getminrms,fluxcalmask,getmultistat,getdiffstats
from GRITS.GPUgrouping import groupvertcols,grouphorizrows,groupdiagcols,groupdiagrows,labelgroups,computelength
from GRITS.imagetools import  maskhorizon, pbcorr, hdrtopixmask,  maskislands
from GRITS.plotresults import plotresults
from GRITS.qc import restrictnumsrc,restrictangvel
from GRITS.readimage import readoims,readpims
from tqdm import tqdm



def main(args):
    for filename in args.filename:
        print("--------",filename,"------")
        # DATA and HEADER
        if ".oims" in filename:
            ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata,station = readoims(filename)
            filetype="oims"
        elif ".pims" in filename:
            ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata,station = readpims(filename)
            filetype="pims"
        else:
            print("unrecognized filename")
            exit()
        if file_start==0.0:
            print(filename, "has invalid data")
            continue
        fullhdrlist = hdrlist
        fulldata = data.copy()
        fullimage = imagedata.copy()
        final = 0
        overlaptime = int(config['Batch Control']['chunktime'])
        chunkt = 0
        if "oims" in filename:
            chunksize = int(config['Batch Control']['chunktime'])
        else:
            chunksize = int(config['Batch Control']['chunktime'])*4
        fullsize = ints
        chunkno = 1
        numchunks = np.ceil(fullsize/chunksize)
        while final<fullsize:
            start = max(chunkt - overlaptime,0)
            final = min(fullsize,chunkt + chunksize)
            hdrlist = fullhdrlist[chunkt:final]
            data = fulldata[chunkt:final]
            imagedata = fullimage[chunkt:final]
            chunkt = chunkt + chunksize
            ints = len(data)
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
            invalid, circle = maskhorizon(data,hdrlist,ngrid,sRad)
            data[:,:,:,invalid[0], invalid[1]] = np.nan
           

            minrms = getminrms(sRad, imagedata, ints, nchan, ngrid)

            # Flux calibrate data and mask off A-Team
            imagedata, data = fluxcalmask(sRad, data, imagedata, hdrlist, minrms, sunpos, juppos, nchan, ngrid)
            
            

            # Get S/N and RMS from calibrated data
            sndata, rmsdata, stdarray, meanarray = getmultistat(sRad, imagedata, ints, nchan, ngrid)

            islanddata = maskislands(data, stdarray, meanarray, ints, nchan, ngrid, nsigma=float(config['Quality Control']['initialnsigma']))

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
            print("Source Grouping Steps:",t2-t1)
            islanddata = islanddata.copy('system')
            maxind = np.amax(islanddata)
            length = BFArray(np.zeros((ints,nchan,4,4,maxind,rotgrid)),space='cuda',dtype='f32')
            width = BFArray(np.zeros((ints,nchan,4,maxind)),space='cuda',dtype='f32')
            index = BFArray(np.arange(maxind)+1,space='cuda',dtype='i8')
            islanddata = islanddata.copy('cuda')
            # COMPUTE length 
            computelength(islanddata,rotateddata,length,width,index,ints,nchan,ngrid,rotgrid,maxind)

            t3 = datetime.datetime.now()
            print("Measure length and width on the GPU:", (t3-t2))
            
            islanddata = islanddata.copy('system')
            rotateddata = rotateddata.copy('system')
            length = length.copy('system')
            width = width.copy('system')

            data = np.where(data==np.nan,floatmax,data)
            meanofdiff, stdofdiff = getdiffstats(data,sRad,ints,nchan,ngrid) 
            data = np.where(data==floatmax,np.nan,data)

            maxwids = np.max(width,axis=(1,2,3))
            # impose restrictions on conditions 
            zeroimagedata = (imagedata.sum(axis=(-1,-2))==0)
            invalidimages = np.where(zeroimagedata)
            for i in invalidimages[0]:
                for j in invalidimages[1]:
                    for k in invalidimages[2]:
                        if i!=0:
                            zeroimagedata[i-1,j,k] = True
                        if (i!=(ints-1)):
                            zeroimagedata[i+1,j,k] = True
            zeroimagedata = ~np.any(zeroimagedata,axis=(1,2))
            numsrccond = restrictnumsrc(islanddata,width,ints,nchan,maxsrcs=int(config['Quality Control']['maxsrcs']))
            islandcondition = islanddata >0
            stdcondition = np.array(stdofdiff) < float(config['Quality Control']['maxstd'])
            rmscond  = np.array(rmsdata) < float(config['Quality Control']['maxrms'])
            absmeancond = np.abs(np.array(meanofdiff)) < float(config['Quality Control']['maxabsmean'])
            widcond = maxwids > float(config['Quality Control']['minwid'])
            drcondition = np.array(sndata) < float(config['Quality Control']['DynamicRange'])
            whereislands = np.where(zeroimagedata & numsrccond & islandcondition.any(axis=(1,2,3,4)) & stdcondition & rmscond & absmeancond & widcond & drcondition)

            angvelcond = restrictangvel(whereislands, islanddata, hdrlist, length, width, nchan, ngrid,sRad,minangvel=float(config['Quality Control']['minangvel']),rtol=float(config['Quality Control']['rtol']))
            print(ints - np.sum(angvelcond),"slow moving sources rejected")
            nsigmacondition = np.zeros(ints,dtype=bool)
            islandloc = np.where(islandcondition) 
            rmsdataarray = np.array(rmsdata)
            for i in np.unique(islandloc[0]):
                islchan, islstokes, islx, isly = np.where((islanddata[i] > 0))
                brightpix = np.amax(data[i,islchan,islstokes,islx,isly])/rmsdataarray[i]
                nsigmacondition[i] = brightpix > float(config['Quality Control']['finalnsigma'])

            print(np.sum(nsigmacondition), "passed nsigma condition")
            whereislands = list(np.where(zeroimagedata & angvelcond & numsrccond & islandcondition.any(axis=(1,2,3,4)) & stdcondition & rmscond & absmeancond & widcond & drcondition & nsigmacondition)[0])
            islandfloat = islanddata.astype(float)
            islandfloat[:,:,:,circle[0],circle[1]] = np.nan
            imagedata[:,:,:,circle[0],circle[1]] = np.nan
            for i in np.unique(whereislands):
                plotresults(i,filename,meanofdiff[i],stdofdiff[i],rmsdata[i],sndata[i],islandfloat,imagedata,data,hdrlist[i],sRad,length,width,catsep=float(config['Quality Control']['catsep']),vmax=args.vmax,chunkt=chunkt)
            


        
        print("-------------------")

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
    parser.add_argument('-p', '--pbcorr', action='store_true',
                        help='Perform primary beam correction on Stokes I')
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
        main(args)
