#!/usr/bin/env python

import os
import sys
import datetime
import argparse
from math import isclose

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

from GRITS.GPUgrouping import groupvertcols,grouphorizrows,groupdiagcols,groupdiagrows
from GRITS.constants import CYG_A, CAS_A, SGR_A, TAU_A, VIRGO_A, HERC_A, floatmax
from GRITS.imagetools import maskhorizon, pbcorroims, hdrtopixmask, hdrtoworld
from GRITS.ateamtools import getsunjup, getcasa, getcyga, getcrab, getvira
from GRITS.readoims import readoims
from tqdm import tqdm



def main(args):
    for filename in args.filename:
        print("--------",filename,"------")
        # DATA and HEADER
        ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata = readoims(filename)
        
        # Do pbcorr
        for c in range(nchan):
            XX,YY = pbcorroims(hdrlist[0],ngrid,c)
            imagedata[:,c,0,:,:]/=((XX+YY)/2)
            data[:,c,0,:,:]/=((XX+YY)/2)

        # Get sun and jupiter position for every time in the file. 
        # This part of the code could use some optimization. 
        # There is FORTRAN code that is publically available to
        # compute these positions. If it seems necessary,
        # I could attempt to adapt that code to the GPU
        sunpos, juppos = getsunjup(file_start, file_end, ints)

        # Get a mask for the parts of the image beyond 60 degrees from the center
        invalid = maskhorizon(data,hdrlist,ngrid,sRad)
        data[:,:,:,invalid[0], invalid[1]] = np.nan


        # Pre-calibration RMS measurement. 
        # IN: imagedata, sRad
        # OUT: rmsdata np.array
        # Set up center region for minrms calculation
        # We want a box in the center that does not go beyond the horizon. Also 
        # We need to account for the extra one pixel perimeter in the computations.
        boxhalflen = np.sin(np.pi/4)*sRad - 2
        imdata = BFArray(imagedata,dtype='f32',space='cuda')
        gpurmsdata = BFArray(np.zeros((ints,nchan,ngrid,ngrid)),dtype='f32',space='cuda')
        BFMap("""auto boxhalflen = %f;
                 auto center = %f/2;
                 auto x = (float) i - center + 0.5 ;
                 auto y = (float) j - center + 0.5 ;
                 auto intermediate = 0.0 ;
                 if ((x > -boxhalflen) && (x < boxhalflen) && (y > -boxhalflen) && (y < boxhalflen)){
                     for (auto offi=-2; offi<3; offi++) { 
                         for (auto offj=-2; offj<3; offj++){
                             intermediate+=dat(time,chan,0,i+offi,j+offj)*dat(time,chan,0,i+offi,j+offj);
                         }
                     }

                     rmsdat(time,chan,i,j) = sqrt(intermediate/25.0) ;
                 }
                 else {
                     rmsdat(time,chan,i,j) = 9999;
                     }""" % (boxhalflen,float(ngrid),),
                     {"dat":imdata, "rmsdat":gpurmsdata},
                     axis_names=('time','chan','i','j'), shape=(ints,nchan,ngrid,ngrid))
        gpurmsdata = gpurmsdata.copy('system')

        minrms = np.zeros((nchan))
        avindices = np.where((gpurmsdata!=0) & (gpurmsdata[:,c:,:]!=9999))
        for c in range(nchan):
            if c not in avindices[1]:
                continue
            minrms[c] = np.amin(gpurmsdata[:,c,:,:])

        # we can calibrate off of cyg a, cas a, virgo a, or tau a for now
        # We do all the masking of regions we don't want to detect transients from here
        # Also we are getting the max flux from the aforementioned a team sources for 
        # flux calibration
        cygaflux = [] 
        casaflux = []
        viraflux = []
        tauaflux = []
        completemask = np.zeros((len(data),ngrid,ngrid),dtype=bool)
        x = np.arange(ngrid) + 0.5
        y = np.arange(ngrid) + 0.5
        x,y = np.meshgrid(x,y)
        for i in range(len(data)):
            ateammask = np.zeros((ngrid,ngrid),dtype=bool)
            pixcoord = []
            for src in [SGR_A,HERC_A,sunpos[i],juppos[i]]:
                pixcoord.extend(hdrtopixmask(hdrlist[i],ngrid,sRad,src[0],src[1]))
            for p in pixcoord:
                if p[0]: 
                    currcond = ((x-p[0])**2 + (y-p[1])**2) < (7**2)
                    completemask[i] = completemask[i] | currcond
                    ateammask = ateammask | currcond
            cygacoord = hdrtopixmask(hdrlist[i],ngrid,sRad,CYG_A[0],CYG_A[1])
            for p in cygacoord:
                if p[0]: 
                    currcond = ((x-p[0])**2 + (y-p[1])**2) < (7**2)
                    ateammask = ateammask | currcond
                    completemask[i] = completemask[i] | currcond
                    cygmask = np.where(currcond)
                    if np.sum(currcond) >0:
                        for c in range(nchan):
                            cygaflux.append((imagedata[i,c,0,cygmask[0],cygmask[1]].max() - minrms[c],c))
            casacoord = hdrtopixmask(hdrlist[i],ngrid,sRad,CAS_A[0],CAS_A[1])
            for p in casacoord:
                if p[0]: 
                    currcond = ((x-p[0])**2 + (y-p[1])**2) < (7**2)
                    ateammask = ateammask | currcond
                    completemask[i] = completemask[i] | currcond
                    casmask = np.where(currcond)
                    if np.sum(currcond) >0:
                        for c in range(nchan):
                            casaflux.append((imagedata[i,c,0,casmask[0],casmask[1]].max() - minrms[c],c))
            viracoord = hdrtopixmask(hdrlist[i],ngrid,sRad,VIRGO_A[0],VIRGO_A[1])
            for p in viracoord:
                if p[0]: 
                    currcond = ((x-p[0])**2 + (y-p[1])**2) < (7**2)
                    ateammask = ateammask | currcond
                    completemask[i] = completemask[i] | currcond
                    virmask = np.where(currcond)
                    if np.sum(currcond) >0:
                        for c in range(nchan):
                            viraflux.append((imagedata[i,c,0,virmask[0],virmask[1]].max() - minrms[c],c))
            tauacoord = hdrtopixmask(hdrlist[i],ngrid,sRad,TAU_A[0],TAU_A[1])
            for p in tauacoord:
                if p[0]: 
                    currcond = ((x-p[0])**2 + (y-p[1])**2) < (7**2)
                    ateammask = ateammask | currcond
                    completemask[i] = completemask[i] | currcond
                    taumask = np.where(currcond)
                    if np.sum(currcond) >0:
                        for c in range(nchan):
                            tauaflux.append((imagedata[i,c,0,taumask[0],taumask[1]].max() - minrms[c],c))
            invalid = np.where( completemask[i] )


            data[i,:,:,invalid[0], invalid[1]] = np.nan

        freq = lambda c: (int(hdrlist[0]['start_freq'])  + ((c+1)*int(hdrlist[0]['bandwidth'])/2))*1e-6
        allcorr = []
        for cyg,c in cygaflux:
            if cyg!=0:
                allcorr.append((c,getcyga(freq(c))/cyg))
        for cas,c in casaflux:
            if cas!=0:
                allcorr.append((c,getcasa(freq(c))/cas))
        for vir,c in viraflux:
            if vir!=0:
                allcorr.append((c,getvira(freq(c))/vir))
        for tau,c in tauaflux:
            if tau!=0:
                allcorr.append((c,getcrab(freq(c))/tau))
        allcorr = np.array(allcorr,dtype=[('channel','i8'),('corrfac','f8')])
        calchans = np.unique(allcorr['channel'])
        removechan = []
        for c in range(nchan):
            if c not in calchans:
                print("Channel",c,"not calibrated. Removing this channel.")
                removechan.append(c)
        keepchan = []
        if removechan:
            for c in range(nchan):
                if c not in removechan:
                    keepchan.append(c)
            nchan = len(keepchan)
            keepchan = tuple(keepchan)
            imagedata = imagedata[:,keepchan,:,:,:]
            data = data[:,keepchan,:,:,:]
        if nchan < 4:
            print("less than 4 channels calibrated. Skipping file.")
            continue
        #### flux correct, 
        for c in range(nchan):
            if keepchan:
                chan = keepchan[c]
            else:
                chan = c
            corrflux = np.nanmean(allcorr['corrfac'][(allcorr['channel']==chan)])
            imagedata[:,c,:,:,:] = (imagedata[:,c,:,:,:] - minrms[c])*corrflux
            data[:,c,:,:,:] = data[:,c,:,:,:]*corrflux

        # Set up center region for minrms calculation
        # We want a box in the center that does not go beyond the horizon. Also 
        # We need to account for the extra one pixel perimeter in the computations.
        boxhalflen = np.sin(np.pi/4)*sRad - 2
        calimdata = BFArray(np.zeros(imagedata.shape),dtype='f32',space='system')
        calimdata[...] = imagedata
        calimdata = calimdata.copy('cuda')
        calrmsdata = BFArray(np.zeros((ints,nchan,4,ngrid,ngrid)),dtype='f32',space='cuda')
        immean = BFArray(np.zeros((ints,nchan,4,ngrid,ngrid)), dtype='f32',space='cuda')
        imstd = BFArray(np.zeros((ints,nchan,4,ngrid,ngrid)), dtype='f32',space='cuda')
        BFMap("""auto boxhalflen = %f;
                 auto center = %f/2;
                 auto x = (float) i - center + 0.5 ;
                 auto y = (float) j - center + 0.5 ;
                 auto mean = 0.0;
                 auto intermediaterms = 0.0;
                 auto intermediatemean = 0.0;
                 auto intermediatestd = 0.0;
                 if ((x > -boxhalflen) && (x < boxhalflen) && (y > -boxhalflen) && (y < boxhalflen)){
                     for (auto offi=-2; offi<3; offi++) { 
                         for (auto offj=-2; offj<3; offj++){
                             intermediaterms+= dat(time,chan,0,i+offi,j+offj)*dat(time,chan,0,i+offi,j+offj);
                             intermediatemean+=dat(time,chan,0,i+offi,j+offj);
                         }
                     }
                     rmsdat(time,chan,stokes,i,j) = sqrt(intermediaterms/25.0);
                     mean = intermediatemean/25.0;
                     meandat(time,chan,stokes,i,j) = mean;
                     for (auto offi=-2; offi<3; offi++) { 
                         for (auto offj=-2; offj<3; offj++){
                             intermediatestd+=(dat(time,chan,0,i+offi,j+offj) - mean)*(dat(time,chan,0,i+offi,j+offj) - mean);
                         }
                     }
                     std(time,chan,stokes,i,j) = sqrt(intermediatestd/25.0) ;
                 }
                 else {
                     rmsdat(time,chan,stokes,i,j) = 9999;
                     }""" % (boxhalflen,float(ngrid),),
                     {"dat":calimdata,"rmsdat":calrmsdata,'meandat':immean,'std':imstd},
                     axis_names=('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))
        calrmsdata = calrmsdata.copy('system')
        calimdata = calimdata.copy('system')
        rmsdata = []
        sndata = []
        for avdat,im in zip(calrmsdata[:,:,0,:,:],calimdata[:,:,:,:,:]):
            currms = np.amin(avdat[avdat!=0])
            rmsdata.append(currms)
            sndata.append(np.amax(im)/currms)
        immean= np.array(immean.copy('system'))
        imstd = np.array(imstd.copy('system'))
        meanstokeep = []
        stdtokeep = []
        meanarray = np.zeros((ints,nchan,4))
        stdarray = np.zeros((ints,nchan,4))
        for i in range(ints):
            for c in range(nchan):
                for s in range(4):
                    meanarr = immean[i,c,s]
                    stdarr = imstd[i,c,s]
                    nonzero = meanarr!=0
                    valid1 = meanarr!=9999
                    meanarr = meanarr[nonzero & valid1]
                    stdarr = stdarr[nonzero & valid1]
                    if np.sum(nonzero & valid1)==0:
                        print("all mean values are zero")
                        print(i,c,s)
                        meanarray[i,c,s] = floatmax
                        stdarray[i,c,s] = floatmax
                    else:
                        minabsmean = np.abs(meanarr)
                        minmean = np.amin(minabsmean)
                        mincond = minabsmean==minmean
                        meanarray[i,c,s] = meanarr[mincond]
                        stdarray[i,c,s] = stdarr[mincond]

        dataislands = np.copy(data)
        islandmasks = np.zeros(data.shape,dtype=bool)
        for i in range(ints):
            for j in range(nchan):
                k=0
                islandcond = np.nan_to_num(data[i,j,k]) < (meanarray[i,j,k] + stdarray[i,j,k]*5)
                islandmasks[i,j,k] = islandmasks[i,j,k] | ~islandcond 
               #  k=3 
               #  negislandcond = (np.nan_to_num(data[i,j,k]) > (meanarray[i,j,0] - stdarray[i,j,0]*5)) 
               #  islandcond = (np.nan_to_num(data[i,j,k]) < (meanarray[i,j,0] + stdarray[i,j,0]*5)) | negislandcond
               #  islandmasks[i,j,k] = islandmasks[i,j,k] | ~islandcond 
                   #      islandcond = np.zeros(islandcond.shape,
              #   for k in range(4):
                   #  if j>0:
                   #      negislandcond = (np.nan_to_num(data[i,j,k]) > (meanarray[i,j,0] - stdarray[i,j,0]*10)) 
                   #      islandcond = (np.nan_to_num(data[i,j,k]) < (meanarray[i,j,0] + stdarray[i,j,0]*10)) | negislandcond
                   #      islandcond = np.zeros(islandcond.shape,
                   #      sumisland = np.sum(~islandcond)
                   #      if sumisland>0:
                   #          print(sumisland)
                   #  else:
                #     whereislandcond = np.where(islandcond)
                #     dataislands[i,j,k,whereislandcond[0],whereislandcond[1]] = np.nan
        sumisland = np.sum(islandmasks,axis=1) 
        t1 = datetime.datetime.now()
        sumisland = BFArray(sumisland,space='cuda',dtype='f32')
        islanddata = BFArray(islandmasks, space='system', dtype='i8')
        rotateddata = BFArray(np.zeros((ints,nchan,4,2*ngrid-1,2*ngrid-1)),space='cuda',dtype='i8')
        islanddata = islanddata.copy('cuda')
        # MULTI Channel detections only
        BFMap("""if (sumisland(time,stokes,i,j) < (5/6)) isldata(time,chan,stokes,i,j) = 0;""",{'sumisland':sumisland,'isldata':islanddata},
            axis_names=('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))

        # LABELGROUPS
        BFMap("""
        auto counter = 0;
	for (auto i = 1; i < %i; i++) {
		for (auto j = 1; j < %i; j++) {
			if (0 != isldata(time,chan,stokes,i,j)) {
				counter++;
				isldata(time,chan,stokes,i,j) += counter - 1;
				if (0 != isldata(time,chan,stokes,i,j - 1)){
                                    isldata(time,chan,stokes,i,j) = isldata(time,chan,stokes,i,j - 1);
                                    }
			}
		}
     	}
            """ % (ngrid,ngrid,), 
            {'isldata':islanddata}, 
            axis_names=('time','chan','stokes'), shape=(ints,nchan,4))
        # GROUPCOLS
        groupvertcols(islanddata,ngrid)
        # GROUPROWS
        grouphorizrows(islanddata,ngrid)
        # GROUPCOLS
        groupvertcols(islanddata,ngrid)
        # ROTATE Island data
        BFMap("""rm(time,chan,stokes,i+j-1,%i-i+j) = isldata(time,chan,stokes,i,j);""" % (ngrid,),
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
        BFMap("""isldata(time,chan,stokes,i,j) = rm(time,chan,stokes,i+j-1,%i-i+j);""" % (ngrid,),
            {'isldata':islanddata,'rm':rotateddata},
            axis_names = ('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))
        # GROUPCOLS
        groupvertcols(islanddata,ngrid)
        # GROUPROWS
        grouphorizrows(islanddata,ngrid)
        # GROUPCOLS
        groupvertcols(islanddata,ngrid)
        # UPDATE ROTATED Island data
        BFMap("""rm(time,chan,stokes,i+j-1,%i-i+j) = isldata(time,chan,stokes,i,j);""" % (ngrid,),
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
        # COMPUTE vert length of non-rotated data
        BFMap("""
            for (auto i=0; i<%i; i++){
                if (isldata(time,chan,stokes,i,j)==index(ind)) length(time,chan,stokes,0,ind,j)+=1.0;
                }""" % (ngrid,),
                {'isldata':islanddata, 'length': length, 'index':index},
                axis_names = ('time','chan','stokes','j','ind'), shape=(ints,nchan,4,ngrid,maxind))
        # COMPUTE horiz length of non-rotated data
        BFMap("""
            for (auto j=0; j<%i; j++){
                if (isldata(time,chan,stokes,i,j)==index(ind)) length(time,chan,stokes,1,ind,j)+=1.0;
                }""" % (ngrid,),
                {'isldata':islanddata, 'length': length, 'index':index},
                axis_names = ('time','chan','stokes','i','ind'), shape=(ints,nchan,4,ngrid,maxind))
        # COMPUTE vert length of rotated data
        BFMap("""
            for (auto i=0; i<%i; i++){
                if (rm(time,chan,stokes,i,j)==index(ind)) length(time,chan,stokes,2,ind,j)+=1.0;
                }
            for (auto i=0; i<%i; i++){
                length(time,chan,stokes,2,ind,i) = (length(time,chan,stokes,2,ind,j)/2.0)*sqrt(2.0) ;
                }""" % (rotgrid,rotgrid,),
                {'rm':rotateddata, 'length': length,'index':index},
                axis_names = ('time','chan','stokes','j','ind'), shape=(ints,nchan,4,rotgrid,maxind))
        # COMPUTE horiz length of rotated data
        BFMap("""
            for (auto j=0; j<%i; j++){
                if (rm(time,chan,stokes,i,j)==index(ind)) length(time,chan,stokes,3,ind,j)+=1.0;
                }
            for (auto j=0; j<%i; j++){
                length(time,chan,stokes,3,ind,j) = (length(time,chan,stokes,3,ind,j)/2.0)*sqrt(2.0);
                }""" % (rotgrid,rotgrid,),
                {'rm':rotateddata, 'length': length, 'index':index},
                axis_names = ('time','chan','stokes','i','ind'), shape=(ints,nchan,4,rotgrid,maxind))
        BFMap("""
             auto maxlen = 0.0;
             auto maxlenk = 0;
             auto widthplane = 0;
             auto maxwidth =0.0 ; 
             for (auto k=0; k<4; k++){
                for (auto i=0; i<%i; i++){
                    if (length(time,chan,stokes,k,ind,i) > maxlen) {
                        maxlen = length(time,chan,stokes,k,ind,i);
                        maxlenk = k;
                        }
                    }
                }
             length(time,chan,stokes,0,ind,0) = maxlen;
             if (maxlenk==0) widthplane = 1;
             if (maxlenk==1) widthplane = 0;
             if (maxlenk==2) widthplane = 3;
             if (maxlenk==3) widthplane = 2;
             for (auto i=0; i<%i; i++){
                 if (length(time,chan,stokes,widthplane,ind,i) > maxwidth){
                     maxwidth = length(time,chan,stokes,widthplane,ind,i);
                 }
                 }
             width(time,chan,stokes,ind) = maxwidth;""" % (rotgrid,rotgrid,),
             {'length':length, 'width':width},
             axis_names = ('time','chan','stokes','ind'), shape=(ints,nchan,4,maxind))
        t3 = datetime.datetime.now()
        print("Measure length and width on the GPU:", (t3-t2))
        islanddata = islanddata.copy('system')
        rotateddata = rotateddata.copy('system')
        length = length.copy('system')
        width = width.copy('system')
        #### Set up extent ###
        pCntr = ngrid/2 + 1 + 0.5 * ((ngrid+1)%2)
        x = np.arange(1, data.shape[-2]+1, dtype=np.float32) - pCntr
        y = np.arange(1, data.shape[-1]+1, dtype=np.float32) - pCntr
        x /= -sRad
        y /= sRad
        x,y = np.meshgrid(x,y)
        x2 = x - 1 / sRad
        y2 = y - 1 / sRad
        extent = (x2.max(), x2.min(), y.min(), y.max())

        # Set up center region for mean and std calculations of diff ims
        # We want a box in the center that does not go beyond the horizon. Also 
        # We need to account for the extra one pixel perimeter in the computations.
        boxhalflen = np.sin(np.pi/4)*sRad - 1
        data = np.where(data==np.nan,floatmax,data)
        diffdata = BFArray(np.zeros(data.shape),dtype='f32',space='system')
        diffdata[...] = data
        diffdata = diffdata.copy('cuda')
        meandata = BFArray(np.zeros((ints,nchan,ngrid,ngrid)),dtype='f32',space='cuda')
        stddata = BFArray(np.zeros((ints,nchan,ngrid,ngrid)),dtype='f32',space='cuda')
        BFMap("""auto boxhalflen = %f;
                 auto center = %f/2;
                 auto x = (float) i - center + 0.5 ;
                 auto y = (float) j - center + 0.5 ;
                 auto mean = 0.0;
                 if ((x > -boxhalflen) && (x < boxhalflen) && (y > -boxhalflen) && (y < boxhalflen)){
                     auto tl = dat(time,chan,0,i-1,j+1);
                     auto tp = dat(time,chan,0,i,j+1);
                     auto tr = dat(time,chan,0,i+1,j+1);
                     auto lt = dat(time,chan,0,i-1,j);
                     auto mid = dat(time,chan,0,i-1,j);
                     auto rt = dat(time,chan,0,i+1,j);
                     auto bl = dat(time,chan,0,i-1,j-1);
                     auto bm = dat(time,chan,0,i,j-1);
                     auto br = dat(time,chan,0,i+1,j-1);
                     mean = (tl + tp + tr + lt + mid + rt + bl + bm + br)/9.0 ;
                     meandat(time,chan,i,j) = mean;
                     std(time,chan,i,j) = sqrt(((tl-mean)*(tl-mean) + (tp-mean)*(tp-mean) + (tr-mean)*(tr-mean) + (lt-mean)*(lt-mean) + (mid-mean)*(mid-mean) + (rt-mean)*(rt-mean) + (bl-mean)*(bl-mean) + (bm-mean)*(bm-mean) + (br-mean)*(br-mean))/9.0) ;
                 }
                 else {
                     meandat(time,chan,i,j) = 9999;
                     std(time,chan,i,j) = 9999;
                     }""" % (boxhalflen,float(ngrid),),
                     {"dat":diffdata, "meandat":meandata,"std":stddata},
                     axis_names=('time','chan','i','j'), shape=(ints,nchan,ngrid,ngrid))
        diffmean= np.array(meandata.copy('system'))
        diffstd = np.array(stddata.copy('system'))
        meanstokeep = []
        stdtokeep = []
        for i in range(ints):
            meanarr = diffmean[i]
            stdarr = diffstd[i]
            nonzero = meanarr!=0
            valid1 = meanarr!=9999
            valid2 = ~np.isnan(meanarr)
            toeval = np.where(nonzero & valid1 & valid2)
            meanarr = meanarr[toeval[0],toeval[1],toeval[2]]
            stdarr = stdarr[toeval[0],toeval[1],toeval[2]]
            minabsmean = np.abs(meanarr)
            minmean = np.amin(minabsmean)
            wheremin = np.where(minabsmean==minmean)
            meanstokeep.append(meanarr[wheremin[0]][0])
            stdtokeep.append(stdarr[wheremin[0]][0])
        meanofdiff = np.array(meanstokeep)
        stdofdiff = np.array(stdtokeep)
        data = np.where(data==floatmax,np.nan,data)

        maxwids = np.max(width,axis=(1,2,3))
        numsrcs = np.zeros(ints,dtype=int)
        for i in range(ints):
            for label in range(0,np.amax(islanddata[i,:,:,:,:])+1):
                for c in range(nchan):
                    for s in [0,3]:
                        srcwid = width[i,c,s,label-1]
                        if srcwid > 1:
                            numsrcs[i] += 1
        numsrccond = numsrcs < 2
        islandcondition = islanddata >0
        stdcondition = np.array(stdofdiff) < 15
        rmscond  = np.array(rmsdata) < 100
        absmeancond = np.abs(np.array(meanofdiff)) < 1e-3
        widcond = maxwids > 0
        drcondition = np.array(sndata) < 1e4
        whereislands = np.where(numsrccond & islandcondition.any(axis=(1,2,3,4)) & stdcondition & rmscond & absmeancond & widcond & drcondition)
        srcinfo = []
        srcid=0
        for i in np.sort(np.unique(whereislands[0])):
            for label in range(0,np.amax(islanddata[i,:,:,:,:])+1):
                for c in range(nchan):
                    for s in [0,3]:
                        srclen = length[i,c,s,0,label-1,0]
                        srcwid = width[i,c,s,label-1]
                        srclocs = np.argwhere(islanddata[i,c,s,:,:]==label)
                        if ((srclen > 0) and (srcwid >0)):
                            avx, avy = np.average(srclocs+0.5,axis=0)
                            radec = hdrtoworld(hdrlist[i],ngrid,sRad,avx,avy)
                            srcinfo.append((i,srcid,radec[0][0],radec[0][1])) 
                            srcid+=1
        srcarray = np.array(srcinfo, dtype=[('imid','i8'),('srcid','i8'),('ra','f8'),('dec','f8')])
        trackno = 0
        tracklist= []
        lasttrack = False
        justfinishedtrack = False
        nummeas = 3
        for i in range(len(srcarray['srcid'])-nummeas):
            idlist = []
            datelist = []
            sclist = []
            for j in range(nummeas):
                idlist.append(srcarray['srcid'][i+j])
            for id in idlist:
                src = srcarray[srcarray['srcid']==id]
                datelist.append(hdrlist[srcarray['imid'][srcarray['srcid']==id][0]]["start_time"])
                sclist.append(SkyCoord(src['ra'],src['dec'],unit='deg'))
            datediff = (np.amax(datelist) - np.amin(datelist))*3600*24
            posangle1 = sclist[0].position_angle(sclist[1]).degree
            posangle2 = sclist[1].position_angle(sclist[2]).degree
            datecond = (datediff < 120)
            poscond = isclose(posangle1,posangle2,rel_tol=0.1) 
            if poscond and datecond:
                for id in idlist:
                    tracklist.append((trackno,id))
                justfinishedtrack=True
            else:
                if tracklist and justfinishedtrack:
                    trackno += 1
                justfinishedtrack=False

                
        print('--------------')        
        print("trackno,srcid")
        print(tracklist)
        trackarray = np.array(tracklist,dtype=int)
        trackprop = []
        for trackid in np.unique(trackarray[:,0]):
            srcs = trackarray[:,1][trackarray[:,0]==trackid]
            angvel = []
            for i in range(len(srcs)-1):
                id1 = srcs[i]
                id2 = srcs[i+1]
                src1 = srcarray[srcarray['srcid']==id1]
                src2 = srcarray[srcarray['srcid']==id2]
                datediff = (hdrlist[srcarray['imid'][srcarray['srcid']==id2][0]]["start_time"] - hdrlist[srcarray['imid'][srcarray['srcid']==id1][0]]["start_time"])*3600*24
                angsep = np.abs(SkyCoord(src1['ra'],src2['dec'],unit='deg').separation(SkyCoord(src2['ra'],src2['dec'],unit='deg')).degree)
                angvel.append(angsep/datediff)
            print(angvel)
            trackprop.append((trackid,np.average(angvel)))
        print(trackprop)
        for i in np.unique(whereislands[0]):
            mean=meanofdiff[i]
            std=stdofdiff[i]
            rms=rmsdata[i]
            dynamicrange = sndata[i]
            skip = False
            chantoplot = 0
            fig, axs = plt.subplots(3,5,figsize=(30,15))
            ytextloc = 0.9
            for label in range(0,np.amax(islanddata[i,:,:,:,:])+1):
                for c in range(nchan):
                    for s in [0,3]:
                        srclen = length[i,c,s,0,label-1,0]
                        srcwid = width[i,c,s,label-1]
                        srclocs = np.argwhere(islanddata[i,c,s,:,:]==label)
                        if ((srclen > 0) and (srcwid >0)):
                            avx, avy = np.average(srclocs+0.5,axis=0)
                            radec = hdrtoworld(hdrlist[i],ngrid,sRad,avx,avy)
                            axs[1,4].text(0,ytextloc,f"{c},{s},{label},({length[i,c,s,0,label-1,0]},{width[i,c,s,label-1]}), pos:{np.round(radec,2)}")
                            ytextloc -= 0.15
            axs[1,4].set_axis_off()
            islandplot = np.zeros((ngrid,ngrid))
            for c in range(nchan):
                islandplot+=islanddata[i,c,0]
            axs[1,2].imshow(islandplot,vmin=0,vmax=1,origin='lower',extent=extent,cmap='binary')
            axs[1,2].set_title('I src')
            islandplot = np.zeros((ngrid,ngrid))
            for c in range(nchan):
                islandplot+=islanddata[i,c,3]
            axs[1,3].imshow(islandplot,vmin=0,vmax=1,origin='lower',extent=extent,cmap='binary')
            axs[1,3].set_title('V src')
            hdr = hdrlist[i]
            mintime = max(0,i-2)
            maxtime = min(ints-1,i+2)
            vmax = 8000
            for timenum, t in enumerate(range(mintime,maxtime+1)):
                currimage = np.zeros(data[t,chantoplot,0,:,:].shape)
                for c in range(nchan):
                    currimage += data[t,c,0,:,:]
                currimage/=nchan
                axs[0,timenum].imshow(currimage,vmin=-vmax/50,vmax=vmax/50,origin='lower',extent=extent)
                # axs[0,timenum].imshow(data[t,chantoplot,0,:,:],vmin=0,vmax=vmax/100,origin='lower',extent=extent)
            for timenum, t in enumerate(range(mintime,maxtime+1)):
                currimage = np.zeros(data[t,chantoplot,3,:,:].shape)
                for c in range(nchan):
                    currimage += data[t,c,3,:,:]
                currimage/=nchan
                axs[2,timenum].imshow(currimage,vmin=-vmax/50,vmax=vmax/50,origin='lower',extent=extent,cmap='coolwarm')
                # axs[0,timenum].imshow(data[t,chantoplot,0,:,:],vmin=0,vmax=vmax/100,origin='lower',extent=extent)
            currimage = np.zeros(imagedata[i,chantoplot,0,:,:].shape)
            for c in range(nchan):
                currimage += imagedata[i,c,0,:,:]
            currimage/=nchan
            axs[1,0].imshow(currimage,vmin=0,vmax=vmax,origin='lower',extent=extent)
            currimage = np.zeros(imagedata[i,chantoplot,0,:,:].shape)
            for c in range(nchan):
                currimage += imagedata[i,c,3,:,:]
            currimage/=nchan
            axs[1,1].imshow(currimage,vmin=-vmax/50,vmax=vmax/50,origin='lower',extent=extent,cmap="coolwarm")
            mjd = int(hdr['start_time'])
            mpm = int((hdr['start_time'] - mjd)*86400.0*1000.0)
            # determine if header is in seconds or days
            if round(hdr['int_len']) > 0:
                tInt = hdr['int_len']
            else:
                tInt = hdr['int_len']*86400.0
            dateObs = mjdmpm_to_datetime(mjd, mpm)
            fig.suptitle(f"{dateObs.isoformat()},std:{std:.2E},mean:{mean:.2E},rms:{rms:.2E},sn:{dynamicrange:.2E}")
            plotfile = filename.replace(".oims",f"-{i}.png")
            plt.savefig(plotfile)
            print("saving file as ",plotfile)
            plt.close()

        
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
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='be verbose during the conversion')
    args = parser.parse_args()
    main(args)
