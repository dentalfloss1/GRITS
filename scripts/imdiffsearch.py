#!/usr/bin/env python
import math
import os
import sys
import datetime
import traceback
start_epoch = datetime.datetime(1858, 11, 17, 00, 00, 00, 00)
import argparse
import sys
from configparser import ConfigParser 
from contextlib import closing
from bifrost.device import set_device

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
from astropy.time import Time 

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
from GRITS.plotresults import plotresults, plotLC
from GRITS.qc import restrictnumsrc,restrictangvel, checkgalactic
from GRITS.readimage import readoims,readpims
from GRITS.fitGaussian import fitGaussian

from lsl_toolkits.OrvilleImage import OrvilleImageDB
from lsl_toolkits.PasiImage import PasiImageDB
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("error",category=RuntimeWarning)

def makeZoomCoords(rad, circular = True):
    grid = np.mgrid[-rad : rad+1, -rad : rad+1].reshape((2, (rad*2+1)**2))
    if circular:
        dists = np.sqrt((grid**2).sum(axis = 0))
        iRegion = (dists < rad + 0.33).nonzero()[0]
        iEdge = (dists[iRegion] > rad - 0.67).nonzero()[0]
        return grid[:, iRegion], iEdge
    else:
        iEdge = ((np.abs(grid[0,:]) == rad) +
                 (np.abs(grid[1,:]) == rad)).nonzero()[0]
        return grid, iEdge

def plotislands(islanddata):
    try:
        islanddata = islanddata.copy('system')
    except:
        pass
    fig = plt.figure()
    subdat = islanddata[1,0,0,:,:]
    plt.imshow(subdat, cmap='coolwarm', vmin=0,vmax=1, origin='upper')
    for label in np.unique(subdat):
        ypix,xpix = np.where(subdat==label)
        for xp,yp in zip(xpix,ypix):
            plt.annotate(str(label), (xp-0.5,yp+0.5))
    plt.show()
    plt.close()

def main(filename,config,noplot=False,quiet=False,skip=0,endearly=None,maskstokes=None,positioncheck=False):
    uniquesrcid=0
    srclist = []
    obslist = []
    nstokes = 4 
    chunking = int( config['Batch Control']['chunktime'])
    if not quiet:
        print(filename)

            
    if ".oims" in filename:
        ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata,station = readoims(filename)
    elif ".pims" in filename:
        ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata,station = readpims(filename)
    else:
        print("unrecognized filename")
        exit()
        
    if ints==0:
        raise Exception("file is empty")
    beampatterns = []
    for c in range(nchan):
        XX,YY = pbcorr(hdrlist[0],ngrid,c,psize,station)
        beampatterns.append((XX,YY))
        imagedata[:,c,0,:,:]/=((XX+YY)/2)
        data[:,c,0,:,:]/=((XX+YY)/2)

        
    if endearly is None:
        endearly = ints+1
    t1 = datetime.datetime.now()
    # Get sun and jupiter position for every time in the file. 
    # This part of the code could use some optimization. 
    # There is FORTRAN code that is publically available to
    # compute these positions. If it seems necessary,
    # I could attempt to adapt that code to the GPU
    sunpos, juppos = getsunjup(file_start, file_end, ints)
    t2 = datetime.datetime.now()
    if not quiet:
        print("Get sun/jup:",t2-t1)

    t1 = datetime.datetime.now()
    # Get a mask for the horizon, initially search the whole image, restrict later
    # this is intended to reduce sources that are included due to edge effects
    invalid, circle = maskhorizon(data,hdrlist,ngrid,sRad,pbpercent=0.95)
    for s in range(nstokes):
        data[:,:,s,invalid[0], invalid[1]] = np.nan


    t2 = datetime.datetime.now()
    if not quiet:
        print("Mask Horizon:",t2-t1)
    
    t1 = datetime.datetime.now()
    minrms = getminrms(sRad, imagedata,ints,nchan,ngrid,station, chunking=chunking)
    t2 = datetime.datetime.now()
    if not quiet:
        print("Compute RMS:",t2-t1)
    t1 = datetime.datetime.now()
    # Flux calibrate data and mask off A-Team
    imagedata, data, corrfluxlist = fluxcalmask(sRad, data, imagedata, hdrlist, minrms[:,:,0], sunpos, juppos, nchan, ngrid, station, filename, positioncheck=positioncheck)
    t2 = datetime.datetime.now()
    if not quiet:
        print("calibrate:",t2-t1)
    # UNpbcorrect diff ims for a flat sky. We measure fluxes using the image data, so we keep that as pb corrected
    t1 = datetime.datetime.now()
    for c in range(nchan):
        XX,YY = beampatterns[c]
        imagedata[:,c,0,:,:]*=((XX+YY)/2)
        data[:,c,0,:,:]*=((XX+YY)/2)
    if not quiet:
        print("corrflux:",corrfluxlist)
    removechan = []
    outchannels=[]
    counter = 0
    flaggedtimes = [] 
    for i,c in enumerate(corrfluxlist):
        if c<=0:
            removechan.append(c)
            for i in range(len(imagedata)):
                flaggedtimes.append((i,c))
        else:
            outchannels.append(i)
         #    rmsdata[:,counter] = minrms[:,i]*c
            counter+=1

    t2 = datetime.datetime.now()
    if not quiet:
        print("un-pbcorr and remove bad channels:",t2-t1)
    nchan = imagedata.shape[1]
    t1 = datetime.datetime.now()
    nomaskdata = np.copy(imagedata)
    nomaskdata[1:] = imagedata[1:]-imagedata[:-1]
    rmsdata = getminrms(sRad, nomaskdata,ints,nchan,ngrid,station, chunking=chunking)
    t2 = datetime.datetime.now()
    if not quiet:
        print("RE-compute RMS:",t2-t1)
    if not quiet:
        try:
            medrms = np.median(rmsdata[1:])
            maxrms = np.amax(rmsdata[1:])
            print("min","max","median")
            print(np.amin(rmsdata[1:]),maxrms,medrms)
        except RuntimeWarning as e:
            print(e)
            print('could not compute median and max rms')

    flaggedtimes, islanddata = maskislands(imagedata, data, rmsdata, ints, nchan, ngrid, flaggedtimes,nsigma=float(config['Quality Control']['nsigma']))
    

    # skipping the first one
    islanddata[0,:,:,:,:] = 0
    for c in range(nchan):
        for s in range(4):
            flaggedtimes.append((0,c,s))
    for ind in range(ints): 
        if ((ind) < skip):
            islanddata[ind,:,:,:,:] = 0
            for c in range(nchan):
                for s in range(4):
                    flaggedtimes.append((ind,c,s))
        if ((ind) > endearly):
            islanddata[ind,:,:,:,:] = 0
            for c in range(nchan):
                for s in range(4):
                    flaggedtimes.append((ind,c,s))
    if maskstokes is not None:
        for m in maskstokes:
            islanddata[:,:,m,:,:] = 0
            for i in range(ints):
                for c in range(nchan):
                    flaggedtimes.append((i,c,m))


    for chunk in range(int(np.ceil(ints/chunking))):
        if chunk==0:
            start = 0
            end = min(chunking, ints)
        else:
            start=chunk*chunking 
            end=min(chunk*chunking + chunking ,ints)
        subints = end-start
        islandcuda = BFArray(islanddata[start:end], space='cuda', dtype='u16')
        t1 = datetime.datetime.now()
        # LABEL GROUPS 
        labelgroups(islandcuda,ngrid,subints,nchan)
        # GROUPCOLS
        groupvertcols(islandcuda,ngrid)
        # GROUPROWS
        grouphorizrows(islandcuda,ngrid)
        # GROUPCOLS
        groupvertcols(islandcuda,ngrid)
        # ROTATE Island data
        rotateddata = BFArray(np.zeros((subints,nchan,4,2*ngrid-1,2*ngrid-1)),space='cuda',dtype='u16')
        BFMap("""rm(time,chan,stokes,%i-i+j-1,i+j-2) = isldata(time,chan,stokes,i,j);""" % (ngrid,),
            {'isldata':islandcuda,'rm':rotateddata},
            axis_names = ('time','chan','stokes','i','j'), shape=(subints,nchan,4,ngrid,ngrid))
        rotgrid = 2*ngrid-1

        # GROUP DIAG COLS
        groupdiagcols(rotateddata,rotgrid)
        # GROUP DIAG ROWS
        groupdiagrows(rotateddata,rotgrid)
        # GROUP DIAG COLS
        groupdiagcols(rotateddata,rotgrid)
        # UPDATE UNROTATED Island data
        BFMap("""isldata(time,chan,stokes,i,j) = rm(time,chan,stokes,%i-i+j-1,i+j-2);""" % (ngrid,),
            {'isldata':islandcuda,'rm':rotateddata},
            axis_names = ('time','chan','stokes','i','j'), shape=(subints,nchan,4,ngrid,ngrid))
        # GROUPCOLS
        groupvertcols(islandcuda,ngrid)
        # GROUPROWS
        grouphorizrows(islandcuda,ngrid)
        # GROUPCOLS
        groupvertcols(islandcuda,ngrid)
        # ROTATE Island data
        rotateddata = BFArray(np.zeros((subints,nchan,4,2*ngrid-1,2*ngrid-1)),space='cuda',dtype='u16')
        BFMap("""rm(time,chan,stokes,%i-i+j-1,i+j-2) = isldata(time,chan,stokes,i,j);""" % (ngrid,),
            {'isldata':islandcuda,'rm':rotateddata},
            axis_names = ('time','chan','stokes','i','j'), shape=(subints,nchan,4,ngrid,ngrid))
        rotgrid = 2*ngrid-1

        # GROUP DIAG COLS
        groupdiagcols(rotateddata,rotgrid)
        # GROUP DIAG ROWS
        groupdiagrows(rotateddata,rotgrid)
        # GROUP DIAG COLS
        groupdiagcols(rotateddata,rotgrid)
        # UPDATE UNROTATED Island data
        BFMap("""isldata(time,chan,stokes,i,j) = rm(time,chan,stokes,%i-i+j-1,i+j-2);""" % (ngrid,),
            {'isldata':islandcuda,'rm':rotateddata},
            axis_names = ('time','chan','stokes','i','j'), shape=(subints,nchan,4,ngrid,ngrid))

        t2 = datetime.datetime.now()
        if not quiet:
            print("Source Grouping Steps:",t2-t1)
        
        islanddata[start:end] = islandcuda.copy('system')
    islanddata = np.array(np.copy(islanddata))
    maxDR=float(config['Quality Control']['DynamicRange'])
    for i in range(1,ints):
        for j in range(nchan):
            for k in range(nstokes):
                nomaskimdiff = imagedata[i,j,k,:,:] - imagedata[i-1,j,k,:,:]
                if rmsdata[i,j,k] == 0:
                    flaggedtimes.append((i,j,k))
                    islanddata[i,j,k,:,:] = 0
                else:
                    # Take absolute value since extreme negative values can also cause problems. 
                    s2n = np.amax(np.abs(nomaskimdiff))/rmsdata[i,j,k]
                    maxyplane, maxxplane = np.where(np.abs(nomaskimdiff)==np.amax(np.abs(nomaskimdiff)))
                    if s2n > maxDR:
                        if np.any(islanddata[i,j,k,maxyplane,maxxplane]!=0):
                            # In this case we have a bright non A-team source. We keep that source and get rid of all the
                            # others. They are likely sidelobes
                            for maxy,maxx in zip(maxyplane,maxxplane):
                                islanddata[i,j,k,:,:][islanddata[i,j,k,:,:]!=islanddata[i,j,k,maxy,maxx]] = 0 
                        else:
                            # Otherwise, the bright source is an A team source. Probably the sun. We flag that time and channel.
                            # We can't really trust the other polarizations either
                            for l in range(4):
                                flaggedtimes.append((i,j,l))
                                islanddata[i,j,l,:,:] = 0
    pblimit = float(config['Quality Control']['pbpercent'])
    invalid, circle = maskhorizon(data,hdrlist,ngrid,sRad,pbpercent=pblimit)
    islanddata[:,:,:,invalid[0], invalid[1]] = 0
    x = np.arange(ngrid) + 0.5
    y = np.arange(ngrid) + 0.5
    x,y = np.meshgrid(x,y)
    islanddata = np.where( ((x-ngrid/2.0)**2 + (y-ngrid/2.0)**2) > ((pblimit*sRad)**2) ,0,islanddata)
    data = np.where(data==floatmax,np.nan,data)
    rmscond  = rmsdata < float(config['Quality Control']['maxrms'])

    numsrccond = restrictnumsrc(islanddata,ints,nchan,maxsrcs=int(config['Quality Control']['maxsrcs']))
    islandcondition = islanddata >0
    firstcut =  rmscond & numsrccond & islandcondition.any(axis=(3,4))  
    whereislands = np.where(firstcut ) 
    if not quiet:
        print(np.sum(numsrccond),"Images with a valid number of sources")
        print(np.sum(islandcondition.any(axis=(1,2,3,4))),"Images with sources")
        print(np.sum(numsrccond),np.sum(islandcondition.any(axis=(2,3,4))))
    if not quiet:
        print("Iterating over sources")
        t1 = datetime.datetime.now()
    for ind in range(len(whereislands[0])):
        i = whereislands[0][ind]
        c = whereislands[1][ind]
        s = whereislands[2][ind]
        hdr = hdrlist[i]
        lastlabel=-1
        flagsrc = []
        for label in np.unique(islanddata[i,c,s,:,:][islanddata[i,c,s,:,:]>0]):
            yloc, xloc = np.where(islanddata[i,c,s,:,:]==label)
            if yloc.size==0:
                continue
            for x,y in zip(xloc,yloc):
                if data[i,c,s,y,x]==np.amax(data[i,c,s,yloc,xloc]):
                    avx = x
                    avy = y 
            if label==lastlabel:
                continue
            negflux = False
            pixdistfromctr = np.sqrt((avx-(ngrid/2))**2 + (avy-(ngrid/2))**2)
            detchan = c

            mjd = int(hdr['start_time'])
            mpm = int((hdr['start_time'] - mjd)*86400.0*1000.0)
            # determine if header is in seconds or days
            if round(hdr['int_len']) > 0:
                tInt = hdr['int_len']
            else:
                tInt = hdr['int_len']*86400.0
            dateObs = mjdmpm_to_datetime(mjd, mpm)
            VLSSrinFOV = VLSSr50jy[angsep(VLSSr50jy['ra'],VLSSr50jy['dec'],hdr['center_ra'],hdr['center_dec']) < 85]
            vlssrpixcoords = np.array(hdrtopixmask(hdr, ngrid, sRad, VLSSrinFOV['ra'], VLSSrinFOV['dec'], station))
            pixdist = np.sqrt((vlssrpixcoords[:,0]-avx)**2 + (vlssrpixcoords[:,1]-avy)**2)
            if (pixdist < float(config['Quality Control']['catsep'])).any():
                vlss = True
            else:
                vlss = False
            cat3cinFOV = cat3c[angsep(cat3c['ra'],cat3c['dec'],hdr['center_ra'],hdr['center_dec']) < 85]
            cat3cpixcoords = np.array(hdrtopixmask(hdr, ngrid, sRad, cat3cinFOV['ra'], cat3cinFOV['dec'], station))
            pixdist = np.sqrt((cat3cpixcoords[:,0]-avx)**2 + (cat3cpixcoords[:,1]-avy)**2)
            if (pixdist < float(config['Quality Control']['catsep'])).any():
                c3 = True
            else:
                c3 = False
            detchan=1
            flagsrc.append(c3 or vlss or negflux) 
            if c3 or vlss  or negflux:
                islanddata[i,c,s,:,:][islanddata[i,c,s,:,:]==label] = 0
            else:
                radec = hdrtoworld(hdr,ngrid,sRad,avx,avy,station)
                srclist.append((i, avx, avy,radec[0][0],radec[0][1],hdrlist[i]['start_time'],dateObs.isoformat(),c3,vlss,i,filename,c,detchan,s))
            lastlabel = label
            uniquesrcid += 1
        if not noplot:
            if not np.all(flagsrc):
                plotresults(i,c,s,filename,rmsdata[i,c,s],islanddata[:,c,:,:,:],imagedata[:,c,:,:,:],data[:,c,:,:,:],hdrlist[i],sRad,station,catsep=float(config['Quality Control']['catsep']))
    if not quiet:
        t2 = datetime.datetime.now()
        print(t2-t1)
        print("Making obslist")
        
    for t in range(ints):
        for c in range(nchan):
            for s in range(nstokes):
                if numsrccond[t,c,s] and (rmsdata[t,c,s] < float(config['Quality Control']['maxrms'])) and ((t,c,s) not in flaggedtimes):
                    if round(hdrlist[t]['int_len']) > 0:
                        tInt = hdrlist[t]['int_len']
                    else:
                        tInt = hdrlist[t]['int_len']*86400.0
                    dtstart = (start_epoch + datetime.timedelta(days=hdrlist[t]['start_time'])).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
                    dtend = (start_epoch + datetime.timedelta(days=hdrlist[t]['start_time']) + datetime.timedelta(seconds=tInt)).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
                    sens = rmsdata[t,c,s]
                    ra = 0.0
                    dec = 0.0
                    gaps="False"
                    fov=sRad*float(config['Quality Control']['pbpercent'])
                    obslist.append((dtstart,dtend,sens,ra,dec,gaps,fov,s))
    if not quiet:
        t3 = datetime.datetime.now()
        print(t3-t2)
    if not noplot:
        uniquecoords = np.unique([(src[3],src[4]) for src in srclist] ,axis=0)
        coordarray = np.array([(src[1],src[2],src[3],src[4]) for src in srclist])
        skiplist = []
        beamI = []
        prad = float(config['Quality Control']['statsep'])/psize
        xcoord = np.arange(ngrid)
        ycoord = np.arange(ngrid)
        xcoord,ycoord = np.meshgrid(xcoord, ycoord)
        for c in range(imagedata.shape[1]):
            XX,YY = pbcorr(hdrlist[0],ngrid,c,psize,station)
            beamI.append((XX+YY)/2)
        for i,(ra,dec) in enumerate(uniquecoords):
            u_x = coordarray[(coordarray[:,2]==ra) & (coordarray[:,3]==dec),0][0]
            u_y = coordarray[(coordarray[:,2]==ra) & (coordarray[:,3]==dec),1][0]
            if not quiet:
                print(i+1,"out of",(len(uniquecoords)),end="\r")
            if (ra,dec) in skiplist:
                continue
            for ra2,dec2 in uniquecoords[(i+1):]:
                u_x2 = coordarray[(coordarray[:,2]==ra) & (coordarray[:,3]==dec),0][0]
                u_y2 = coordarray[(coordarray[:,2]==ra) & (coordarray[:,3]==dec),1][0]
                pixdist = np.sqrt( (u_x - u_x2)**2 + (u_y - u_y2)**2)
                # If within the specified localization, we will skip it when we get to it in the primary loop
                if pixdist < float(config['Quality Control']['statsep']):
                    skiplist.append((ra2,dec2))
            dettimechan = []
            timelist = []
            for src_id,src_x,src_y,src_ra,src_dec,src_mjd,_,_,_,src_index,src_filename,src_channel,src_nchan,src_stokes in srclist:
                pixdist = np.sqrt( (u_x - src_x)**2 + (u_y - src_y)**2)
                if pixdist < float(config['Quality Control']['statsep']):
                    srcid = src_id # Grab a src id for a valid detection
                    dettimechan.append((src_index,src_channel,src_ra,src_dec,src_stokes))
                    timelist.append(str(src_index))
            dettimechan = np.array(dettimechan, dtype=[('index','i8'),('c','u2'),('ra','f8'),('dec','f8'),('stokes',int)])
            for udet in np.unique(dettimechan['index']):
                plotdata = []
                zoomplotdata = []
                curdettimechan = dettimechan[dettimechan['index']==udet]
                detstokes = np.unique(curdettimechan['stokes'])[0]
                if curdettimechan.size <= 0:
                    continue
                for timeind in range(1,len(hdrlist)):
                    if timeind in dettimechan['index']:
                        measRA = dettimechan['ra'][dettimechan['index']==timeind][0]
                        measDec = dettimechan['dec'][dettimechan['index']==timeind][0]
                        hasDETECT = True
                    else:
                        measRA = ra
                        measDec = dec
                        hasDETECT= False
                    x,y = hdrtopixmask(hdrlist[timeind], ngrid, sRad, measRA,measDec, station)[0]
                    for c in range(imagedata.shape[1]):
                        if detstokes==0:
                            corbeam = beamI[c]
                        else:
                            corbeam = 1
                        pbcorrallimage= (imagedata[timeind,c,detstokes]/corbeam) - (imagedata[timeind-1,c,detstokes]/corbeam)
                        if hasDETECT:
                            flux = pbcorrallimage[int(y),int(x)]
                        else:
                            flux = np.nanmean(np.where(((xcoord-x)**2 + (ycoord-y)**2) < prad**2,pbcorrallimage,np.nan))
                        midfreq = round((hdrlist[timeind]['start_freq']  + (c*hdrlist[timeind]['bandwidth']) + (hdrlist[timeind]['bandwidth']/2))*1e-6,2)
                        flagged = (timeind,c) in flaggedtimes
                        mjd = hdrlist[timeind]['start_time']
                        hourspastmid = (mjd - np.floor(mjd))*24
                        plotdata.append((timeind,flux,c,midfreq,flagged,False,hourspastmid))
                mintime = max(0,curdettimechan['index'].min()-12)
                maxtime = min(len(hdrlist)-1, curdettimechan['index'].max()+12)
                if mintime > maxtime:
                    raise RuntimeWarning
                plotdata = np.array(plotdata, dtype=[('index','i8'),('flux','f8'),('c','u2'),('freq','f4'),('flagged',bool),('statmatch',bool),('hrpst','f8')])
                zoomplotdata = np.copy(plotdata[(plotdata['index']>mintime) & (plotdata['index'] < maxtime)])
                tdet = Time(hdrlist[udet]['start_time'],format='mjd',scale='utc')
                if plotdata.size>0 :
                    plotLC(src_filename.replace(".oims",f"stokes{detstokes}.oims").replace(".pims",f"stokes{detstokes}.pims"), plotdata, zoomplotdata, curdettimechan,srcid,str(udet),tdet.isot,ra,dec)
    




    srcarray =  np.array(srclist,dtype=[('id','i8'),('x','f8'),('y','f8'),('ra','f8'),('dec','f8'),('mjd','f8'),('isodate','<U32'),('3c',"<U9"),('VLSSr50',"<U9"),('index','i8'),('filename',"<U256"),('channel',int),("nchan",int),("stokes",int)])           
    if not quiet:
        print("Grouping sources")
        t1 = datetime.datetime.now()
    for udate in np.unique(srcarray['mjd']):
        cursrc = srcarray[srcarray['mjd']==udate]
        if len(cursrc) > 1:
            completedsrc = []
            for src1 in cursrc:
                if src1 not in completedsrc:
                    for src2 in cursrc:
                        if src1!=src2:
                            pixdist = np.sqrt( (src1['x']-src2['x'])**2 + (src1['y'] - src2['y'])**2)
                            if (pixdist < float(config['Quality Control']['catsep'])) and (src1['channel']!=src2['channel']):
                                srcarray['nchan'][(srcarray['mjd']==udate) & (srcarray['ra']==src1['ra']) & (srcarray['dec']==src1['dec'])]+=1
                                completedsrc.append(src2)
    if not quiet: 
        t2 = datetime.datetime.now()
        print(t2-t1)
        print("-------------------")
    return srcarray, rmsdata, corrfluxlist, imagedata, islanddata, hdrlist, station, obslist, sRad, flaggedtimes

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
    parser.add_argument('--device', type=int, default=0,
                        help="Use this GPU device number")
    parser.add_argument('--noplot', action="store_true")
    parser.add_argument('-q','--quiet', action='store_true')
    parser.add_argument('--skip',type=int,default=0)
    parser.add_argument('--endearly',type=int,default=None)
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='be verbose during the conversion')
    parser.add_argument("--targetsource", type=float, nargs=2,
                        help='Look for source at this ra dec')
    parser.add_argument("--maskstokes",nargs="+",type=int, default=None)
    parser.add_argument("--positioncheck",action="store_true")
    args = parser.parse_args()
    config = ConfigParser()
    set_device(args.device)
    if not os.path.isfile(args.configfile):
        print(f"{args.configfile} does not exist. Creating one and stopping.")
        with open(args.configfile,"w") as f:
            f.write(configfile)
    else:
        config.read(args.configfile)
        fullobslist = []
        outsrc = []
        if args.positioncheck:
            for posfile in ["casoffsets.csv","cygoffsets.csv"]:
                if os.path.exists(posfile):
                    os.remove(posfile)
        for i,filename in enumerate(args.filename):
            print(i+1,"out of",len(args.filename))
            try:
                srcarr, minrms, corrfluxlist, allimagedata, allisland, fullhdrlist, station, obslist, sRad,allflaggedtimes = main(filename,config,args.noplot,quiet=args.quiet,skip=args.skip,endearly=args.endearly,maskstokes=args.maskstokes,positioncheck=args.positioncheck)
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
                            pixdist = np.sqrt( (s1['x']-s['x'])**2 + (s1['y'] - s['y'])**2)
                            seps[j] = pixdist < float(config['Quality Control']['catsep'])
                    matching2 = src[seps]
                    outsrc.append((src['isodate'],np.unique(len(matching2['channel']))))
                    srcleft = srcleft[~np.isin(srcleft,matching2)]
                    nsrcleft = srcleft.size
            if args.targetsource:
                for s in srcarr:
                    if angsep(s['ra'],s['dec'],args.targetsource[0],args.targetsource[1]) < float(config['Quality Control']['catsep']):
                        print(s)
        outobsfile = f"obslist{datetime.datetime.now().timestamp()}.txt"
        with open(outobsfile,"w") as f:
            for o in fullobslist:
                f.write(f"{o[0]},{o[1]},{o[2]},{o[3]},{o[4]},{o[5]},{o[6]},{o[7]}\n")
        if not args.quiet:
            print("date","nchans")
      #   for o in outsrc:
      #       if not quiet:
                # print(*o)
