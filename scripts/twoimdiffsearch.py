#!/usr/bin/env python

import os
import sys
import datetime
import argparse
from configparser import ConfigParser 
from multiprocessing import Pool

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

from GRITS.ateamtools import getsunjup, getcasa, getcyga, getcrab, getvira
from GRITS.configtemplate import configfile
from GRITS.constants import floatmax
from GRITS.GPUimagestats import getminrms,fluxcalmask,getmultistat,getdiffstats
from GRITS.GPUgrouping import groupvertcols,grouphorizrows,groupdiagcols,groupdiagrows,labelgroups,computelength
from GRITS.imagetools import angsep, maskhorizon, pbcorr, hdrtopixmask, hdrtoworld, maskislands
from GRITS.qc import restrictnumsrc,restrictangvel
from GRITS.readimage import readoims,readpims
from GRITS.catalogs import cat3c,VLSSr50jy
from tqdm import tqdm

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx
    
def constrainsources(i,hdr1,j,hdr2,npfile1mask,npfile2mask):
    file1time = hdr1['start_time']
    file2time =  hdr2['start_time']
    timeconstraint = (np.abs(file1time-file2time)*24*60*60) < 30
    hasmask = (np.nansum(npfile1mask[i,:,:,:,:]) > 0) & (np.nansum(npfile2mask[j,:,:,:,:]) > 0)
    if timeconstraint & hasmask: 
        return (i,j)


def main(args):
    if args.ignorematching:
        matchingfilepairs = []
        for f1 in args.station1:
            for f2 in args.station2:
                matchingfilepairs.append((f1,f2))
    else:
        f1times = []
        for f1 in args.station1:
            if "oims" in f1:
                file_start,file_end = readoims(f1,hdronly=True)
            else:
                file_start,file_end = readpims(f1,hdronly=True)
            f1times.append((f1,file_start + (file_end-file_start)/2))
        f2times = []
        for f2 in args.station2:
            if "oims" in f2:
                file_start,file_end = readoims(f2,hdronly=True)
            else:
                file_start,file_end = readpims(f2,hdronly=True)
            f2times.append((f2,file_start + (file_end-file_start)/2))
        matchingfilepairs = []
        for f1,f1t in f1times:
            for f2,f2t in f2times:
                difftime = np.abs(f1t-f2t)*24*60 # minutes
                if difftime < 30:
                    matchingfilepairs.append((f1,f2))
    for filepair in matchingfilepairs:
        for fileno,filename in enumerate(filepair):
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
            # Do pbcorr
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

            if fileno==0:
                islanddata = maskislands(data, stdarray, meanarray, ints, nchan, ngrid, nsigma=float(config['Quality Control']['finalnsigma']))
            else:
                islanddata = maskislands(data, stdarray, meanarray, ints, nchan, ngrid, nsigma=float(config['Quality Control']['initialnsigma']))

            # For the first file we just get the mask and nothing else. 
            # We will use the mask restrict the file2 sources.
            islanddata = islanddata.copy("system")
            if fileno==0:
                file1mask = BFArray(shape=islanddata.shape,space="system",dtype="i8")
                file1mask[...] = islanddata
                file1mask[file1mask==np.nan] = 0
                npfile1mask = np.zeros(file1mask.shape,dtype='i8')
                npfile1mask[...] = file1mask
                file1hdrlist = hdrlist
                ngrid1 = ngrid
                sRad1 = sRad
                diff1data = np.copy(data)
                continue
            elif fileno==1:
                file2mask = BFArray(shape=islanddata.shape,space="system",dtype="i8")
                file2mask[...] = islanddata
                file2mask[file2mask==np.nan] = 0
                file2mask = file2mask.copy('cuda')
                file1mask = file1mask.copy('cuda')
                print("constraining time and detections between stations")
                file2hdrlist = hdrlist
                matchgrid = BFArray(np.zeros((len(file1hdrlist),len(file2hdrlist))),space='cuda',dtype='i8')
                file1time = BFArray(np.array([hdr1['start_time'] for hdr1 in file1hdrlist]),space='cuda',dtype='f64')
                file2time = BFArray(np.array([hdr2['start_time'] for hdr2 in hdrlist]),space='cuda',dtype='f64')
                BFMap("""auto timediff = 0.0;
                         timediff = fabs(f1t(i) - f2t(j))*24.0*60.0*60.0;
                         auto f1msum = 0;
                         auto f2msum = 0;
                         for (auto c = 0; c < %i; c++){
                             for (auto s = 0; s < %i; s++){
                                 for (auto x=0; x< %i; x++){
                                     for (auto y=0; y<%i; y++){
                                         f1msum += f1m(i,c,s,x,y) ; 
                                         f2msum += f2m(j,c,s,x,y) ; 
                                         }
                                     }
                                 }
                             }
                         if ( (timediff < 30.0) && (f1msum > 0) && (f2msum > 0) ) {
                             matchgrid(i,j) = 1;
                             }"""        % (nchan,4,ngrid,ngrid,),
                    {'f1t':file1time,'f2t':file2time,'matchgrid':matchgrid,"f1m":file1mask,"f2m":file2mask},
                    axis_names = ('i','j'), shape=(len(file1hdrlist),len(file2hdrlist)))
                matchgrid = matchgrid.copy("system")
                file1mask = file1mask.copy("system")
                file2mask = file2mask.copy("system")
                ijkeep = [(i,j) for i in range(len(file1hdrlist)) for j in range(len(file2hdrlist)) if (matchgrid[i,j]==1)]
                islanddata = BFArray(np.zeros(islanddata.shape),space="system", dtype="i8")
                print("Checking for matches between stations")
                for i,j in tqdm(ijkeep):
                    # Images have different x and y sizes, get around this by
                    # converting to RA and Dec and do angular separation check
                    _,_,x1all,y1all = np.where(file1mask[i,:,:,:,:])
                    try:
                        radec1 =  hdrtoworld(file1hdrlist[i],ngrid1,sRad1,x1all,y1all)
                        ra1 = np.array([rd[0] for rd in radec1])
                        dec1 = np.array([rd[1] for rd in radec1])
                    except IndexError:
                        print(np.where(file1mask[i,:,:,:,:]))
                        print("empty coordinate list")
                        continue
                    _,_,x2all,y2all = np.where(file2mask[j,:,:,:,:])
                    radec2 =  hdrtoworld(hdrlist[j],ngrid,sRad,x2all,y2all)
                    ra2 = np.array([rd[0] for rd in radec2])
                    dec2 = np.array([rd[1] for rd in radec2])
                    for r2,d2 in zip(ra2,dec2):
                        if np.any(angsep(ra1,dec1,r2,d2) < float(config['Quality Control']['catsep'])):
                            x2, y2 = hdrtopixmask(hdrlist[j],ngrid,sRad,r2,d2)[0] 
                            islanddata[j,:,:,int(round(x2,0)),int(round(y2,0))] = 1 


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
            islanddata = islanddata.copy('cuda')
            if maxind==0:
                continue
            length = BFArray(np.zeros((ints,nchan,4,4,maxind,rotgrid)),space='cuda',dtype='f32')
            width = BFArray(np.zeros((ints,nchan,4,maxind)),space='cuda',dtype='f32')
            index = BFArray(np.arange(maxind)+1,space='cuda',dtype='i8')
            islanddata = islanddata.copy('cuda')
            # COMPUTE length 
            try:
                computelength(islanddata,rotateddata,length,width,index,ints,nchan,ngrid,rotgrid,maxind)
            except RuntimeError as e:
                print(e)
                continue
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
        
            pCntr = ngrid/2 + 1 + 0.5 * ((ngrid+1)%2)
            x = np.arange(1, data.shape[-2]+1, dtype=np.float32) - pCntr
            y = np.arange(1, data.shape[-1]+1, dtype=np.float32) - pCntr
            x /= -sRad
            y /= sRad
            x,y = np.meshgrid(x,y)
            x2 = x - 1 / sRad
            y2 = y - 1 / sRad
            extent = (x2.max(), x2.min(), y.min(), y.max())

            pCntr1 = ngrid1/2 + 1 + 0.5 * ((ngrid1+1)%2)
            x1 = np.arange(1, diff1data.shape[-2]+1, dtype=np.float32) - pCntr1
            y1 = np.arange(1, diff1data.shape[-1]+1, dtype=np.float32) - pCntr1
            x1 /= -sRad1
            y1 /= sRad1
            x1,y1 = np.meshgrid(x1,y1)
            x1_2 = x1 - 1 / sRad1
            y1_2 = y1 - 1 / sRad1
            extent1 = (x1_2.max(), x1_2.min(), y1_2.min(), y1_2.max())
            for i in np.unique(whereislands):
                if i==(len(data)-1):
                    continue
                mean=meanofdiff[i]
                std=stdofdiff[i]
                rms=rmsdata[i]
                dynamicrange = sndata[i]
                skip = False
                chantoplot = 0
                fig, axs = plt.subplots(4,5,figsize=(30,25))
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
                                VLSSrclose = np.argmin(angsep(VLSSr50jy['ra'],VLSSr50jy['dec'],radec[0][0],radec[0][1]))        
                                VLSSrnum = VLSSr50jy['sourcenum'][VLSSrclose] 
                                VLSSrsep = np.amin(angsep(VLSSr50jy['ra'],VLSSr50jy['dec'],radec[0][0],radec[0][1]))     
                                if VLSSrsep < float(config['Quality Control']['catsep']):
                                    axs[1,4].text(0,ytextloc,f"VLSSr #{VLSSrnum} is {round(VLSSrsep,1)} deg away")
                                    ytextloc -= 0.15
                                cat3cclose = np.argmin(angsep(cat3c['ra'],cat3c['dec'],radec[0][0],radec[0][1]))
                                cat3cnum = cat3c['3C'][cat3cclose]
                                cat3csep = np.amin(angsep(cat3c['ra'],cat3c['dec'],radec[0][0],radec[0][1]))
                                if cat3csep < float(config['Quality Control']['catsep']):
                                    axs[1,4].text(0,ytextloc,f"3C{cat3cnum} is {round(cat3csep,1)} deg away")
                                    ytextloc -= 0.15

                axs[1,4].set_axis_off()

                VLSSrclose = np.argmin(angsep(VLSSr50jy['ra'],VLSSr50jy['dec'],radec[0][0],radec[0][1]))        
                VLSSrname = VLSSr50jy['name'][VLSSrclose] 
                VLSSrsep = np.amin(angsep(VLSSr50jy['ra'],VLSSr50jy['dec'],radec[0][0],radec[0][1]))     
                islandplot = np.zeros((ngrid,ngrid))
                xc = np.arange(ngrid) + 0.5
                yc = np.arange(ngrid) + 0.5
                xc,yc = np.meshgrid(xc,yc)
                # circle = np.where(np.isclose((xc - ngrid/2)**2 + (yc-ngrid/2)**2,(2*sRad/3),rtol=0.2))
                for c in range(nchan):
                    islandplot+=islanddata[i,c,0]
                islandplot[circle[0],circle[1]] = np.nan
                xval = np.arange(ngrid)
                yval = np.arange(ngrid)
                axs[1,2].imshow(islandplot,vmin=0,vmax=1,origin='lower',extent=extent,cmap='viridis')

                axs[1,2].set_title('I src')

                islandplot = np.zeros((ngrid,ngrid))
                for c in range(nchan):
                    islandplot+=islanddata[i,c,3]
                islandplot[circle[0],circle[1]] = np.nan
                axs[1,3].imshow(islandplot,vmin=0,vmax=1,origin='lower',extent=extent,cmap='viridis')
                axs[1,3].set_title('V src')
                hdr = hdrlist[i]
                mintime = max(0,i-2)
                maxtime = min(ints-1,i+2)
                if not args.vmax2:
                    vmax=8000
                else:
                    vmax = args.vmax2
                if not args.vmax1:
                    vmax1=8000
                else:
                    vmax1 = args.vmax1

                allfile1times = [hdr['start_time'] for hdr in file1hdrlist]
                for timenum, t in enumerate(range(mintime,maxtime+1)):
                    currimage = np.zeros(data[t,chantoplot,0,:,:].shape)
                    for c in range(nchan):
                        currimage += data[t,c,0,:,:]
                    currimage/=nchan
                    axs[3,timenum].imshow(currimage,vmin=-vmax/50,vmax=vmax/50,origin='lower',extent=extent)
                    # Now plot file 1
                    file1time,file1timeindex = find_nearest(allfile1times,hdrlist[i]['start_time'])
                    currimage = np.zeros(diff1data[file1timeindex,chantoplot,0,:,:].shape)
                    for c in range(diff1data.shape[1]):
                        currimage += diff1data[file1timeindex,c,0,:,:]
                    currimage/=nchan
                    axs[0,timenum].imshow(currimage,vmin=-vmax1/50,vmax=vmax1/50,origin='lower',extent=extent1)
                    mjd = int(file1hdrlist[file1timeindex]['start_time'])
                    mpm = int((file1hdrlist[file1timeindex]['start_time'] - mjd)*86400.0*1000.0)
                    file1Obs = mjdmpm_to_datetime(mjd,mpm)
                    axs[0,2].set_title(f"{file1Obs.isoformat()[11:]}")
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
                currimage[circle[0],circle[1]] = np.nan
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
                plotfile = filename.replace(".oims",f"-{i}.png").replace(".pims",f"-{i}.png").rsplit('/')[-1]
                plt.savefig(plotfile)
                print("saving file as ",plotfile)
                plt.close()
                for obj in [data,imagedata,islanddata,length,width ,rotateddata]:
                    del obj

            
        print("-------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='convert the images contained in one or more .oims files into FITS images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('--channel', type=int,
                        help="Only image this channel")
    parser.add_argument('--station1', type=str, nargs='+',
                        help="images from the first station",required=True)
    parser.add_argument('--station2', type=str, nargs='+',
                        help="images from the station we check anti-concidence with",
                        required=True)
    parser.add_argument("--vmax1", type=float, help="manually specify vmax1")
    parser.add_argument("--vmax2", type=float, help="manually specify vmax2")
    parser.add_argument("--ignorematching", action="store_true", help="No matching files by time. Data is sorted already")
    args = parser.parse_args()
    config = ConfigParser()
    if not os.path.isfile("config.ini"):
        print("config.ini does not exist. Creating one and stopping.")
        with open("config.ini","w") as f:
            f.write(configfile)
    else:
        config.read("config.ini")
        main(args)
