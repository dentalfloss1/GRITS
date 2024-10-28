#!/usr/bin/env python
from configparser import ConfigParser
import numpy as np
from math import isclose

from tqdm import tqdm

import datetime
import os 
import argparse
import imdiffsearch
from astropy.time import Time
from bifrost.device import set_device

from GRITS.configtemplate import configfile
from GRITS.plotresults import plotresults,plotsource,plotLC
from GRITS.readimage import matchfilepairs
from GRITS.imagetools import pimsfitsout, oimsfitsout, exportcrtf,pbcorr, hdrtopixmask, angsep, sourcefitting
from GRITS.fitGaussian import fitGaussian
import glob
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



if __name__=="__main__":
    t1 = datetime.datetime.now()
    # Parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--configfile1', type=str,
                        default='config.ini',
                        help="Configuration file.")
    parser.add_argument('--configfile2', type=str,
                        default='config.ini',
                        help="Configuration file.")
    parser.add_argument('--device', type=int, default=0,
                        help="Use this GPU device number")
    parser.add_argument('--station1', type=str, nargs='+',
                        help="images from the first station",required=True)
    parser.add_argument('--station2', type=str, nargs='+',
                        help="images from the station we check anti-concidence with",
                        required=True)
    parser.add_argument("--bigrun", action="store_true",
                        help="search all PASI and Orville files")
    parser.add_argument("--timedelta",default=5,type=float, help="Time in seconds between source appearing in the two source lists")
    parser.add_argument("--makefits",action="store_true", help="Make fits and region files")
    parser.add_argument("--resume",action="store_true", help="Resume previous run by checking for obsfiles")
    parser.add_argument("--verbose",action="store_true")
    parser.add_argument("--maskstokes",nargs="+",type=int, default=None)
    args = parser.parse_args()
    if args.verbose:
        quiet = False
    else:
        quiet = True
    set_device(args.device)
    config1 = ConfigParser()
    config2 = ConfigParser()
    if args.bigrun:
        config1.read("orvilleconfig.ini")
        config2.read("pasiconfig.ini")
        if os.path.exists("matchedfiles.npy"):
            matchedfiles= np.load("matchedfiles.npy")
        else:
            station1 = glob.glob("/FileStore/Orville/*/*.oims")
            station2 = glob.glob("/FileStore/PASI/*/*.pims")
            matchedfiles = matchfilepairs(station1,station2)
            print(matchedfiles)
            np.save("matchedfiles.npy",matchedfiles)
    else:
        # Read config.ini or generate new one
        if not os.path.isfile(args.configfile1):
            print(f"{args.configfile1} does not exist. Creating one and stopping.")
            with open(args.configfile1,"w") as f:
                f.write(configfile)
            exit()
        else:
            config1.read(args.configfile1)
        if not os.path.isfile(args.configfile2):
            print(f"{args.configfile2} does not exist. Creating one and stopping.")
            with open(args.configfile2,"w") as f:
                f.write(configfile)
            exit()
        else:
            config2.read(args.configfile2)

        # run main execution loop
        import glob
        matchedfiles = matchfilepairs(args.station1,args.station2) 
    lastfilename1 = ""
    lastfilename2 = ""
    results = []
    fullobslist = []
    fullsrclist = []
    uniqueid=0
    for filename1,filename2 in tqdm(matchedfiles):
        obsfile1= filename1.split('/')[-1].replace(":",'-')+"_obsfile.npy"
        obsfile2= filename2.split('/')[-1].replace(":","-")+"_obsfile.npy"
        if args.resume:
            if os.path.exists(obsfile1) and os.path.exists(obsfile2):
                continue
        if lastfilename1!=filename1:
            dat1,minrms1,corrfluxlist1,imagedata1,islands1,hdrlist1,station1,obslist1, sRad1, allflaggedtimes1 = imdiffsearch.main(filename1,config1,noplot=True,quiet=quiet,maskstokes=args.maskstokes)
            diffdata1 = np.copy(imagedata1)
            diffdata1[1:] = diffdata1[1:] - diffdata1[:-1]
        if lastfilename2!=filename2:
            dat2,minrms2,corrfluxlist2,imagedata2,islands2,hdrlist2,station2, obslist2,sRad2,allflaggedtimes2 = imdiffsearch.main(filename2,config2,noplot=True,quiet=quiet,maskstokes=args.maskstokes)
            diffdata2 = np.copy(imagedata2)
            diffdata2[1:] = diffdata2[1:] - diffdata2[:-1]
        if (dat1 is None) or (dat2 is None):
            continue
        lastfilename1 = filename1
        lastfilename2 = filename2
        np.save(obsfile1,obslist1)
        np.save(obsfile2,obslist2)
        keepdat1 = dat1[(dat1['3c']=="False") & (dat1['VLSSr50']=="False")]
        keepdat2 = dat2[(dat2['3c']=="False") & (dat2['VLSSr50']=="False")]
        plottedlc = []
        srcfile1= filename1.split('/')[-1]+"_srcfile"
        allmatch = []
        for s1 in keepdat1:
            addsource = False
            currentmatch = [(s1,station1)]
            for m in allmatch:
                if s1 in [row[0] for row in m]:
                    continue
            simul = np.abs(s1['mjd'] - keepdat1['mjd'])*24.0*60.0*60.0 < args.timedelta
            if np.sum(simul) > 0:
                matching1 = keepdat1[simul]
                for s1_2 in matching1:
                    if s1_2!=s1:
                         # x and y pixel in image 2 from position in ra,dec image 1
                        x1_2f1, y1_2f1 = hdrtopixmask(hdrlist1[s1_2['index']], imagedata1.shape[-1],sRad1, s1['ra'], s1['dec'],station1)[0]
                        x1f1_2, y1f1_2 = hdrtopixmask(hdrlist1[s1['index']], imagedata1.shape[-1],sRad1, s1_2['ra'], s1_2['dec'],station1)[0]
                        sourcesep1 =  np.sqrt((s1['x']-x1f1_2)**2 + (s1['y']-y1f1_2)**2)
                        sourcesep1_2 =  np.sqrt((s1_2['x']-x1_2f1)**2 + (s1_2['y']-y1_2f1)**2)
                        sourcesep = min(sourcesep1,sourcesep1_2)
                        if sourcesep < float(config1['Quality Control']['statsep']):
                            currentmatch.append((s1_2,station1))
            simul = np.abs(s1['mjd'] - keepdat2['mjd'])*24.0*60.0*60.0 < args.timedelta
            if np.sum(simul) > 0:
                matching2 = keepdat2[simul]
                for s2 in matching2:
                     # x and y pixel in image 2 from position in ra,dec image 1
                    x2f1, y2f1 = hdrtopixmask(hdrlist2[s2['index']], imagedata2.shape[-1],sRad2, s1['ra'], s1['dec'],station2)[0]
                    x1f2, y1f2 = hdrtopixmask(hdrlist1[s1['index']], imagedata1.shape[-1],sRad1, s2['ra'], s2['dec'],station1)[0]
                    sourcesep1 =  np.sqrt((s1['x']-x1f2)**2 + (s1['y']-y1f2)**2)
                    sourcesep2 =  np.sqrt((s2['x']-x2f1)**2 + (s2['y']-y2f1)**2)
                    sourcesep = min(sourcesep1,sourcesep2)
                    if sourcesep < float(config1['Quality Control']['statsep']):
                        addsource = True
                        currentmatch.append((s2,station2))
                        if args.verbose:
                            print("----separation----")
                            print("index:",s1['index'],s1['x'],x1f2,s1['y'],y1f2,s2['x'],x2f1,s2['y'],y2f1,sourcesep1,sourcesep2,s1,s2,sourcesep,angsep(s1['ra'],s1['dec'],s2['ra'],s2['dec']))
                            print("--------")
            if addsource:
                allmatch.append(currentmatch)
        for m in allmatch:
            print(m)
            stat1srcs = []
            stat2srcs = []
            for matchrow in m:
                if matchrow[1]==station1:
                    s1 = matchrow[0]
                    stat1srcs.append(s1)
                elif matchrow[1]==station2:
                    s2 = matchrow[0]
                    stat2srcs.append(s2)
            # Set some convenient variables for each station source
            s1 = stat1srcs[0]
            s2 = stat2srcs[0]
            index1 = s1['index']
            fullsrclist.append((uniqueid,s1['ra'],s1['dec'],s1['nchan'],s2['ra'],s2['dec'],s2['nchan'],sourcesep))
            filename1 = str(s1['filename'])
            ra1 = s1['ra']
            dec1 = s1['dec']
            c1 = s1['channel']
            stokes1 = s1['stokes']
            plotfile1 = filename1.replace(".oims",f"-src{uniqueid}-station1-t{index1}-c{c1}-stokes{stokes1}.png").replace(".pims",f"-src{uniqueid}-station1-t{index1}-c{c1}-stokes{stokes1}.png").replace(':','-').split('/')[-1]
            index2 = s2['index']
            filename2 = str(s2['filename'])
            ra2 = s2['ra']
            dec2 = s2['dec']
            c2 = s2['channel']
            stokes2 = s2['stokes']
            plotfile2 = filename2.replace(".oims",f"-src{uniqueid}-station2-t{index2}-c{c2}-stokes{stokes2}.png").replace(".pims",f"-src{uniqueid}-station2-t{index2}-c{c2}-stokes{stokes2}.png").replace(':','-').split('/')[-1]

            if stokes1!=stokes2:
                # print("detection stokes does not match")
                continue
            # Measure fluxes, serves as another QC step. The detection must fit as a guassian
            imagefit1, diffFIT1, colnames1, coldtypes1, pixmeas1 =  sourcefitting(diffdata1,imagedata1,allflaggedtimes1,s1['ra'],s1['dec'],hdrlist1,sRad1,station1,index1,float(config1['Quality Control']['statsep']))
            if not diffFIT1:
                continue
            imagefit2, diffFIT2, colnames2,coldtypes2, pixmeas2 =  sourcefitting(diffdata2,imagedata2,allflaggedtimes2,s2['ra'],s2['dec'],hdrlist2,sRad2,station2,index2,float(config2['Quality Control']['statsep']))
            if not diffFIT2:
                continue
            imageflux1 = np.array(imagefit1,dtype=[(name,dtype) for name,dtype in zip(colnames1,coldtypes1)])
            diffFLUX1 = np.array(diffFIT1,dtype=[(name,dtype) for name,dtype in zip(colnames1,coldtypes1)])
            pixmeas1 = np.array(pixmeas1,dtype=[(name,dtype) for name,dtype in zip(colnames1,coldtypes1)])
            imageflux2 = np.array(imagefit2,dtype=[(name,dtype) for name,dtype in zip(colnames2,coldtypes2)])
            diffFLUX2 = np.array(diffFIT2,dtype=[(name,dtype) for name,dtype in zip(colnames2,coldtypes2)])
            pixmeas2 = np.array(pixmeas2,dtype=[(name,dtype) for name,dtype in zip(colnames2,coldtypes2)])
            formattedoutput = []
            for entry in m:
                row = []
                for col in entry[0]:
                    row.append(col)
                row.append(uniqueid)
                row.append(entry[1].decode())
                formattedoutput.append(tuple(row))

            srcarray =  np.array(formattedoutput,dtype=[('id','i8'),('x','f8'),('y','f8'),('ra','f8'),('dec','f8'),('mjd','f8'),('isodate','<U32'),('3c',"<U9"),('VLSSr50',"<U9"),('index','i8'),('filename',"<U256"),('channel',int),("nchan",int),("stokes",int),("uniqueid",'i4'),('station','<U8')])           
            detection1times = srcarray['mjd'][srcarray['station']==station1.decode()]
            detection2times = srcarray['mjd'][srcarray['station']==station2.decode()]

            if ~np.isin(detection1times, diffFLUX1['mjd']).any():
                continue
            if ~np.isin(detection2times, diffFLUX2['mjd']).any():
                continue


            np.savez(filename1.split('/')[-1].replace(":","-").replace(filename1[-4:],f"src{uniqueid}_fluxfile"),
                     imageflux1=imageflux1,
                     diffFLUX1=diffFLUX1,
                     minrms1=minrms1,
                     imageflux2=imageflux2,
                     diffFLUX2=diffFLUX2,
                     minrms2=minrms2,
                     srcarray=srcarray,
                     pixmeas1=pixmeas1,
                     pixmeas2=pixmeas2)

            for s1 in stat1srcs:
                for s2 in stat2srcs:
                    index1 = s1['index']
                    fullsrclist.append((uniqueid,s1['ra'],s1['dec'],s1['nchan'],s2['ra'],s2['dec'],s2['nchan'],sourcesep))
                    filename1 = str(s1['filename'])
                    ra1 = s1['ra']
                    dec1 = s1['dec']
                    c1 = s1['channel']
                    stokes1 = s1['stokes']
                    plotfile1 = filename1.replace(".oims",f"-src{uniqueid}-station1-t{index1}-c{c1}-stokes{stokes1}.png").replace(".pims",f"-src{uniqueid}-station1-t{index1}-c{c1}-stokes{stokes1}.png").replace(':','-').split('/')[-1]
                    index2 = s2['index']
                    filename2 = str(s2['filename'])
                    ra2 = s2['ra']
                    dec2 = s2['dec']
                    c2 = s2['channel']
                    stokes2 = s2['stokes']
                    plotfile2 = filename2.replace(".oims",f"-src{uniqueid}-station2-t{index2}-c{c2}-stokes{stokes2}.png").replace(".pims",f"-src{uniqueid}-station2-t{index2}-c{c2}-stokes{stokes2}.png").replace(':','-').split('/')[-1]

                 
                    # Make plots showing sources in diff images
                    plotresults(index1,c1,stokes1,plotfile1,minrms1[index1,c1,stokes1],islands1[:,c1,:,:,:],imagedata1[:,c1,:,:,:],diffdata1[:,c1,:,:,:],hdrlist1[index1],sRad1,station1)
                    plotresults(index2,c2,stokes2,plotfile2,minrms2[index2,c2,stokes2],islands2[:,c2,:,:,:],imagedata2[:,c2,:,:,:],diffdata2[:,c2,:,:,:],hdrlist2[index2],sRad2,station2)


                    if args.makefits:
                        if "pims" in filename1:
                            pimsfitsout(imagedata1, diffdata1[:,c1,:,:,:], index1, hdrlist1, station1, plotfile1.replace(".png",".fits"),c1)
                        if "pims" in filename2:
                            pimsfitsout(imagedata2, diffdata2[:,c2,:,:,:], index2, hdrlist2, station2,  plotfile2.replace(".png",".fits"), c2)
                        if "oims" in filename1:
                            oimsfitsout(imagedata1, diffdata1[:,c1,:,:,:], index1, hdrlist1, station1, plotfile1.replace(".png",".fits"), c1)
                        if "oims" in filename2:
                            oimsfitsout(imagedata2, diffdata2[:,c2,:,:,:], index2, hdrlist2, station2, plotfile2.replace(".png",".fits"), c2)
                        exportcrtf(ra1,dec1,plotfile1.replace(".png",".crtf"))
                        exportcrtf(ra2,dec2,plotfile2.replace(".png",".crtf"))
            uniqueid+=1


            




    outobsfile = f"obslist{datetime.datetime.now().timestamp()}.txt"
    with open(outobsfile,"w") as f:
        for o in fullobslist:
            f.write(f"{o[0]},{o[1]},{o[2]},{o[3]},{o[4]},{o[5]},{o[6]}\n")

    outsrcfile = f"srclist{datetime.datetime.now().timestamp()}.txt"
    with open(outsrcfile,"w") as f:
        for s in fullsrclist:
            f.write(f"{s[0]},{s[1]},{s[2]},{s[3]},{s[4]},{s[5]},{s[6]},{s[7]}\n")

                        


        
                


