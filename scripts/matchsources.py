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
from GRITS.imagetools import pimsfitsout, oimsfitsout, exportcrtf,pbcorr, hdrtopixmask, angsep
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
        print("Let us match the files")
        import glob
        # station1 = glob.glob("/FileStore/Orville/*/*.oims")
        # station2 = glob.glob("/FileStore/PASI/*/*.pims")
       #  if os.path.exists("matchedfiles.npy"):
       #      matchedfiles= np.load("matchedfiles.npy")
       #  else:
        matchedfiles = matchfilepairs(args.station1,args.station2) 
        #     matchedfiles = matchfilepairs(station1,station2)
        #     print(matchedfiles)
        #     np.save("matchedfiles.npy",matchedfiles)
    lastfilename1 = ""
    lastfilename2 = ""
    results = []
    fullobslist = []
    fullsrclist = []
    uniqueid=0
    for filename1,filename2 in tqdm(matchedfiles):
        obsfile1= filename1.split('/')[-1]+"_obsfile.txt"
        obsfile2= filename2.split('/')[-1]+"_obsfile.txt"
        if args.resume:
            if os.path.exists(obsfile1+'.npy') and os.path.exists(obsfile2+'.npy'):
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
        obsfile1= filename1.split('/')[-1]+"_obsfile.txt"
        obsfile2= filename2.split('/')[-1]+"_obsfile.txt"
        np.save(obsfile1,obslist1)
        np.save(obsfile2,obslist2)



        keepdat1 = dat1[(dat1['3c']=="False") & (dat1['VLSSr50']=="False")]
        keepdat2 = dat2[(dat2['3c']=="False") & (dat2['VLSSr50']=="False")]
        plottedlc = []
        srcfile1= filename1.split('/')[-1]+"_srcfile.txt"
        srcfile2= filename2.split('/')[-1]+"_srcfile.txt"
        srclist1 = []
        srclist2 = []
        for s1 in keepdat1:
            simul = np.abs(s1['mjd'] - keepdat2['mjd'])*24.0*60.0*60.0 < args.timedelta
            if np.sum(simul) > 0:
                matching2 = keepdat2[simul]
                for s2 in matching2:
                    if args.verbose:
                        print(s1,s2)
                    sourcesep =  angsep(s1['ra'],s1['dec'],s2['ra'],s2['dec'])

                    if sourcesep > float(config1['Quality Control']['catsep']):
                        
                       #  if (s1['nchan'] < 2) and (s2['nchan'] < 2):
                       #      if args.verbose:
                       #          print("not multi channel")
                        continue
                    index1 = s1['index']
                    uniqueid = index1
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
                    if stokes1!=stokes2:
                        # print("detection stokes does not match")
                        continue
                    plotfile2 = filename2.replace(".oims",f"-src{uniqueid}-station2-t{index2}-c{c2}-stokes{stokes2}.png").replace(".pims",f"-src{uniqueid}-station2-t{index2}-c{c2}-stokes{stokes2}.png").replace(':','-').split('/')[-1]
                    plotresults(index1,c1,stokes1,plotfile1,minrms1[index1,c1,stokes1],islands1[:,c1,:,:,:],imagedata1[:,c1,:,:,:],diffdata1[:,c1,:,:,:],hdrlist1[index1],sRad1,station1)
                    plotresults(index2,c2,stokes2,plotfile2,minrms2[index2,c2,stokes2],islands2[:,c2,:,:,:],imagedata2[:,c2,:,:,:],diffdata2[:,c2,:,:,:],hdrlist2[index2],sRad2,station2)
                    # plotsource(index1, plotfile1, filename1, ra1, dec1, minrms1, corrfluxlist1, c1, vmax=args.vmax1)
                    # plotsource(index2, plotfile2, filename2, ra2, dec2, minrms2, corrfluxlist2, c2, vmax=args.vmax2)
                    plotfilelist = []
                    plotdatalist = []
                    curdettimechanlist = []
                    udetlist = []
                    tdetlist = []
                    ralist = []
                    declist = []
                    srclist1.append(s1)
                    srclist2.append(s2)
                    for filename, rmsdata, imagedata, diffdata, s, hdrlist, sRad,station,allflaggedtimes, kd, detstokes in zip([filename1,filename2], [minrms1,minrms2], [imagedata1,imagedata2],[diffdata1,diffdata2],[s1,s2],[hdrlist1,hdrlist2],[sRad1,sRad2],[station1,station2],[allflaggedtimes1,allflaggedtimes2],[keepdat1,keepdat2],[stokes1,stokes2]):        
                            # If within the specified localization, we will skip it when we get to it in the primary loop
                        # print(filename,station)
                        if filename==filename1:
                            plotfile = filename.replace(".oims",f"-stokes{detstokes}-station1.oims").replace('.pims',f'-stokes{detstokes}-station1.pims')
                        else:
                            plotfile = filename.replace(".oims",f"-stokes{detstokes}-station2.oims").replace('.pims',f'-stokes{detstokes}-station2.pims')
                        
                            
                        ngrid = imagedata.shape[-1]
                        mintime = max(0,s['index']-12)
                        maxtime = min(len(hdrlist)-1, s['index']+12)
                        plotdata = []
                        zoomplotdata = []
                        ra = s['ra']
                        dec = s['dec']
                        psize = hdrlist[0]['psize']
                        udet = s['index']
                        dettimechan = []
                        xcoord = np.arange(ngrid)
                        ycoord = np.arange(ngrid)
                        xcoord,ycoord = np.meshgrid(xcoord, ycoord)
                        for keeprow in kd:
                            if angsep(ra,dec,keeprow['ra'],keeprow['dec']) < float(config1['Quality Control']['statsep']):
                                dettimechan.append((keeprow['index'],keeprow['channel'],keeprow['ra'],keeprow['dec']))
                            
                        dettimechan = np.array(dettimechan, dtype=[('index','i8'),('c','u2'),('ra','f8'),('dec','f8')])
                        curdettimechan = dettimechan[dettimechan['index']==udet]
                        for pra,pdec,pstation,pindex in plottedlc:
                            if (pstation==station) and (udet==pindex):
                                if angsep(pra,pdec,ra,dec) < float(config1['Quality Control']['statsep']):
                                    continue
                        prad = float(config1['Quality Control']['statsep'])/psize
                        beamI = []
                        for c in range(imagedata.shape[1]):
                            XX,YY = pbcorr(hdrlist[0],ngrid,c,psize,station)
                            beamI.append((XX+YY)/2)
                        # print(imagedata.shape, len(hdrlist))

                        for timeind in range(1,len(hdrlist)):
                            if timeind in dettimechan['index']:
                                measRA = dettimechan['ra'][dettimechan['index']==timeind][0]
                                measDec = dettimechan['dec'][dettimechan['index']==timeind][0]
                            else:
                                measRA = ra
                                measDec = dec
                            x,y = hdrtopixmask(hdrlist[timeind], ngrid, sRad, measRA,measDec, station)[0]
                            for c in range(imagedata.shape[1]):
                                if (timeind in dettimechan['index']) and (c in dettimechan['c']):
                                    hasDETECT = True
                                else:
                                    hasDETECT = False
                                if detstokes==0:
                                    corbeam = beamI[c]
                                    pbcorrallimage= (imagedata[timeind,c,detstokes]/corbeam) - (imagedata[timeind-1,c,detstokes]/corbeam)
                                else:
                                    corbeam = 1
                                    pbcorrallimage= (imagedata[timeind,c,detstokes]/corbeam) - (imagedata[timeind-1,c,detstokes]/corbeam)
                                if hasDETECT:
                                    flux = pbcorrallimage[int(y),int(x)]
                                else:
                                    try:
                                        flux = np.nanmean(np.where(((xcoord-x)**2 + (ycoord-y)**2) < prad**2,pbcorrallimage,np.nan))
                                    except RuntimeWarning:
                                        flux = 0
                                midfreq = round((hdrlist[timeind]['start_freq']  + (c*hdrlist[timeind]['bandwidth']) + (hdrlist[timeind]['bandwidth']/2))*1e-6,2)
                                flagged = (timeind,c) in allflaggedtimes
                                mjd = hdrlist[timeind]['start_time']
                                hourspastmid = (mjd - np.floor(mjd))*24
                                plotdata.append((timeind,flux,c,midfreq,flagged,False,hourspastmid))

                        plotdata = np.array(plotdata, dtype=[('index','i8'),('flux','f8'),('c','u2'),('freq','f4'),('flagged',bool),('statmatch',bool),('hrpst','f8')])
                        zoomplotdata = np.copy(plotdata[(plotdata['index']>mintime) & (plotdata['index'] < maxtime)])
                        tdet = Time(hdrlist[udet]['start_time'],format='mjd',scale='utc')
                        plotfilelist.append(plotfile)
                        plotdatalist.append(plotdata)
                        curdettimechanlist.append(curdettimechan)
                        udetlist.append(str(udet))
                        tdetlist.append(tdet.isot)
                        ralist.append(ra)
                        declist.append(dec)
                        plottedlc.append((ra,dec,station,udet))
                    completed1 = []
                    completed2 = []
                    matchedsrc = []
                    for hrstat1 in plotdatalist[0][plotdatalist[0]['flagged']==False]['hrpst']:
                        if hrstat1 in completed1:
                            continue
                        for hrstat2 in plotdatalist[1][plotdatalist[1]['flagged']==False]['hrpst']:
                            if hrstat2 in completed2:
                                continue
                            if np.abs(hrstat2-hrstat1)*3600 < args.timedelta:
                                matchtime1 = plotdatalist[0]['hrpst']==hrstat1
                                matchtime2 = plotdatalist[1]['hrpst']==hrstat2
                                simulplot1 = plotdatalist[0][matchtime1]
                                simulplot2 = plotdatalist[1][matchtime2]
                                for freq1 in np.unique(simulplot1['freq']):
                                    for freq2 in np.unique(simulplot2['freq']):
                                        if round(freq1,0)==round(freq2,0):
                                            curflux1 = simulplot1[simulplot1['freq']==freq1]['flux']
                                            curflux2 = simulplot2[simulplot2['freq']==freq2]['flux']
                                            for f1  in curflux1:
                                                for f2 in curflux2:
                                                    if isclose(f1,f2,rel_tol=0.2):
                                                        matchedsrc.append((f1,f2,freq1,freq2))
                                                        plotdatalist[0][plotdatalist[0]['hrpst']==hrstat1]['flagged'] = False
                                                        plotdatalist[1][plotdatalist[1]['hrpst']==hrstat2]['flagged'] = False
                                                        completed1.append(hrstat1)
                                                        completed2.append(hrstat2)

                    for plotfile,plotdata,curdettimechan,udet,tdet,ra,dec,s,hdrlist in zip(plotfilelist,plotdatalist, curdettimechanlist,udetlist,tdetlist,ralist,declist,[s1,s2],[hdrlist1,hdrlist2]):
                        mintime = max(0,s['index']-12)
                        maxtime = min(len(hdrlist)-1, s['index']+12)
                        for f1,f2,freq1,freq2 in matchedsrc:
                            plotdata['statmatch'][((plotdata['flux']==f1) & (plotdata['freq']==freq1)) | ((plotdata['flux']==f2) & (plotdata['freq']==freq2))] = True
                        zoomplotdata = np.copy(plotdata[(plotdata['index']>mintime) & (plotdata['index'] < maxtime)])
                        plotLC(plotfile, plotdata, zoomplotdata,curdettimechan,uniqueid,udet,tdet,ra,dec)
                                            
                                    


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
        if srclist1:
            np.save(srcfile1,srclist1)
            np.save(srcfile2,srclist2)



    outobsfile = f"obslist{datetime.datetime.now().timestamp()}.txt"
    with open(outobsfile,"w") as f:
        for o in fullobslist:
            f.write(f"{o[0]},{o[1]},{o[2]},{o[3]},{o[4]},{o[5]},{o[6]}\n")

    outsrcfile = f"srclist{datetime.datetime.now().timestamp()}.txt"
    with open(outsrcfile,"w") as f:
        for s in fullsrclist:
            f.write(f"{s[0]},{s[1]},{s[2]},{s[3]},{s[4]},{s[5]},{s[6]},{s[7]}\n")

                        


        
                


