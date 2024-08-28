#!/usr/bin/env python
from configparser import ConfigParser
import numpy as np

import datetime
import os 
import argparse
import imdiffsearch
from astropy.time import Time

from GRITS.configtemplate import configfile
from GRITS.plotresults import plotresults,plotsource,plotLC
from GRITS.readimage import matchfilepairs
from GRITS.imagetools import pimsfitsout, oimsfitsout, exportcrtf,pbcorr, hdrtopixmask, angsep
from GRITS.fitGaussian import fitGaussian

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
    parser.add_argument('--station1', type=str, nargs='+',
                        help="images from the first station",required=True)
    parser.add_argument('--station2', type=str, nargs='+',
                        help="images from the station we check anti-concidence with",
                        required=True)
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='be verbose during the conversion')
    parser.add_argument("--timedelta",default=5,type=float, help="Time in seconds between source appearing in the two source lists")
    parser.add_argument("--makefits",action="store_true", help="Make fits and region files")
    args = parser.parse_args()
    # Read config.ini or generate new one
    config1 = ConfigParser()
    if not os.path.isfile(args.configfile1):
        print(f"{args.configfile1} does not exist. Creating one and stopping.")
        with open(args.configfile1,"w") as f:
            f.write(configfile)
        exit()
    else:
        config1.read(args.configfile1)
    config2 = ConfigParser()
    if not os.path.isfile(args.configfile2):
        print(f"{args.configfile2} does not exist. Creating one and stopping.")
        with open(args.configfile2,"w") as f:
            f.write(configfile)
        exit()
    else:
        config2.read(args.configfile2)

    # run main execution loop
    print("Let us match the files")
    matchedfiles = matchfilepairs(args.station1,args.station2) 
    print(matchedfiles)
    lastfilename1 = ""
    lastfilename2 = ""
    results = []
    fullobslist = []
    fullsrclist = []
    uniqueid=0
    for filename1,filename2 in matchedfiles:
        if lastfilename1!=filename1:
            dat1,minrms1,corrfluxlist1,imagedata1,islands1,hdrlist1,station1,obslist1, sRad1, allflaggedtimes1 = imdiffsearch.main(filename1,config1,noplot=True)
            diffdata1 = np.copy(imagedata1)
            diffdata1[1:]  = diffdata1[1:] - diffdata1[:-1]
        if lastfilename2!=filename2:
            dat2,minrms2,corrfluxlist2,imagedata2,islands2,hdrlist2,station2, obslist2,sRad2,allflaggedtimes2 = imdiffsearch.main(filename2,config2,noplot=True)
            diffdata2 = np.copy(imagedata2)
            diffdata2[1:]  = diffdata2[1:] - diffdata2[:-1]
        if (dat1 is None) or (dat2 is None):
            continue
        lastfilename1 = filename1
        lastfilename2 = filename2
        for o1 in obslist1:
            o1sens = o1[2]
            o1t1 = datetime.datetime.strptime(o1[0],"%Y-%m-%dT%H:%M:%S.%f+00:00")
            o1t2 = datetime.datetime.strptime(o1[1],"%Y-%m-%dT%H:%M:%S.%f+00:00")
            for o2 in obslist2:
                o2sens = o2[2]
                o2t1 = datetime.datetime.strptime(o2[0],"%Y-%m-%dT%H:%M:%S.%f+00:00")
                o2t2 = datetime.datetime.strptime(o2[1],"%Y-%m-%dT%H:%M:%S.%f+00:00")
                if (np.abs((o1t1-o2t1).total_seconds()) < args.timedelta):
                    start = max(o1t1,o2t1)
                    stop = min(o1t2,o2t2)
                    outsens = max(o1sens,o2sens)
                    outfov = min(o1[-1],o2[-1])
                    fullobslist.append((start.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"),(start+datetime.timedelta(seconds=5)).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"),outsens,o1[3],o1[4],o1[5],outfov))



       #  if args.keepcatmatch:
       #  keepdat1 = dat1
       #  keepdat2 = dat2
       #  else:
        keepdat1 = dat1[(dat1['3c']=="False") & (dat1['VLSSr50']=="False")]
        keepdat2 = dat2[(dat2['3c']=="False") & (dat2['VLSSr50']=="False")]
        plottedlc = []
        for s1 in keepdat1:
            simul = np.abs(s1['mjd'] - keepdat2['mjd'])*24.0*60.0*60.0 < args.timedelta
            if np.sum(simul) > 0:
                matching2 = keepdat2[simul]
                for s2 in matching2:
                    print(s1,s2)
                    sourcesep =  angsep(s1['ra'],s1['dec'],s2['ra'],s2['dec'])
                    if sourcesep > float(config1['Quality Control']['catsep']):
                        if (s1['nchan'] < 2) and (s2['nchan'] < 2):
                            print("not multi channel")
                            continue
                    fullsrclist.append((uniqueid,s1['ra'],s1['dec'],s1['nchan'],s2['ra'],s2['dec'],s2['nchan'],sourcesep))
                    index1 = s1['index']
                    filename1 = str(s1['filename'])
                    ra1 = s1['ra']
                    dec1 = s1['dec']
                    c1 = s1['channel']
                    plotfile1 = filename1.replace(".oims",f"-src{uniqueid}-station1-t{index1}-c{c1}.png").replace(".pims",f"-src{uniqueid}-station1-t{index1}-c{c1}.png").replace(':','-').split('/')[-1]
                    index2 = s2['index']
                    filename2 = str(s2['filename'])
                    ra2 = s2['ra']
                    dec2 = s2['dec']
                    c2 = s2['channel']
                    plotfile2 = filename2.replace(".oims",f"-src{uniqueid}-station2-t{index2}-c{c2}.png").replace(".pims",f"-src{uniqueid}-station2-t{index2}-c{c2}.png").replace(':','-').split('/')[-1]
                    print(index1,index2)
                    plotresults(index1,c1,plotfile1,minrms1[index1,c1],islands1[:,c1,:,:,:],imagedata1[:,c1,:,:,:],diffdata1[:,c1,:,:,:],hdrlist1[index1],sRad1,station1)
                    plotresults(index2,c2,plotfile2,minrms2[index2,c2],islands2[:,c2,:,:,:],imagedata2[:,c2,:,:,:],diffdata2[:,c2,:,:,:],hdrlist2[index2],sRad2,station2)
                    # plotsource(index1, plotfile1, filename1, ra1, dec1, minrms1, corrfluxlist1, c1, vmax=args.vmax1)
                    # plotsource(index2, plotfile2, filename2, ra2, dec2, minrms2, corrfluxlist2, c2, vmax=args.vmax2)
                     
                    for filename, rmsdata, imagedata, diffdata, s, hdrlist, sRad,station,allflaggedtimes, kd in zip([filename1,filename2], [minrms1,minrms2], [imagedata1,imagedata2],[diffdata1,diffdata2],[s1,s2],[hdrlist1,hdrlist2],[sRad1,sRad2],[station1,station2],[allflaggedtimes1,allflaggedtimes2],[keepdat1,keepdat2]):        
                            # If within the specified localization, we will skip it when we get to it in the primary loop
                        print(filename,station)
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
                                dettimechan.append((keeprow['index'],keeprow['channel']))
                            
                        dettimechan = np.array(dettimechan, dtype=[('index','i8'),('c','u2')])
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
                                pbcorrallimage= (imagedata[timeind,c,0]/beamI[c]) - (imagedata[timeind-1,c,0]/beamI[c])
                                if hasDETECT:
                                    flux = pbcorrallimage[int(y),int(x)]
                                else:
                                    flux = np.nanmax(np.where(((xcoord-x)**2 + (ycoord-y)**2) < prad**2,pbcorrallimage,np.nan))
                                midfreq = round((hdrlist[timeind]['start_freq']  + (c*hdrlist[timeind]['bandwidth']) + (hdrlist[timeind]['bandwidth']/2))*1e-6,2)
                                flagged = (timeind,c) in allflaggedtimes
                                mjd = hdrlist[timeind]['start_time']
                                hourspastmid = (mjd - np.floor(mjd))*24
                                plotdata.append((timeind,flux,c,midfreq,flagged,hourspastmid))

                        plotdata = np.array(plotdata, dtype=[('index','i8'),('flux','f8'),('c','u2'),('freq','f4'),('flagged',bool),('hrpst','f8')])
                        zoomplotdata = np.copy(plotdata[(plotdata['index']>mintime) & (plotdata['index'] < maxtime)])
                        tdet = Time(hdrlist[udet]['start_time'],format='mjd',scale='utc')
                        dettimechan = np.array(dettimechan, dtype=[('index','i8'),('c','u2')])
                        plotLC(filename, plotdata, zoomplotdata,curdettimechan,uniqueid,str(udet),tdet.isot,ra,dec)
                        plottedlc.append((ra,dec,station,udet))
                    
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

                        


        
                


