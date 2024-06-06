#!/usr/bin/env python
from configparser import ConfigParser
import numpy as np

import datetime
import os 
import argparse
import imdiffsearch
from GRITS.configtemplate import configfile
from GRITS.plotresults import plotsource
from GRITS.readimage import matchfilepairs
from GRITS.imagetools import pimsfitsout, oimsfitsout, exportcrtf

def angsep(ra1deg,dec1deg,ra2deg,dec2deg):
    ra1 = ra1deg*np.pi/180
    dec1 = dec1deg*np.pi/180
    ra2 = ra2deg*np.pi/180
    dec2 = dec2deg*np.pi/180

    sep =  np.arccos(np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2))
    return 180*sep/np.pi


if __name__=="__main__":
    t1 = datetime.datetime.now()
    # Parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--station1', type=str, nargs='+',
                        help="images from the first station",required=True)
    parser.add_argument('--station2', type=str, nargs='+',
                        help="images from the station we check anti-concidence with",
                        required=True)
    parser.add_argument("--vmax1", type=float, default=8000, help="controls contrast of station 1 plots")
    parser.add_argument("--vmax2", type=float, default=8000, help="controls contrast of station 2 plots")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='be verbose during the conversion')
    parser.add_argument("--timedelta",default=5,type=float, help="Time in seconds between source appearing in the two source lists")
    parser.add_argument("--angdelta",default=5,type=float, help="Maximum angular separation between a source in the two lists in degrees")
    parser.add_argument("--keepcatmatch",action="store_true", help="Keep sources with catalog matches")
    args = parser.parse_args()
    # Read config.ini or generate new one
    config = ConfigParser()
    if not os.path.isfile("config.ini"):
        print("config.ini does not exist. Creating one and stopping.")
        with open("config.ini","w") as f:
            f.write(configfile)
        exit()
    else:
        config.read("config.ini")

    # run main execution loop
    matchedfiles = matchfilepairs(args.station1,args.station2) 
    print(matchedfiles)
    lastfilename1 = ""
    lastfilename2 = ""
    results = []
    fullobslist = []
    uniqueid=0
    for filename1,filename2 in matchedfiles:
        if lastfilename1!=filename1:
            dat1,minrms1,corrfluxlist1,imagedata1,hdrlist1,station1,obslist1 = imdiffsearch.main(filename1,config,vmax=args.vmax1,noplot=True)
            diffdata1 = np.copy(imagedata1)
            diffdata1[:-1]  = diffdata1[1:] - diffdata1[:-1]
        if lastfilename2!=filename2:
            dat2,minrms2,corrfluxlist2,imagedata2,hdrlist2,station2, obslist2 = imdiffsearch.main(filename2,config,vmax=args.vmax2,noplot=True)
            diffdata2 = np.copy(imagedata2)
            diffdata2[:-1]  = diffdata2[1:] - diffdata2[:-1]
        if (dat1 is None) or (dat2 is None):
            continue
        lastfilename1 = filename1
        lastfilename2 = filename2
        for o1 in obslist1:
            o1sens = o1[2]
            o1t1 = datetime.datetime.strptime(o1[0],"%Y-%m-%dT%H:%M:%S.%f+00:00")
            o1t2 = datetime.datetime.strptime(o1[0],"%Y-%m-%dT%H:%M:%S.%f+00:00")
            for o2 in obslist2:
                o2sens = o2[2]
                o2t1 = datetime.datetime.strptime(o2[0],"%Y-%m-%dT%H:%M:%S.%f+00:00")
                o2t2 = datetime.datetime.strptime(o2[0],"%Y-%m-%dT%H:%M:%S.%f+00:00")
                if (np.abs((o1t-o2t).total_seconds()) < args.timedelta):
                    start = max(o1t1,o2t1)
                    stop = min(o1t2,o2t2)
                    outsens = max(o1sens,o2sens)
                    outfov = min(o1[-1],o2[-1])
                    fullobslist.append((start.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"),stop.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"),outsens,o1[3],o1[4],o1[5],outfov))



        if args.keepcatmatch:
            keepdat1 = dat1
            keepdat2 = dat2
        else:
            keepdat1 = dat1[(dat1['3c']=="False") & (dat1['VLSSr50']=="False")]
            keepdat2 = dat2[(dat2['3c']=="False") & (dat2['VLSSr50']=="False")]
        for s1 in keepdat1:
            seps = angsep(s1['ra'],s1['dec'],keepdat2['ra'],keepdat2['dec']) < args.angdelta
            simul = np.abs(s1['mjd'] - keepdat2['mjd'])*24.0*60.0*60.0 < args.timedelta
            if np.sum(seps & simul) > 0:
                matching2 = keepdat2[seps & simul]
                for s2 in matching2:
                    print(s1,s2)
                    index1 = s1['index']
                    filename1 = str(s1['filename'])
                    ra1 = s1['ra']
                    dec1 = s1['dec']
                    plotfile1 = f"src{uniqueid}-station1-"+filename1.replace(".oims",f"-{index1}.png").replace(".pims",f"-{index1}.png").replace(':','-').split('/')[-1]
                    index2 = s2['index']
                    filename2 = str(s2['filename'])
                    ra2 = s2['ra']
                    dec2 = s2['dec']
                    plotfile2 = f"src{uniqueid}-station2-"+filename2.replace(".oims",f"-{index2}.png").replace(".pims",f"-{index2}.png").replace(':','-').split('/')[-1]
                    plotsource(index1,plotfile1,filename1,ra1,dec1,minrms1,corrfluxlist1,vmax=args.vmax1)
                    plotsource(index2,plotfile2,filename2,ra2,dec2,minrms2,corrfluxlist2,vmax=args.vmax2)
                    if "pims" in filename1:
                        pimsfitsout(imagedata1, diffdata1, index1, hdrlist1, station1, plotfile1.replace(".png",".fits"))
                    if "pims" in filename2:
                        pimsfitsout(imagedata2, diffdata2, index2, hdrlist2, station2, plotfile2.replace(".png",".fits"))
                    if "oims" in filename1:
                        oimsfitsout(imagedata1, diffdata1, index1, hdrlist1, station1, plotfile1.replace(".png",".fits"))
                    if "oims" in filename2:
                        oimsfitsout(imagedata2, diffdata2, index2, hdrlist2, station2, plotfile2.replace(".png",".fits"))
                    exportcrtf(ra1,dec1,f"src{uniqueid}-region-station1.crtf")
                    exportcrtf(ra2,dec2,f"src{uniqueid}-region-station2.crtf")
                    uniqueid+=1


    outobsfile = f"obslist{datetime.datetime.now().timestamp()}.txt"
    with open(outobsfile,"w") as f:
        for o in fullobslist:
            f.write(f"{o[0]},{o[1]},{o[2]},{o[3]},{o[4]},{o[5]},{o[6]}\n")

                        


        
                


