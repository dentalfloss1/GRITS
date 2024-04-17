#!/usr/bin/env python
from configparser import ConfigParser
import numpy as np

import datetime
import os 
import argparse
import imdiffsearch
from GRITS.configtemplate import configfile
# from GRITS.plotresults import plotsource
from GRITS.readimage import matchfilepairs

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
    parser.add_argument("--timedelta",default=20,type=float, help="Time in seconds between source appearing in the two source lists")
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
    for filename1,filename2 in matchedfiles:
        if lastfilename1!=filename1:
            dat1,_,minrms1,corrfluxlist1,keepchan1 = imdiffsearch.main(filename1,config,vmax=args.vmax1,noplot=True)
        if lastfilename2!=filename2:
            dat2,_,minrms2,corrfluxlist2,keepchan2 = imdiffsearch.main(filename2,config,vmax=args.vmax2,noplot=True)
        lastfilename1 = filename1
        lastfilename2 = filename2
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
                    results.append((s1,minrms1,corrfluxlist1,keepchan1,s2,minrms2,corrfluxlist2,keepchan2))
    t2 = datetime.datetime.now()
    print(t2-t1, "elapsed")
    print("--------results-------")
    for uniqueid, (s1,minrms1,corrfluxlist1,keepchan1,s2,minrms2,corrfluxlist2,keepchan2) in enumerate(results):
     #    index1 = s1['index']
     #    filename1 = f"src{uniqueid}-{s1['filename']}"
     #    ra1 = s1['ra']
     #    dec1 = s1['dec']
     #    plotfile1 = f"src{uniqueid}-"+filename1.replace(".oims",f"-{index}.png").replace(".pims",f"-{index}.png").replace(':','-').split('/')[-1]
     #    plotsource(index1,plotfile1,filename1,ra1,dec1,minrms1,corrfluxlist1,keepchan1,vmax=args.vmax1)
     #    index2 = s2['index']
     #    filename2 = f"src{uniqueid}-{s2['filename']}"
     #    ra2 = s2['ra']
     #    dec2 = s2['dec']
     #    plotfile2 = f"src{uniqueid}-"+filename1.replace(".oims",f"-{index}.png").replace(".pims",f"-{index}.png").replace(':','-').split('/')[-1]
     #    plotsource(index2,plotfile2,filename2,ra2,dec2,minrms2,corrfluxlist2,keepchan2,vmax=args.vmax2)

        print(r)
        
                


