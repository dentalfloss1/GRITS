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
    parser.add_argument('files', type=str, nargs='+',
                        help="files to search",required=True)
    parser.add_argument("--vmax", type=float, default=8000, help="controls contrast of station 1 plots")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='be verbose during the conversion')
    parser.add_argument("--angdelta",default=5,type=float, help="Maximum angular separation between a found source and the supplied source")
    parser.add_argument("--targets",type=str, 
                        help="CSV data file with sources to search for. Has format: MJD,RA (deg), Dec (deg). It is assumed that the file has no header.",required=True)
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
    print(matchedfiles)
    lastfilename1 = ""
    lastfilename2 = ""
    results = []
    searchdata = np.loadtxt(args.targets, delimiter=',', dtype=[('date','f8'),('ra','f8'),('dec','f8')])
    for filename in args.files:
        dat,_,minrms,corrfluxlist,keepchan = imdiffsearch.main(filename,config,vmax=args.vmax,noplot=False)
        if args.keepcatmatch:
            keepdat = dat
        else:
            keepdat = dat[(dat['3c']=="False") & (dat['VLSSr50']=="False")]
        for s in keepdat:
            seps = angsep(s['ra'],s['dec'],searchdata['ra'],searchdata['dec']) < args.angdelta
            
            if seps.any(): 
                matching = keepdat[seps]
                for s in matching2:
                    print(s)
                    results.append((np.copy(s),minrms,corrfluxlist,keepchan))
    t2 = datetime.datetime.now()
    print(t2-t1, "elapsed")
    print("--------results-------")
    for uniqueid,r in enumerate(results):
        (s,minrms,corrfluxlist,keepchan) = r
        index = s['index']
        filename = str(s['filename'])
        ra = s['ra']
        dec = s['dec']
        plotfile = f"src{uniqueid}-"+filename.replace(".oims",f"-{index}.png").replace(".pims",f"-{index}.png").replace(':','-').split('/')[-1]
        plotsource(index,plotfile,filename,ra,dec,minrms,corrfluxlist,keepchan,vmax=args.vmax)
        print(r)
        print((s['mjd'] - searchdata['date'])*24*60*60)

        
                


