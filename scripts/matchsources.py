#!/usr/bin/env python
from configparser import ConfigParser
import numpy as np
import os 
import argparse
import imdiffsearch
from GRITS import plotresults
from GRITS.configtemplate import configfile

def angsep(ra1deg,dec1deg,ra2deg,dec2deg):
    ra1 = ra1deg*np.pi/180
    dec1 = dec1deg*np.pi/180
    ra2 = ra2deg*np.pi/180
    dec2 = dec2deg*np.pi/180

    sep =  np.arccos(np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2))
    return 180*sep/np.pi


if __name__=="__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--station1', type=str, nargs='+',
                        help="images from the first station",required=True)
    parser.add_argument('--station2', type=str, nargs='+',
                        help="images from the station we check anti-concidence with",
                        required=True)
    parser.add_argument('--noplot', action="store_false")
    parser.add_argument("--vmax1", type=float, default=8000, help="controls contrast of station 1 plots")
    parser.add_argument("--vmax2", type=float, default=8000, help="controls contrast of station 2 plots")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='be verbose during the conversion')
    parser.add_argument("--timedelta",default=20,type=float, help="Time between source appearing in the two source lists")
    parser.add_argument("--angdelta",default=5,type=float, help="Maximum angular separation between a source in the two lists")
    args = parser.parse_args()
    # Read config.ini or generate new one
    config = ConfigParser()
    if not os.path.isfile("config.ini"):
        print("config.ini does not exist. Creating one and stopping.")
        with open("config.ini","w") as f:
            f.write(configfile)
    else:
        config.read("config.ini")

    # run main execution loop
    dat1 = imdiffsearch.main(args,args.station1,config)
    dat2 = imdiffsearch.main(args,args.station2,config)
    keepdat1 = dat1[(dat1['3c']=="False") & (dat1['VLSSr50']=="False")]
    keepdat2 = dat2[(dat2['3c']=="False") & (dat2['VLSSr50']=="False")]
    for s1 in keepdat1:
        seps = angsep(s1['ra'],s1['dec'],keepdat2['ra'],keepdat2['dec']) < args.angdelta
        simul = np.abs(s1['mjd'] - keepdat2['mjd'])*24.0*60.0*60.0 < args.timedelta
        if np.sum(seps & simul) > 0:
            matching2 = keepdat2[seps & simul]
            for s2 in matching2:
                print(s1,s2)


                


