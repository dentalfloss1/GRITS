import numpy as np
import argparse

def angsep(ra1deg,dec1deg,ra2deg,dec2deg):
    ra1 = ra1deg*np.pi/180
    dec1 = dec1deg*np.pi/180
    ra2 = ra2deg*np.pi/180
    dec2 = dec2deg*np.pi/180

    sep =  np.arccos(np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2))
    return 180*sep/np.pi
parser = argparse.ArgumentParser()
parser.add_argument("srcfile1",type=str, help="CSV file of sources from GRITS")
parser.add_argument("srcfile2",type=str, help="CSV file of sources from GRITS")
parser.add_argument("--timedelta",default=20, help="Time between source appearing in the two source lists")
parser.add_argument("--angdelta",default=5, help="Maximum angular separation between a source in the two lists")
args = parser.parse_args()


if __name__=="__main__":
    dat1 = np.loadtxt(args.srcfile1, delimiter=',', skiprows=1, dtype=[('id','i8'),('ra','f8'),('dec','f8'),('mjd','f8'),('isodate','<U32'),('3c',bool),('VLSSr50',bool)])
    dat2 = np.loadtxt(args.srcfile2, delimiter=',', skiprows=1, dtype=[('id','i8'),('ra','f8'),('dec','f8'),('mjd','f8'),('isodate','<U32'),('3c',bool),('VLSSr50',bool)])
    keepdat1 = dat1[(~dat1['3c']) & (~dat1['VLSSr50'])]
    keepdat2 = dat2[(~dat2['3c']) & (~dat2['VLSSr50'])]
    for s1 in keepdat1:
        seps = angsep(s1['ra'],s1['dec'],keepdat2['ra'],keepdat2['dec'])
        simul = np.abs(s1['mjd'] - keepdat2['mjd'])*24.0*60.0*60.0
        if np.sum(seps & simul) > 0:
            print(s1,keepdat2[seps & simul])

