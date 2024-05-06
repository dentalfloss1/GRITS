#!/usr/bin/env python

import numpy as np 
import csv

import os

import fnmatch
import paramiko
 
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from lsl.common.stations import lwasv as lsllwasv
from lsl.common.stations import lwana as lsllwana
from lsl.common.stations import lwa1 as lsllwa1
from lsl.common.mcs import mjdmpm_to_datetime

import argparse

def getfiles(basepath, dates, hrmargin=1):
    stationfilelist= []
    for mymjd, myhr in dates:
        try:
            for p in [s.filename for s in sftp.listdir_attr(f'{basepath}{mymjd}/')]:
                myhr = str(myhr)
                print(f"{mymjd}:{myhr.zfill(2)}_*MHz*pims")
                if fnmatch.fnmatch(p, f"{mymjd}:{myhr.zfill(2)}_*MHz*pims"):
                    filepath = f'{basepath}{mymjd}/{p}'
                    info = sftp.stat(filepath)
                    fullfilelist.append((date,ra,dec,station,filepath,info.st_size))
                if fnmatch.fnmatch(p, f"{mymjd}_{myhr.zfill(2)}0000_*oims"):
                    filepath = f'{basepath}{mymjd}/{p}'
                    info = sftp.stat(filepath)
                    stationfilelist.append((date,ra,dec,station,filepath,info.st_size))
        except FileNotFoundError:
            pass
    return stationfilelist
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("datafile", type=str, help="CSV data file of format: MJD,RA (deg), Dec (deg). It is assumed that the file has no header.")
    parser.add_argument("--hrmargin",type=int, default=1,help="List files that are +/- hrmargin")
    parser.add_argument("--alt",type=float, default=30,help="Minimum alt")
    args = parser.parse_args()
    hrmargin = args.hrmargin
    host = os.environ["SSHHOST"]
    user = os.environ["SSHUSER"]
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy( paramiko.AutoAddPolicy() )
    ssh.connect(host, username=user)
    sftp = ssh.open_sftp()

    searchdata = []
    with open(args.datafile,newline='') as csvfile:
        filereader = csv.reader(csvfile,delimiter=',')
        for date,ra,dec in filereader:
            searchdata.append((float(date),float(ra),float(dec)))

    visiblesrcs = []
    for date,ra,dec in searchdata:
        mjd = int(date)
        mpm = int((date - mjd)*86400.0*1000.0)
        dateObs = mjdmpm_to_datetime(mjd, mpm)
        time = Time(dateObs.strftime("%Y-%m-%dT%H:%M:%S"),format="isot")
        for name,station in [('lwa1',lsllwa1),('lwasv',lsllwasv),('lwana',lsllwana)]: 
            x,y,z = station.geocentric_location
            stationloc = EarthLocation.from_geocentric(x=x,y=y,z=z,unit='m')
            aa = AltAz(location=stationloc, obstime=time)
            sc = SkyCoord(ra,dec,unit='deg')
            myaltaz = sc.transform_to(aa)
            alt = myaltaz.alt.deg
            az = myaltaz.az.deg
            if alt > (args.alt):
                visiblesrcs.append((date,ra,dec,name))
    fullfilelist = [] 
    for date,ra,dec,station in visiblesrcs:
        obsday = int(round(date,0))
        obshr = int(round((date - obsday)*24,0))
        dates = []
        for mymjd in range(obsday-1, obsday+2):
            for myhr in range(obshr-hrmargin, obshr+hrmargin+1):
                if mymjd!=obsday:
                    if (mymjd==(obsday-1)) and (myhr<0):
                        dates.append((mymjd, myhr % 24))
                    if (mymjd==(obsday+1)) and (myhr>23):
                        dates.append((mymjd, myhr % 24))
                if mymjd==obsday:
                    if (myhr < 24) and (myhr >= 0):
                        dates.append((mymjd, myhr ))
        print(dates)
        if station=='lwa1':
            basepath= ['/FileStore/PASI/']
        elif station=='lwasv':
            basepath= ['/FileStore/PASI-SV/','/FileStore/Orville/']
        elif station=='lwana':
            basepath= ['/FileStore/Orville-NA/']
        for b in basepath:
            results = getfiles(b, dates, hrmargin=args.hrmargin)
            if results:
                fullfilelist.extend(results)
            

            
    for f in fullfilelist:
        print(f) 
    
    pulldat = np.unique(np.array(fullfilelist, dtype=[('date','f8'),('ra','f8'),('dec','f8'),('station','<U10'),('filepath','<U256'),('size','i8')]))
    for p in pulldat:
        print(p)
    for p in pulldat:
        print(p['filepath'])
    print(np.sum(pulldat['size']/1e9))

