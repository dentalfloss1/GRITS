#!/usr/bin/env python

import numpy as np 

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

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", type=str, help="CSV data file of format: MJD,RA (deg), Dec (deg). It is assumed that the file has no header.")
    args = parser.parse_args()
    host = os.environ["SSHHOST"]
    user = os.environ["SSHUSER"]
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy( paramiko.AutoAddPolicy() )
    ssh.connect(host, username=user)
    sftp = ssh.open_sftp()

    searchdata = np.loadtxt(args.datafile, delimiter=',', dtype=[('date','f8'),('ra','f8'),('dec','f8')])

    visiblesrcs = []
    for s in searchdata:
        mjd = int(s['date'])
        mpm = int((s['date'] - mjd)*86400.0*1000.0)
        dateObs = mjdmpm_to_datetime(mjd, mpm)
        time = Time(dateObs.strftime("%Y-%m-%dT%H:%M:%S"),format="isot")
        for name,station in [('lwa1',lsllwa1),('lwasv',lsllwasv),('lwana',lsllwana)]: 
            x,y,z = station.geocentric_location
            stationloc = EarthLocation.from_geocentric(x=x,y=y,z=z,unit='m')
            aa = AltAz(location=stationloc, obstime=time)
            sc = SkyCoord(s['ra'],s['dec'],unit='deg')
            myaltaz = sc.transform_to(aa)
            alt = myaltaz.alt.deg
            az = myaltaz.az.deg
            if alt > (90-60):
                visiblesrcs.append((s['date'],s['ra'],s['dec'],name))
    fullfilelist = [] 
    for date,ra,dec,station in visiblesrcs:
        obsday = int(round(date,0))
        obshr = int(round((date - obsday)*24,0))
        if station=='lwa1':
            for mymjd in [obsday-1,obsday,obsday+1]:
                if mymjd!=obsday:
                    if ~((obshr - 2) < 0) or ~((obshr + 2) > 23):
                        continue
                try:
                    for p in [s.filename for s in sftp.listdir_attr(f'/FileStore/PASI/{mymjd}/')]:
                        for myhr in (str(ohr) for ohr in range(obshr-2,obshr+3)):
                            if fnmatch.fnmatch(p, f"{mymjd}*{myhr.zfill(2)}_*MHz*pims"):
                                filepath = f'/FileStore/PASI/{mymjd}/{p}'
                                fullfilelist.append((date,ra,dec,station,filepath))
                except FileNotFoundError:
                    pass

        if station=='lwasv':
            for mymjd in [obsday-1,obsday,obsday+1]:
                if mymjd!=obsday:
                    if ~((obshr - 2) < 0) or ~((obshr + 2) > 23):
                        continue
                try:
                    for p in [s.filename for s in sftp.listdir_attr(f'/FileStore/PASI-SV/{mymjd}/')]:
                        for myhr in (str(ohr) for ohr in range(obshr-2,obshr+3)):
                            if fnmatch.fnmatch(p, f"{mymjd}:{myhr.zfill(2)}_*MHz*pims"):
                                filepath = f'/FileStore/PASI-SV/{mymjd}/{p}'
                                fullfilelist.append((date,ra,dec,station,filepath))
                except FileNotFoundError:
                    pass
                try:
                    for o in [s.filename for s in sftp.listdir_attr(f'/FileStore/Orville/{mymjd}/')]:
                        for myhr in (str(ohr) for ohr in range(obshr-2,obshr+3)):
                            if fnmatch.fnmatch(o, f"{mymjd}_{myhr.zfill(2)}0000_*oims"):
                                filepath = f'/FileStore/Orville/{mymjd}/{o}'
                                fullfilelist.append((date,ra,dec,station,filepath))
                except FileNotFoundError:
                    pass
        if station=='lwana':
            for mymjd in [obsday-1,obsday,obsday+1]:
                if mymjd!=obsday:
                    if ~((obshr - 2) < 0) or ~((obshr + 2) > 23):
                        continue
                try:
                    for o in [s.filename for s in sftp.listdir_attr(f'/FileStore/Orville-NA/{mymjd}/')]:
                        for myhr in (str(ohr) for ohr in range(obshr-2,obshr+3)):
                            if fnmatch.fnmatch(o, f"{mymjd}_{myhr.zfill(2)}0000_*oims"):
                                filepath = f'/FileStore/Orville-NA/{mymjd}/{o}'
                                fullfilelist.append((date,ra,dec,station,filepath))
                except FileNotFoundError:
                    pass
    for f in fullfilelist:
        print(f) 


