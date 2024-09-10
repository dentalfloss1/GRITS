from lsl_toolkits.OrvilleImage import OrvilleImageDB
from contextlib import closing
import numpy as np 
from lsl_toolkits.PasiImage import PasiImageDB
from tqdm import tqdm

def matchfilepairs(station1,station2):
    """Given filelists from two stations, read file times
    and pair up"""
    f1times = []
    for f1 in station1:
        print(f1)
        try:
            if "oims" in f1:
                hasoims = True
                file_start,file_end,minfreq = readoims(f1,hdronly=True)
            else:
                file_start,file_end,minfreq = readpims(f1,hdronly=True)
            if minfreq < 18e6:
                print(f1,"has frequencies that are too low for good results. Skipping")
                continue
            if minfreq > 70e6:
                print(f1,"has frequencies that are too high for good results. Skipping")
                continue
            if file_start != 0.0:
                f1times.append((f1,file_start, file_end))
            else:
                print(f1, "has invalid data")
        except Exception as e:
            print(e)
            print("Errors on file read, skip this one")
    f2times = []
    for f2 in station2:
        print(f2)
        try:
            if "oims" in f2:
                hasoims = True
                file_start,file_end,minfreq = readoims(f2,hdronly=True)
            else:
                file_start,file_end,minfreq = readpims(f2,hdronly=True)
            if minfreq < 18e6:
                print(f2,"has frequencies that are too low for good results. Skipping")
                continue
            if file_start != 0.0:
                f2times.append((f2,file_start,file_end))
            else:
                print(f2, "has invalid data")
        except Exception as e:
            print(e)
            print("Errors on file read, skip this one")
    matchingfilepairs = []
    for f1,f1s,f1e in f1times:
        for f2,f2s,f2e in f2times:
            # Check that file 2 does not end before file 1 starts
            f2end = f2e - f1s 
            # Check that file 2 does not start after file 1 ends
            f2start = f1e - f2s 
            if (f2end > 0) and (f2start > 0):
                matchingfilepairs.append((f1,f2))
    return matchingfilepairs

def readoims(filename,hdronly=False,startind=None,stopind=None):
    with closing(OrvilleImageDB(filename, 'r')) as db:
        # Get parameters from the input file
        file_start = db.header.start_time # file start time
        ngrid = db.header.ngrid # image size (x-axis)
        psize = db.header.pixel_size # angular size of a pixel (at zenith)
        nchan = db.header.nchan # number of frequency channels
        station = db.header.station
        allints = db.nint # number of integrations
        if (startind!=None) and (stopind!=None):
            ints = stopind - startind + 1
        else:
            ints = db.nint # number of integrations
            startind = 0
            stopind = ints-1
        print("about to read",ints,"imagecubes into memory")
        sRad   = 360.0/psize/np.pi / 2
        db.seek(stopind)
        hdr,_ = db.read_image()
        intlen = hdr['int_len']
        if station not in [b'LWASV',b'LWANA']:
           station=b'LWASV'
        minfreq = (hdr['start_freq']  + (hdr['bandwidth']/2))
        if intlen > 1: # if larger than one, has to be in seconds
            intlen = intlen/24.0/60.0/60.0 # seconds to days
        file_end = hdr['start_time'] + intlen
        if not hdronly:
            # Collect header and data from the whole file
            hdrlist = []
            data = np.zeros((ints,nchan,4,ngrid,ngrid))
            for i in tqdm(range(ints)):
                db.seek(i+startind)
                hdr,alldata = db.read_image()
                hdr['psize'] = psize
                hdrlist.append(hdr)
                data[i] = np.asarray(alldata.data)
            imagedata = np.copy(data)
            # first image is not a diff, we'll mask it later
            data[1:,:,0,:,:] = data[1:,:,0,:,:] - data[:-1:,:,0,:,:]# PB Correct before masking
            for stokesno in range(1,4):
                data[1:,:,stokesno,:,:] = data[1:,:,stokesno,:,:] - data[:-1:,:,stokesno,:,:]# PB Correct before masking
            return ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata,station
        else:
            return file_start,file_end,minfreq
def readpims(filename,hdronly=False,startind=None,stopind=None):
    with closing(PasiImageDB(filename, 'r')) as db:
        file_start = db.header['startTime'] 
        file_end = db.header['stopTime']
        ngrid = db.header['xSize']
        psize = db.header['xPixelSize']
        nchan = 1 
        sRad = 360.0/psize/np.pi / 2
        print(startind,stopind)
        if (startind!=None) and (stopind!=None):
            ints = stopind - startind + 1
        else:
            ints = len(db)
            startind = 0
            stopind = ints - 1
        hdrlist = []
        db.seek(stopind)
        hdr,_,_ = db.readImage()
        minfreq=hdr['freq']
        data = np.zeros((ints,nchan,4,ngrid,ngrid))
        if not hdronly:
            for i in range(ints):
                db.seek(i+startind)
                hdr,alldata,spec = db.readImage()
                hdr['start_time'] = hdr['startTime']
                hdr['start_freq'] = hdr['freq'] - hdr['bandwidth']/2
                hdr['stop_freq'] = hdr['freq'] - hdr['bandwidth']/2
                hdr['center_ra'] = hdr['zenithRA']
                hdr['center_dec'] = hdr['zenithDec']
                hdr['int_len'] = hdr['intLen']
                hdr['psize'] = psize
                hdrlist.append(hdr)
                thisdata = alldata.data
                data[i,0] = np.copy(np.asarray(alldata.data).transpose(0,2,1))
            imagedata = np.copy(data)
            #  first image is not a diff, we'll mask it later
            data[1:,:,0,:,:] = data[1:,:,0,:,:] - data[:-1:,:,0,:,:]# PB Correct before masking
            for stokesno in range(1,4):
                data[1:,:,stokesno,:,:] = np.abs(data[1:,:,stokesno,:,:]) - np.abs(data[:-1:,:,stokesno,:,:])# PB Correct before masking
            return ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata,db.header['station']
        else:
            
            return file_start,file_end,minfreq


