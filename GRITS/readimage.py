from GRITS import OrvilleImageDB
from contextlib import closing
import numpy as np 
from lsl_toolkits.PasiImage import PasiImageDB

def readoims(filename):
    with closing(OrvilleImageDB.OrvilleImageDB(filename, 'r')) as db:
        # Get parameters from the input file
        ints = db.nint # number of integrations
        file_start = db.header.start_time # file start time
        file_end = db.header.stop_time   # file end time
        ngrid = db.header.ngrid # image size (x-axis)
        psize = db.header.pixel_size # angular size of a pixel (at zenith)
        nchan = db.header.nchan # number of frequency channels
        station = db.header.station
        sRad   = 360.0/psize/np.pi / 2
        # Collect header and data from the whole file
        hdrlist = []
        data = np.zeros((ints,nchan,4,ngrid,ngrid))
        for i in range(ints):
            db.seek(i)
            hdr,alldata = db.read_image()
            hdrlist.append(hdr)
            data[i] = np.asarray(alldata.data)
        imagedata = np.copy(data)
        # diffim stokes I only, last image is not a diff, we'll mask it later
        data[:-1,:,0,:,:] = data[1:,:,0,:,:] - data[:-1,:,0,:,:]# PB Correct before masking
    return ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata,station

def readpims(filename):

    with closing(PasiImageDB(filename, 'r')) as db:
        ints = len(db)
        file_start = db.header['startTime'] 
        file_end = db.header['stopTime']
        ngrid = db.header['xSize']
        psize = db.header['xPixelSize']
        nchan = 1 
        sRad = 360.0/psize/np.pi / 2
        hdrlist = []
        data = np.zeros((ints,nchan,4,ngrid,ngrid))
        for i in range(ints):
            db.seek(i)
            hdr,alldata,spec = db.readImage()
            hdr['start_time'] = hdr['startTime']
            hdr['start_freq'] = hdr['freq'] - hdr['bandwidth']/2
            hdr['stop_freq'] = hdr['freq'] - hdr['bandwidth']/2
            hdr['center_ra'] = hdr['zenithRA']
            hdr['center_dec'] = hdr['zenithDec']
            hdr['int_len'] = hdr['intLen']
            hdrlist.append(hdr)
            data[i] = np.asarray(alldata.data)
        imagedata = np.copy(data)
        # diffim stokes I only, last image is not a diff, we'll mask it later
        data[:-1,:,0,:,:] = data[1:,:,0,:,:] - data[:-1,:,0,:,:]# PB Correct before masking
    return ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata,db.header['station']


