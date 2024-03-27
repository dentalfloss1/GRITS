from GRITS import OrvilleImageDB
from contextlib import closing
import numpy as np 
from lsl_toolkits.PasiImage import PasiImageDB

def readoims(filename,hdronly=False):
    with closing(OrvilleImageDB.OrvilleImageDB(filename, 'r')) as db:
        # Get parameters from the input file
        ints = db.nint # number of integrations
        file_start = db.header.start_time # file start time
        ngrid = db.header.ngrid # image size (x-axis)
        psize = db.header.pixel_size # angular size of a pixel (at zenith)
        nchan = db.header.nchan # number of frequency channels
        station = db.header.station
        try:
            sRad   = 360.0/psize/np.pi / 2
            db.seek(ints-1)
            hdr,_ = db.read_image()
            intlen = hdr['int_len']
            if intlen > 1: # if larger than one, has to be in seconds
                intlen = intlen/24.0/60.0/60.0 # seconds to days
            file_end = hdr['start_time'] + intlen
            if not hdronly:
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
            else:
                return file_start,file_end
        except ZeroDivisionError:
            if hdronly:
                return 0.0,0.0
            else:
                ints= 0
                file_start= 0.0
                file_end = 0.0
                ngrid = 0
                psize = 0
                nchan = 0
                sRad = 0
                hdrlist = []
                data = []
                imagedata = []
                station = b''
                return ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata,station
def readpims(filename,hdronly=False):

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
        if not hdronly:
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
                thisdata = alldata.data
                data[i,0] = np.copy(np.asarray(alldata.data).transpose(0,2,1))
            imagedata = np.copy(data)
            # diffim stokes I only, last image is not a diff, we'll mask it later
            data[:-1,:,0,:,:] = data[1:,:,0,:,:] - data[:-1,:,0,:,:]# PB Correct before masking
            return ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata,db.header['station']
        else:
            
            return file_start,file_end


