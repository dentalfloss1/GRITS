import numpy as np 
from GRITS import OrvilleImageDB
from GRITS.ateamtools import getsunjup, getcasa, getcyga, getcrab, getvira
from GRITS.GPUimagestats import getminrms,fluxcalmask,getmultistat,getdiffstats
from GRITS.imagetools import  angsep,maskhorizon, pbcorr, hdrtopixmask, hdrtoworld, maskislands
from GRITS.readimage import readoims,readpims
import unittest 

class TestIMSTATS(unittest.TestCase):
    def test_all_stats(self):
        filename="data/oneofftransient.oims"
        ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata,station = readoims(filename)
        prepb = np.copy(imagedata)
        for c in range(nchan):
            XX,YY = pbcorr(hdrlist[0],ngrid,c,psize,station)
            imagedata[:,c,0,:,:]/=((XX+YY)/2)
            data[:,c,0,:,:]/=((XX+YY)/2)
        pbarr = np.load('data/pbimage.npy')
        np.testing.assert_array_equal(pbarr, imagedata)
        sunpos, juppos = getsunjup(file_start, file_end, ints)
        sunarr = np.load("data/sunarr.npy")
        juparr = np.load("data/juparr.npy")
        np.testing.assert_array_equal(sunarr,sunpos)
        np.testing.assert_array_equal(juparr,juppos)
        
        invalid, circle = maskhorizon(data,hdrlist,ngrid,sRad,pbpercent=0.667)
        data[:,:,:,invalid[0], invalid[1]] = np.nan
        dataver = np.load("data/posthorizonmask.npy")
        np.testing.assert_array_equal(dataver,data)
        
        minrms = getminrms(sRad, imagedata,ints,nchan,ngrid)
        minrmsver = np.load("data/minrms.npy")
        np.testing.assert_array_equal(minrmsver, np.array(minrms))
        
        
        imagedata, data ,corrfluxlist = fluxcalmask(sRad, data, imagedata, hdrlist, minrms, sunpos, juppos, nchan, ngrid)
        calimver = np.load("data/calimver.npy")
        caldataver = np.load("data/caldataver.npy")
        corrfluxver = np.load("data/corrfluxver.npy")
        np.testing.assert_array_equal(imagedata, calimver)
        np.testing.assert_array_equal(data, caldataver)
        np.testing.assert_array_equal(np.array(corrfluxlist), corrfluxver)
        
        meanofdiff, stdofdiff = getdiffstats(data,sRad,ints,nchan,ngrid)
        diffmeanver = np.load("data/diffmeanver.npy")
        diffstdver = np.load("data/diffstdver.npy")
        np.testing.assert_array_equal(meanofdiff, diffmeanver)
        np.testing.assert_array_equal(stdofdiff, diffstdver)
