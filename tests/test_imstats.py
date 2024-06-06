import numpy as np 
from GRITS import OrvilleImageDB
from GRITS.ateamtools import getsunjup, getcasa, getcyga, getcrab, getvira
from GRITS.GPUimagestats import getminrms,fluxcalmask,getmultistat,getdiffstats
from GRITS.imagetools import  angsep,maskhorizon, pbcorr, hdrtopixmask, hdrtoworld, maskislands
from GRITS.readimage import readoims,readpims
import unittest 
import matplotlib.pyplot as plt

def plotislands(islanddata):
    islanddata = islanddata.copy('system')
    fig = plt.figure()
    subdata = islanddata[2,0,0,60:80,60:80]
    plt.imshow(subdata, cmap='coolwarm', vmin=0,vmax=1, origin='upper')
    for label in np.unique(subdata):
        ypix,xpix = np.where(subdata==label)
        for xp,yp in zip(xpix,ypix):
            plt.annotate(str(label), (xp-0.5,yp+0.5))
    plt.show()
    plt.close()

class TestIMSTATS(unittest.TestCase):
    def test_all_stats(self):
        filename="data/oneofftransient.oims"
        ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata,station = readoims(filename)
        prepb = np.copy(imagedata)
        for c in range(nchan):
            XX,YY = pbcorr(hdrlist[0],ngrid,c,psize,station)
            imagedata[:,c,0,:,:]/=((XX+YY)/2)
            data[:,c,0,:,:]/=((XX+YY)/2)
       #  pbarr = np.save('data/pbimage.npy',imagedata)
        pbarr = np.load('data/pbimage.npy')
        np.testing.assert_array_equal(pbarr, imagedata)
        sunpos, juppos = getsunjup(file_start, file_end, ints)
        sunarr = np.load("data/sunarr.npy")
        juparr = np.load("data/juparr.npy")
        np.testing.assert_array_equal(sunarr,sunpos)
        np.testing.assert_array_equal(juparr,juppos)
        
        invalid, circle = maskhorizon(data,hdrlist,ngrid,sRad,pbpercent=0.667)
        data[:,:,:,invalid[0], invalid[1]] = np.nan
        # dataver = np.save("data/posthorizonmask.npy",data)
        dataver = np.load("data/posthorizonmask.npy")
        np.testing.assert_array_equal(dataver,data)
        
        minrms = getminrms(sRad, imagedata,ints,nchan,ngrid)
       #  np.save("data/minrms",minrms)
        minrmsver = np.load("data/minrms.npy")
        np.testing.assert_array_equal(minrmsver, minrms)
        
        
        imagedata, data ,corrfluxlist = fluxcalmask(sRad, data, imagedata, hdrlist, minrms, sunpos, juppos, nchan, ngrid)
        # np.save("data/calimver.npy",imagedata)
        # np.save("data/caldataver.npy",data)
        # np.save("data/corrfluxver.npy",np.array(corrfluxlist))
        calimver = np.load("data/calimver.npy")
        caldataver = np.load("data/caldataver.npy")
        corrfluxver = np.load("data/corrfluxver.npy")
        np.testing.assert_array_equal(imagedata, calimver)
        np.testing.assert_array_equal(data, caldataver)
        np.testing.assert_array_equal(np.array(corrfluxlist), corrfluxver)
        
        meanofdiff, stdofdiff = getdiffstats(data,sRad,ints,nchan,ngrid)
        # np.save("data/diffmeanver.npy",meanofdiff)
        # np.save("data/diffstdver.npy",stdofdiff)
        diffmeanver = np.load("data/diffmeanver.npy")
        diffstdver = np.load("data/diffstdver.npy")
        np.testing.assert_array_equal(meanofdiff, diffmeanver)
        np.testing.assert_array_equal(stdofdiff, diffstdver)

        removechan = []
        outchannels=0
        for i,c in enumerate(corrfluxlist):
            if c!=0:
                outchannels+=1
        rmsdata = np.zeros((ints,outchannels))
        for i,c in enumerate(corrfluxlist):
            if c==0:
                removechan.append(c)
            else:
                rmsdata[:,i] = minrms[:,i]*c
        allchannels = np.arange(data.shape[1])
        data  = np.copy(data[:,~np.isin(allchannels,removechan),:,:,:])
        imagedata  = np.copy(imagedata[:,~np.isin(allchannels,removechan),:,:,:])
        nchan = imagedata.shape[1]


        islanddata = maskislands(data,rmsdata,ints,nchan,ngrid,nsigma=5)
        islanddata = islanddata.copy("system")
        isldat = np.load("data/imstatisl.npy")
        np.testing.assert_array_equal(isldat,np.array(islanddata))


