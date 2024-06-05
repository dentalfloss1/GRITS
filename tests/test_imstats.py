import numpy as np 
from GRITS import OrvilleImageDB
from GRITS.ateamtools import getsunjup, getcasa, getcyga, getcrab, getvira
from GRITS.GPUimagestats import getminrms,fluxcalmask,getmultistat,getdiffstats
from GRITS.imagetools import  angsep,maskhorizon, pbcorr, hdrtopixmask, hdrtoworld, maskislands
from GRITS.readimage import readoims,readpims
filename="data/oneofftransient.oims"
ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata,station = readoims(filename)
for c in range(nchan):
    XX,YY = pbcorr(hdrlist[0],ngrid,c,psize,station)
    imagedata[:,c,0,:,:]/=((XX+YY)/2)
    data[:,c,0,:,:]/=((XX+YY)/2)

sunpos, juppos = getsunjup(file_start, file_end, ints)
invalid, circle = maskhorizon(data,hdrlist,ngrid,sRad,pbpercent=0.667)
data[:,:,:,invalid[0], invalid[1]] = np.nan

minrms = getminrms(sRad, imagedata,ints,nchan,ngrid)
print(minrms)
print(imagedata[0,0,0,:,:].max())
