import unittest 
import numpy as np 
from bifrost import map as BFMAP
from bifrost import ndarray as BFArray
from GRITS.GPUgrouping import *
import matplotlib.pyplot as plt
ints = 10
nchan = 1
ngrid = 25

def genimagecube():
    islanddata = np.zeros((ints,nchan,4,ngrid,ngrid),dtype=int)
    islanddata[:,0,0,2:5,1] = 1 
    islanddata[:,0,0,6,2:6] = 1
    for i in range(5):
        starty = 1
        startx = 14 
        islanddata[:,0,0,starty+i,startx-i] = 1
    for i in range(5):
        starty = 12
        startx = 12
        islanddata[:,0,0,starty+i,startx+i]  = 1
    oddshape=[(3,18),(3,21),(4,20),(4,19),(5,18),(6,17)]
    for xp, yp in oddshape:
        islanddata[:,0,0,yp,xp] = 1
    return islanddata 
def plotislands(islanddata):
    islanddata = islanddata.copy('system')
    fig = plt.figure()
    plt.imshow(islanddata[0,0,0], cmap='coolwarm', vmin=0,vmax=1, origin='upper')
    for label in np.unique(islanddata[0,0,0]):
        ypix,xpix = np.where(islanddata[0,0,0]==label)
        for xp,yp in zip(xpix,ypix):
            plt.annotate(str(label), (xp-0.5,yp+0.5))
    plt.show()
    plt.close()
def plottwoisland(isl1,isl2):
    fig,axs = plt.subplots(1,2)
    for ax,isl in zip(axs,[isl1,isl2]):
        ax.imshow(isl, cmap='coolwarm', vmin=0,vmax=1, origin='upper')
        for label in np.unique(isl):
            ypix,xpix = np.where(isl==label)
            for xp,yp in zip(xpix,ypix):
                ax.annotate(str(label), (xp-0.5,yp+0.5))
    plt.show()
    plt.close()

class TestFFT(unittest.TestCase):
    # Might be better split up, but I want to be sure that each step is run in order.
    # Otherwise the output will not match.
    def test_all_grouping(self):  
        islarr = genimagecube()
        islanddata = BFArray(islarr, space='cuda')
     #   plotislands(islanddata)
        labelgroups(islanddata, ngrid, ints,nchan)
        islanddata = islanddata.copy('system')
        for t in range(ints):
            for c in range(nchan):
                for s in range(4):
                    counter = 0
                    for i in range(ngrid):
                        for j in range(ngrid):
                            if islarr[t,c,s,i,j]!=0:
                                counter+=1
                                islarr[t,c,s,i,j] = counter
        time,chan,stokes,y,x = np.where(islarr!=np.array(islanddata))
        if time.size>0:
            for t,yp,xp in zip(time,y,x):
                print(t,yp,xp)
                plottwoisland(np.array(islanddata[time,chan,stokes][0]),islarr[time,chan,stokes][0])
        np.testing.assert_array_equal(islarr, np.array(islanddata))
        islanddata = islanddata.copy('cuda')
        # GROUPCOLS
        groupvertcols(islanddata,ngrid)
        islanddata = islanddata.copy('system')
        for t in range(ints):
            for c in range(nchan):
                for s in range(4):
                    for j in range(ngrid):
                        if ((j>0) and (j<(ngrid-1))):
                            for i in range(1,ngrid-1):
                                if ((islarr[t,c,s,i,j]!=0) and (islarr[t,c,s,i-1,j]!=0)):
                                    islarr[t,c,s,i,j] = min(islarr[t,c,s,i-1,j],islarr[t,c,s,i,j])
                            for i in range(ngrid-2,0,-1):
                                if ((islarr[t,c,s,i,j]!=0) and (islarr[t,c,s,i+1,j]!=0)):
                                    islarr[t,c,s,i,j] = min(islarr[t,c,s,i+1,j],islarr[t,c,s,i,j])
                            
        time,chan,stokes,y,x = np.where(islarr!=np.array(islanddata))
        if time.size>0:
            for t,yp,xp in zip(time,y,x):
                print(t,yp,xp)
                plottwoisland(np.array(islanddata[time,chan,stokes][0]),islarr[time,chan,stokes][0])
        np.testing.assert_array_equal(islarr, np.array(islanddata))
        islanddata = islanddata.copy('cuda')
        # GROUPROWS
        grouphorizrows(islanddata,ngrid)
        islanddata = islanddata.copy('system')
        for t in range(ints):
            for c in range(nchan):
                for s in range(4):
                    for i in range(ngrid):
                        if ((i>0) and (i<(ngrid-1))):
                            for j in range(1,ngrid-1):
                                if ((islarr[t,c,s,i,j]!=0) and (islarr[t,c,s,i,j+1]!=0) and (islarr[t,c,s,i,j+1] < islarr[t,c,s,i,j])):
                                    for k in range(0,ngrid):
                                        if (islarr[t,c,s,i,k]==islarr[t,c,s,i,j]):
                                            islarr[t,c,s,i,k] = islarr[t,c,s,i,j+1]
                                if ((islarr[t,c,s,i,j]!=0) and (islarr[t,c,s,i,j-1]!=0) and (islarr[t,c,s,i,j-1] < islarr[t,c,s,i,j])):
                                    for k in range(0,ngrid):
                                        if (islarr[t,c,s,i,k]==islarr[t,c,s,i,j]):
                                            islarr[t,c,s,i,k] = islarr[t,c,s,i,j-1]
        time,chan,stokes,y,x = np.where(islarr!=np.array(islanddata))
        if time.size>0:
            for t,yp,xp in zip(time,y,x):
                print(t,yp,xp)
                plottwoisland(np.array(islanddata[time,chan,stokes][0]),islarr[time,chan,stokes][0])
        np.testing.assert_array_equal(islarr, np.array(islanddata))
        # GROUPCOLS
        islanddata = islanddata.copy('cuda')
        groupvertcols(islanddata,ngrid)
        islanddata = islanddata.copy('system')
        for t in range(ints):
            for c in range(nchan):
                for s in range(4):
                    for j in range(ngrid):
                        if ((j>0) and (j<(ngrid-1))):
                            for i in range(1,ngrid-1):
                                if ((islarr[t,c,s,i,j]!=0) and (islarr[t,c,s,i-1,j]!=0)):
                                    islarr[t,c,s,i,j] = min(islarr[t,c,s,i-1,j],islarr[t,c,s,i,j])
                            for i in range(ngrid-2,0,-1):
                                if ((islarr[t,c,s,i,j]!=0) and (islarr[t,c,s,i+1,j]!=0)):
                                    islarr[t,c,s,i,j] = min(islarr[t,c,s,i+1,j],islarr[t,c,s,i,j])
                            
        time,chan,stokes,y,x = np.where(islarr!=np.array(islanddata))
        if time.size>0:
            for t,yp,xp in zip(time,y,x):
                print(t,yp,xp)
                plottwoisland(np.array(islanddata[time,chan,stokes][0]),islarr[time,chan,stokes][0])
        np.testing.assert_array_equal(islarr, np.array(islanddata))
        islanddata = islanddata.copy('cuda')
        # ROTATE Island data
        rotarr = np.zeros((ints,nchan,4,2*ngrid-1,2*ngrid-1),dtype=int)
        rotateddata = BFArray(rotarr,space='cuda',dtype=int)
        BFMap("""rm(time,chan,stokes,%i-i+j-1,i+j-2) = isldata(time,chan,stokes,i,j);""" % (ngrid,),
            {'isldata':islanddata,'rm':rotateddata},
            axis_names = ('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))
        rotgrid = 2*ngrid-1
        rotateddata = rotateddata.copy('system')
        for t in range(ints):
            for c in range(nchan):
                for s in range(4):
                    for i in range(ngrid):
                        for j in range(ngrid):
                            rotarr[t,c,s,ngrid-i+j-1,i+j-2] = islarr[t,c,s,i,j]
        time,chan,stokes,y,x = np.where(rotarr!=np.array(rotateddata))
        if time.size>0:
            for t,yp,xp in zip(time,y,x):
                print(t,yp,xp)
                plottwoisland(np.array(rotateddata[time,chan,stokes][0]),rotarr[time,chan,stokes][0])
        np.testing.assert_array_equal(rotarr, np.array(rotateddata))
        rotateddata = rotateddata.copy('cuda')
        # GROUP DIAG COLS
        groupdiagcols(rotateddata,rotgrid)
        rotateddata = rotateddata.copy('system')
        for t in range(ints):
            for c in range(nchan):
                for s in range(4):
                    for j in range(rotgrid):
                        if ((j>0) and (j<(rotgrid-1))):
                            for i in range(2,rotgrid-2):
                                if ((rotarr[t,c,s,i,j]!=0) and (rotarr[t,c,s,i-2,j]!=0)):
                                    rotarr[t,c,s,i,j] = min(rotarr[t,c,s,i-2,j],rotarr[t,c,s,i,j])
                            for i in range(rotgrid-3,1,-1):
                                if ((rotarr[t,c,s,i,j]!=0) and (rotarr[t,c,s,i+2,j]!=0)):
                                    rotarr[t,c,s,i,j] = min(rotarr[t,c,s,i+2,j],rotarr[t,c,s,i,j])
        time,chan,stokes,y,x = np.where(rotarr!=np.array(rotateddata))
        if time.size>0:
            for t,yp,xp in zip(time,y,x):
                print(t,yp,xp)
                plottwoisland(np.array(rotateddata[time,chan,stokes][0]),rotarr[time,chan,stokes][0])
        np.testing.assert_array_equal(rotarr, np.array(rotateddata))
        rotateddata = rotateddata.copy('cuda')
        # GROUP DIAG ROWS
        groupdiagrows(rotateddata,rotgrid)
        rotateddata = rotateddata.copy('system')
        for t in range(ints):
            for c in range(nchan):
                for s in range(4):
                    for i in range(2,rotgrid-2):
                        for j in range(2,rotgrid-2):
                            if ((rotarr[t,c,s,i,j]!=0) and (rotarr[t,c,s,i,j+2]!=0) and (rotarr[t,c,s,i,j+2] < rotarr[t,c,s,i,j])):
                                for k in range(0,rotgrid):
                                    if (rotarr[t,c,s,i,k]==rotarr[t,c,s,i,j]):
                                        rotarr[t,c,s,i,k] = rotarr[t,c,s,i,j+2]
                            if ((rotarr[t,c,s,i,j]!=0) and (rotarr[t,c,s,i,j-2]!=0) and (rotarr[t,c,s,i,j-2] < rotarr[t,c,s,i,j])):
                                for k in range(0,rotgrid):
                                    if (rotarr[t,c,s,i,k]==rotarr[t,c,s,i,j]):
                                        rotarr[t,c,s,i,k] = rotarr[t,c,s,i,j-2]
        time,chan,stokes,y,x = np.where(rotarr!=np.array(rotateddata))
        if time.size>0:
            for t,yp,xp in zip(time,y,x):
                print(t,yp,xp)
                plottwoisland(np.array(rotateddata[time,chan,stokes][0]),rotarr[time,chan,stokes][0])
        np.testing.assert_array_equal(rotarr, np.array(rotateddata))
        rotateddata = rotateddata.copy('cuda')
        # GROUP DIAG COLS
        groupdiagcols(rotateddata,rotgrid)
        rotateddata = rotateddata.copy('system')
        for t in range(ints):
            for c in range(nchan):
                for s in range(4):
                    for j in range(rotgrid):
                        if ((j>0) and (j<(rotgrid-1))):
                            for i in range(2,rotgrid-2):
                                if ((rotarr[t,c,s,i,j]!=0) and (rotarr[t,c,s,i-2,j]!=0)):
                                    rotarr[t,c,s,i,j] = min(rotarr[t,c,s,i-2,j],rotarr[t,c,s,i,j])
                            for i in range(rotgrid-3,1,-1):
                                if ((rotarr[t,c,s,i,j]!=0) and (rotarr[t,c,s,i+2,j]!=0)):
                                    rotarr[t,c,s,i,j] = min(rotarr[t,c,s,i+2,j],rotarr[t,c,s,i,j])
        time,chan,stokes,y,x = np.where(rotarr!=np.array(rotateddata))
        if time.size>0:
            for t,yp,xp in zip(time,y,x):
                print(t,yp,xp)
                plottwoisland(np.array(rotateddata[time,chan,stokes][0]),rotarr[time,chan,stokes][0])
        np.testing.assert_array_equal(rotarr, np.array(rotateddata))
        rotateddata = rotateddata.copy('cuda')

        # UPDATE UNROTATED Island data
        BFMap("""isldata(time,chan,stokes,i,j) = rm(time,chan,stokes,%i-i+j-1,i+j-2);""" % (ngrid,),
            {'isldata':islanddata,'rm':rotateddata},
            axis_names = ('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))
        for t in range(ints):
            for c in range(nchan):
                for s in range(4):
                    for i in range(ngrid):
                        for j in range(ngrid):
                            islarr[t,c,s,i,j] = rotarr[t,c,s,ngrid-i+j-1,i+j-2]
        islanddata = islanddata.copy("system")
        np.testing.assert_array_equal(np.array(islanddata),islarr)
        islanddata = islanddata.copy("cuda")
        # GROUPCOLS
        groupvertcols(islanddata,ngrid)
        islanddata = islanddata.copy('system')
        for t in range(ints):
            for c in range(nchan):
                for s in range(4):
                    for j in range(ngrid):
                        if ((j>0) and (j<(ngrid-1))):
                            for i in range(1,ngrid-1):
                                if ((islarr[t,c,s,i,j]!=0) and (islarr[t,c,s,i-1,j]!=0)):
                                    islarr[t,c,s,i,j] = min(islarr[t,c,s,i-1,j],islarr[t,c,s,i,j])
                            for i in range(ngrid-2,0,-1):
                                if ((islarr[t,c,s,i,j]!=0) and (islarr[t,c,s,i+1,j]!=0)):
                                    islarr[t,c,s,i,j] = min(islarr[t,c,s,i+1,j],islarr[t,c,s,i,j])
                            
        time,chan,stokes,y,x = np.where(islarr!=np.array(islanddata))
        if time.size>0:
            for t,yp,xp in zip(time,y,x):
                print(t,yp,xp)
                plottwoisland(np.array(islanddata[time,chan,stokes][0]),islarr[time,chan,stokes][0])
        np.testing.assert_array_equal(islarr, np.array(islanddata))
        islanddata = islanddata.copy('cuda')
        # GROUPROWS
        grouphorizrows(islanddata,ngrid)
        islanddata = islanddata.copy('system')
        for t in range(ints):
            for c in range(nchan):
                for s in range(4):
                    for i in range(ngrid):
                        if ((i>0) and (i<(ngrid-1))):
                            for j in range(1,ngrid-1):
                                if ((islarr[t,c,s,i,j]!=0) and (islarr[t,c,s,i,j+1]!=0) and (islarr[t,c,s,i,j+1] < islarr[t,c,s,i,j])):
                                    for k in range(0,ngrid):
                                        if (islarr[t,c,s,i,k]==islarr[t,c,s,i,j]):
                                            islarr[t,c,s,i,k] = islarr[t,c,s,i,j+1]
                                if ((islarr[t,c,s,i,j]!=0) and (islarr[t,c,s,i,j-1]!=0) and (islarr[t,c,s,i,j-1] < islarr[t,c,s,i,j])):
                                    for k in range(0,ngrid):
                                        if (islarr[t,c,s,i,k]==islarr[t,c,s,i,j]):
                                            islarr[t,c,s,i,k] = islarr[t,c,s,i,j-1]
        time,chan,stokes,y,x = np.where(islarr!=np.array(islanddata))
        if time.size>0:
            for t,yp,xp in zip(time,y,x):
                print(t,yp,xp)
                plottwoisland(np.array(islanddata[time,chan,stokes][0]),islarr[time,chan,stokes][0])
        np.testing.assert_array_equal(islarr, np.array(islanddata))
        # GROUPCOLS
        islanddata = islanddata.copy('cuda')
        groupvertcols(islanddata,ngrid)
        islanddata = islanddata.copy('system')
        for t in range(ints):
            for c in range(nchan):
                for s in range(4):
                    for j in range(ngrid):
                        if ((j>0) and (j<(ngrid-1))):
                            for i in range(1,ngrid-1):
                                if ((islarr[t,c,s,i,j]!=0) and (islarr[t,c,s,i-1,j]!=0)):
                                    islarr[t,c,s,i,j] = min(islarr[t,c,s,i-1,j],islarr[t,c,s,i,j])
                            for i in range(ngrid-2,0,-1):
                                if ((islarr[t,c,s,i,j]!=0) and (islarr[t,c,s,i+1,j]!=0)):
                                    islarr[t,c,s,i,j] = min(islarr[t,c,s,i+1,j],islarr[t,c,s,i,j])
                            
        time,chan,stokes,y,x = np.where(islarr!=np.array(islanddata))
        if time.size>0:
            for t,yp,xp in zip(time,y,x):
                print(t,yp,xp)
                plottwoisland(np.array(islanddata[time,chan,stokes][0]),islarr[time,chan,stokes][0])
        np.testing.assert_array_equal(islarr, np.array(islanddata))
        islanddata = islanddata.copy('cuda')
        # ROTATE Island data
        rotarr = np.zeros((ints,nchan,4,2*ngrid-1,2*ngrid-1),dtype=int)
        rotateddata = BFArray(rotarr,space='cuda',dtype=int)
        BFMap("""rm(time,chan,stokes,%i-i+j-1,i+j-2) = isldata(time,chan,stokes,i,j);""" % (ngrid,),
            {'isldata':islanddata,'rm':rotateddata},
            axis_names = ('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))
        rotgrid = 2*ngrid-1
        rotateddata = rotateddata.copy('system')
        for t in range(ints):
            for c in range(nchan):
                for s in range(4):
                    for i in range(ngrid):
                        for j in range(ngrid):
                            rotarr[t,c,s,ngrid-i+j-1,i+j-2] = islarr[t,c,s,i,j]
        time,chan,stokes,y,x = np.where(rotarr!=np.array(rotateddata))
        if time.size>0:
            for t,yp,xp in zip(time,y,x):
                print(t,yp,xp)
                plottwoisland(np.array(rotateddata[time,chan,stokes][0]),rotarr[time,chan,stokes][0])
        np.testing.assert_array_equal(rotarr, np.array(rotateddata))
        rotateddata = rotateddata.copy('cuda')
        # GROUP DIAG COLS
        groupdiagcols(rotateddata,rotgrid)
        rotateddata = rotateddata.copy('system')
        for t in range(ints):
            for c in range(nchan):
                for s in range(4):
                    for j in range(rotgrid):
                        if ((j>0) and (j<(rotgrid-1))):
                            for i in range(2,rotgrid-2):
                                if ((rotarr[t,c,s,i,j]!=0) and (rotarr[t,c,s,i-2,j]!=0)):
                                    rotarr[t,c,s,i,j] = min(rotarr[t,c,s,i-2,j],rotarr[t,c,s,i,j])
                            for i in range(rotgrid-3,1,-1):
                                if ((rotarr[t,c,s,i,j]!=0) and (rotarr[t,c,s,i+2,j]!=0)):
                                    rotarr[t,c,s,i,j] = min(rotarr[t,c,s,i+2,j],rotarr[t,c,s,i,j])
        time,chan,stokes,y,x = np.where(rotarr!=np.array(rotateddata))
        if time.size>0:
            for t,yp,xp in zip(time,y,x):
                print(t,yp,xp)
                plottwoisland(np.array(rotateddata[time,chan,stokes][0]),rotarr[time,chan,stokes][0])
        np.testing.assert_array_equal(rotarr, np.array(rotateddata))
        rotateddata = rotateddata.copy('cuda')
        # GROUP DIAG ROWS
        groupdiagrows(rotateddata,rotgrid)
        rotateddata = rotateddata.copy('system')
        for t in range(ints):
            for c in range(nchan):
                for s in range(4):
                    for i in range(2,rotgrid-2):
                        for j in range(2,rotgrid-2):
                            if ((rotarr[t,c,s,i,j]!=0) and (rotarr[t,c,s,i,j+2]!=0) and (rotarr[t,c,s,i,j+2] < rotarr[t,c,s,i,j])):
                                for k in range(0,rotgrid):
                                    if (rotarr[t,c,s,i,k]==rotarr[t,c,s,i,j]):
                                        rotarr[t,c,s,i,k] = rotarr[t,c,s,i,j+2]
                            if ((rotarr[t,c,s,i,j]!=0) and (rotarr[t,c,s,i,j-2]!=0) and (rotarr[t,c,s,i,j-2] < rotarr[t,c,s,i,j])):
                                for k in range(0,rotgrid):
                                    if (rotarr[t,c,s,i,k]==rotarr[t,c,s,i,j]):
                                        rotarr[t,c,s,i,k] = rotarr[t,c,s,i,j-2]
        time,chan,stokes,y,x = np.where(rotarr!=np.array(rotateddata))
        if time.size>0:
            for t,yp,xp in zip(time,y,x):
                print(t,yp,xp)
                plottwoisland(np.array(rotateddata[time,chan,stokes][0]),rotarr[time,chan,stokes][0])
        np.testing.assert_array_equal(rotarr, np.array(rotateddata))
        rotateddata = rotateddata.copy('cuda')
        # GROUP DIAG COLS
        groupdiagcols(rotateddata,rotgrid)
        rotateddata = rotateddata.copy('system')
        for t in range(ints):
            for c in range(nchan):
                for s in range(4):
                    for j in range(rotgrid):
                        if ((j>0) and (j<(rotgrid-1))):
                            for i in range(2,rotgrid-2):
                                if ((rotarr[t,c,s,i,j]!=0) and (rotarr[t,c,s,i-2,j]!=0)):
                                    rotarr[t,c,s,i,j] = min(rotarr[t,c,s,i-2,j],rotarr[t,c,s,i,j])
                            for i in range(rotgrid-3,1,-1):
                                if ((rotarr[t,c,s,i,j]!=0) and (rotarr[t,c,s,i+2,j]!=0)):
                                    rotarr[t,c,s,i,j] = min(rotarr[t,c,s,i+2,j],rotarr[t,c,s,i,j])
        time,chan,stokes,y,x = np.where(rotarr!=np.array(rotateddata))
        if time.size>0:
            for t,yp,xp in zip(time,y,x):
                print(t,yp,xp)
                plottwoisland(np.array(rotateddata[time,chan,stokes][0]),rotarr[time,chan,stokes][0])
        np.testing.assert_array_equal(rotarr, np.array(rotateddata))
        rotateddata = rotateddata.copy('cuda')

        # UPDATE UNROTATED Island data
        BFMap("""isldata(time,chan,stokes,i,j) = rm(time,chan,stokes,%i-i+j-1,i+j-2);""" % (ngrid,),
            {'isldata':islanddata,'rm':rotateddata},
            axis_names = ('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))
        for t in range(ints):
            for c in range(nchan):
                for s in range(4):
                    for i in range(ngrid):
                        for j in range(ngrid):
                            islarr[t,c,s,i,j] = rotarr[t,c,s,ngrid-i+j-1,i+j-2]
        islanddata = islanddata.copy("system")
        np.testing.assert_array_equal(np.array(islanddata),islarr)
        

