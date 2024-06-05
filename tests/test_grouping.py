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
        islanddata = genimagecube()
        islanddata = BFArray(islanddata, space='cuda')
     #   plotislands(islanddata)
        labelgroups(islanddata, ngrid, ints,nchan)
        islanddata = islanddata.copy('system')
        islarr = np.load("data/label1.npy")
        np.testing.assert_array_equal(islarr, np.array(islanddata))
        islanddata = islanddata.copy('cuda')
        # GROUPCOLS
        groupvertcols(islanddata,ngrid)
        islanddata = islanddata.copy('system')
        islarr = np.load("data/groupvert1.npy")
        np.testing.assert_array_equal(islarr, np.array(islanddata))
        islanddata = islanddata.copy('cuda')
        # GROUPROWS
        grouphorizrows(islanddata,ngrid)
        islanddata = islanddata.copy('system')
    #    np.save("grouprow1.npy",np.array(islanddata))
        islarr = np.load('data/grouprow1.npy')
        np.testing.assert_array_equal(islarr, np.array(islanddata))
        # GROUPCOLS
        islanddata = islanddata.copy('cuda')
        groupvertcols(islanddata,ngrid)
        islanddata = islanddata.copy('system')
        islarr = np.load("data/groupvert2.npy")
        np.testing.assert_array_equal(islarr, np.array(islanddata))
        islanddata = islanddata.copy('cuda')
        # ROTATE Island data
        rotateddata = BFArray(np.zeros((ints,nchan,4,2*ngrid-1,2*ngrid-1)),space='cuda',dtype=int)
        BFMap("""rm(time,chan,stokes,%i-i+j-1,i+j-2) = isldata(time,chan,stokes,i,j);""" % (ngrid,),
            {'isldata':islanddata,'rm':rotateddata},
            axis_names = ('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))
        rotgrid = 2*ngrid-1
    # 
        # GROUP DIAG COLS
        groupdiagcols(rotateddata,rotgrid)
        rotateddata = rotateddata.copy('system')
        rotarr = np.load("data/groupdiagcol1.npy")
        np.testing.assert_array_equal(rotarr, np.array(rotateddata))
        rotateddata = rotateddata.copy('cuda')
        # GROUP DIAG ROWS
        groupdiagrows(rotateddata,rotgrid)
        rotateddata = rotateddata.copy('system')
        rotarr = np.load("data/groupdiagrow1.npy")
        np.testing.assert_array_equal(rotarr,np.array(rotateddata))
        rotateddata = rotateddata.copy('cuda')
        # GROUP DIAG COLS
        groupdiagcols(rotateddata,rotgrid)
        rotateddata = rotateddata.copy('system')
        rotarr = np.load("data/groupdiagcol2.npy")
        np.testing.assert_array_equal(rotarr,np.array(rotateddata))
        rotateddata = rotateddata.copy('cuda')
        # UPDATE UNROTATED Island data
        BFMap("""isldata(time,chan,stokes,i,j) = rm(time,chan,stokes,%i-i+j-1,i+j-2);""" % (ngrid,),
            {'isldata':islanddata,'rm':rotateddata},
            axis_names = ('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))
        # GROUPCOLS
        islanddata = islanddata.copy('cuda')
        groupvertcols(islanddata,ngrid)
        islanddata = islanddata.copy('system')
        islarr = np.load("data/groupvert3.npy")
        np.testing.assert_array_equal(islarr, np.array(islanddata))
        islanddata = islanddata.copy('cuda')
        # GROUPROWS
        grouphorizrows(islanddata,ngrid)
        islanddata = islanddata.copy('system')
        islarr = np.load('data/grouprow2.npy')
        np.testing.assert_array_equal(islarr, np.array(islanddata))
        # GROUPCOLS
        islanddata = islanddata.copy('cuda')
        groupvertcols(islanddata,ngrid)
        islanddata = islanddata.copy('system')
        islarr = np.load("data/groupvert4.npy")
        np.testing.assert_array_equal(islarr, np.array(islanddata))
        islanddata = islanddata.copy('cuda')
    
        # ROTATE Island data
        rotateddata = BFArray(np.zeros((ints,nchan,4,2*ngrid-1,2*ngrid-1)),space='cuda',dtype=int)
        BFMap("""rm(time,chan,stokes,%i-i+j-1,i+j-2) = isldata(time,chan,stokes,i,j);""" % (ngrid,),
            {'isldata':islanddata,'rm':rotateddata},
            axis_names = ('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))
        rotgrid = 2*ngrid-1
        # GROUP DIAG COLS
        groupdiagcols(rotateddata,rotgrid)
        rotateddata = rotateddata.copy('system')
        rotarr = np.load("data/groupdiagcol3.npy")
        np.testing.assert_array_equal(rotarr, np.array(rotateddata))
        rotateddata = rotateddata.copy('cuda')
        # GROUP DIAG ROWS
        groupdiagrows(rotateddata,rotgrid)
        rotateddata = rotateddata.copy('system')
        rotarr = np.load("data/groupdiagrow2.npy")
        np.testing.assert_array_equal(rotarr,np.array(rotateddata))
        rotateddata = rotateddata.copy('cuda')
        # GROUP DIAG COLS
        groupdiagcols(rotateddata,rotgrid)
        rotateddata = rotateddata.copy('system')
        rotarr = np.load("data/groupdiagcol4.npy")
        np.testing.assert_array_equal(rotarr,np.array(rotateddata))
        rotateddata = rotateddata.copy('cuda')
        # UPDATE UNROTATED Island data
        BFMap("""isldata(time,chan,stokes,i,j) = rm(time,chan,stokes,%i-i+j-1,i+j-2);""" % (ngrid,),
            {'isldata':islanddata,'rm':rotateddata},
            axis_names = ('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))
        islanddata = islanddata.copy("system")
        islarr = np.load("data/finalislands.npy")
        np.testing.assert_array_equal(islarr,np.array(islanddata))
        
    
