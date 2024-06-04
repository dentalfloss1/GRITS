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

if __name__=='__main__':
    islanddata = genimagecube()
    islanddata = BFArray(islanddata, space='cuda')
 #   plotislands(islanddata)
    labelgroups(islanddata, ngrid, ints,nchan)
    islanddata = islanddata.copy('system')
    islarr = np.load("label1.npy")
    time,chan,stokes,y,x = np.where(islarr!=np.array(islanddata))
    if time.size>0:
        for t,yp,xp in zip(time,y,x):
            print(t,yp,xp)
            plottwoisland(np.array(islanddata[time,chan,stokes][0]),islarr[time,chan,stokes][0])
    np.testing.assert_array_equal(islarr, np.array(islanddata))
#     islanddata = islanddata.copy('cuda')
#     # GROUPCOLS
#     groupvertcols(islanddata,ngrid)
#     islanddata = islanddata.copy('system')
#     islarr = np.load("groupvert1.npy")
#     time,chan,stokes,y,x = np.where(islarr!=np.array(islanddata))
#     if time.size>0:
#         for t,yp,xp in zip(time,y,x):
#             print(t,yp,xp)
#             plottwoisland(np.array(islanddata[time,chan,stokes][0]),islarr[time,chan,stokes][0])
#     np.testing.assert_array_equal(islarr, np.array(islanddata))
#     # plottwoisland(np.array(islanddata),islarr)
#     islanddata = islanddata.copy('cuda')
#     # GROUPROWS
#     grouphorizrows(islanddata,ngrid)
#     islanddata = islanddata.copy('system')
#     islarr = np.load('grouprow1.npy')
#     np.testing.assert_array_equal(islarr, np.array(islanddata))
    # islarr = np.load("groupvert1.npy")
#     # GROUPCOLS
#     groupvertcols(islanddata,ngrid)
#     # ROTATE Island data
#     rotateddata = BFArray(np.zeros((ints,nchan,4,2*ngrid-1,2*ngrid-1)),space='cuda',dtype='i8')
#     BFMap("""rm(time,chan,stokes,i+j-2,%i-i+j-1) = isldata(time,chan,stokes,i,j);""" % (ngrid,),
#         {'isldata':islanddata,'rm':rotateddata},
#         axis_names = ('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))
#     rotgrid = 2*ngrid-1
# 
#     # GROUP DIAG COLS
#     groupdiagcols(rotateddata,rotgrid)
#     # GROUP DIAG ROWS
#     groupdiagrows(rotateddata,rotgrid)
#     # GROUP DIAG COLS
#     groupdiagcols(rotateddata,rotgrid)
#     # UPDATE UNROTATED Island data
#     BFMap("""isldata(time,chan,stokes,i,j) = rm(time,chan,stokes,i+j-2,%i-i+j-1);""" % (ngrid,),
#         {'isldata':islanddata,'rm':rotateddata},
#         axis_names = ('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))
#     # GROUPCOLS
#     groupvertcols(islanddata,ngrid)
#     # GROUPROWS
#     grouphorizrows(islanddata,ngrid)
#     # GROUPCOLS
#     groupvertcols(islanddata,ngrid)
#     # ROTATE Island data
#     rotateddata = BFArray(np.zeros((ints,nchan,4,2*ngrid-1,2*ngrid-1)),space='cuda',dtype='i8')
#     BFMap("""rm(time,chan,stokes,i+j-2,%i-i+j-1) = isldata(time,chan,stokes,i,j);""" % (ngrid,),
#         {'isldata':islanddata,'rm':rotateddata},
#         axis_names = ('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))
#     rotgrid = 2*ngrid-1
# 
#     # GROUP DIAG COLS
#     groupdiagcols(rotateddata,rotgrid)
#     # GROUP DIAG ROWS
#     groupdiagrows(rotateddata,rotgrid)
#     # GROUP DIAG COLS
#     groupdiagcols(rotateddata,rotgrid)
#     # UPDATE UNROTATED Island data
#     BFMap("""isldata(time,chan,stokes,i,j) = rm(time,chan,stokes,i+j-2,%i-i+j-1);""" % (ngrid,),
#         {'isldata':islanddata,'rm':rotateddata},
#         axis_names = ('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))
#     islanddata = islanddata.copy('system')
    

