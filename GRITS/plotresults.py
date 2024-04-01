import matplotlib.pyplot as plt
import numpy as np
from lsl.common.mcs import mjdmpm_to_datetime
from GRITS.catalogs import cat3c,VLSSr50jy
from GRITS.imagetools import angsep, hdrtoworld

def SetExtent(sRad, data):
    #### Set up extent for plot ## 
    ngrid = data.shape[-1]
    pCntr = ngrid/2 + 1 + 0.5 * ((ngrid+1)%2)
    x = np.arange(1, data.shape[-2]+1, dtype=np.float32) - pCntr
    y = np.arange(1, data.shape[-1]+1, dtype=np.float32) - pCntr
    x /= -sRad
    y /= sRad
    x,y = np.meshgrid(x,y)
    x2 = x - 1 / sRad
    y2 = y - 1 / sRad
    extent = (x2.max(), x2.min(), y.min(), y.max())
    return extent
def plotresults(i,filename, mean,std,rms,dynamicrange,islands,imagedata,data,hdr,sRad,length,width,catsep=5,vmax=8000,chunkt=0):
    ints = data.shape[0]
    nchan = data.shape[1]
    ngrid = data.shape[-1]
    skip = False
    extent = SetExtent(sRad,data)
    chantoplot = 0
    fig, axs = plt.subplots(3,5,figsize=(30,15))
    ytextloc = 0.9
    for label in range(0,int(np.nanmax(islands[i,:,:,:,:]))+1):
        for c in range(nchan):
            for s in [0,3]:
                srclen = length[i,c,s,0,label-1,0]
                srcwid = width[i,c,s,label-1]
                srclocs = np.argwhere(islands[i,c,s,:,:]==label)
                if ((srclen > 0) and (srcwid >0)):
                    avx, avy = np.average(srclocs+0.5,axis=0)
                    radec = hdrtoworld(hdr,ngrid,sRad,avx,avy)
                    axs[1,4].text(0,ytextloc,f"{c},{s},{label},({length[i,c,s,0,label-1,0]},{width[i,c,s,label-1]}), pos:{np.round(radec,2)}")
                    
                    ytextloc -= 0.15
                    VLSSrclose = np.argmin(angsep(VLSSr50jy['ra'],VLSSr50jy['dec'],radec[0][0],radec[0][1]))        
                    VLSSrnum = VLSSr50jy['sourcenum'][VLSSrclose] 
                    VLSSrsep = np.amin(angsep(VLSSr50jy['ra'],VLSSr50jy['dec'],radec[0][0],radec[0][1]))     
                    if VLSSrsep < catsep: 
                        axs[1,4].text(0,ytextloc,f"VLSSr #{VLSSrnum} is {round(VLSSrsep,1)} deg away")
                        ytextloc -= 0.15
                    cat3cclose = np.argmin(angsep(cat3c['ra'],cat3c['dec'],radec[0][0],radec[0][1]))
                    cat3cnum = cat3c['3C'][cat3cclose]
                    cat3csep = np.amin(angsep(cat3c['ra'],cat3c['dec'],radec[0][0],radec[0][1]))
                    if cat3csep < catsep:
                        axs[1,4].text(0,ytextloc,f"3C{cat3cnum} is {round(cat3csep,1)} deg away")
                        ytextloc -= 0.15

    axs[1,4].set_axis_off()

    VLSSrclose = np.argmin(angsep(VLSSr50jy['ra'],VLSSr50jy['dec'],radec[0][0],radec[0][1]))        
    VLSSrname = VLSSr50jy['name'][VLSSrclose] 
    VLSSrsep = np.amin(angsep(VLSSr50jy['ra'],VLSSr50jy['dec'],radec[0][0],radec[0][1]))     
    islandplot = np.zeros((ngrid,ngrid))
    for c in range(nchan):
        islandplot+=islands[i,c,0]
    axs[1,2].imshow(islandplot,vmin=0,vmax=1,origin='lower',extent=extent,cmap='coolwarm')
    axs[1,2].set_title('I src')
    islandplot = np.zeros((ngrid,ngrid))
    for c in range(nchan):
        islandplot+=islands[i,c,3]
    axs[1,3].imshow(islandplot,vmin=0,vmax=1,origin='lower',extent=extent,cmap='coolwarm')
    axs[1,3].set_title('V src')
    mintime = max(0,i-2)
    maxtime = min(ints-1,i+2)
    for timenum, t in enumerate(range(mintime,maxtime+1)):
        currimage = np.zeros(data[t,chantoplot,0,:,:].shape)
        for c in range(nchan):
            currimage += data[t,c,0,:,:]
        currimage/=nchan
        axs[0,timenum].imshow(currimage,vmin=-vmax/50,vmax=vmax/50,origin='lower',extent=extent)
        # axs[0,timenum].imshow(data[t,chantoplot,0,:,:],vmin=0,vmax=vmax/100,origin='lower',extent=extent)
    for timenum, t in enumerate(range(mintime,maxtime+1)):
        currimage = np.zeros(data[t,chantoplot,3,:,:].shape)
        for c in range(nchan):
            currimage += data[t,c,3,:,:]
        currimage/=nchan
        axs[2,timenum].imshow(currimage,vmin=-vmax/50,vmax=vmax/50,origin='lower',extent=extent,cmap='coolwarm')
        # axs[0,timenum].imshow(data[t,chantoplot,0,:,:],vmin=0,vmax=vmax/100,origin='lower',extent=extent)
    currimage = np.zeros(imagedata[i,chantoplot,0,:,:].shape)
    for c in range(nchan):
        currimage += imagedata[i,c,0,:,:]
    currimage/=nchan
    axs[1,0].imshow(currimage,vmin=0,vmax=vmax,origin='lower',extent=extent)
    currimage = np.zeros(imagedata[i,chantoplot,0,:,:].shape)
    for c in range(nchan):
        currimage += imagedata[i,c,3,:,:]
    currimage/=nchan
    axs[1,1].imshow(currimage,vmin=-vmax/50,vmax=vmax/50,origin='lower',extent=extent,cmap="coolwarm")
    mjd = int(hdr['start_time'])
    mpm = int((hdr['start_time'] - mjd)*86400.0*1000.0)
    # determine if header is in seconds or days
    if round(hdr['int_len']) > 0:
        tInt = hdr['int_len']
    else:
        tInt = hdr['int_len']*86400.0
    dateObs = mjdmpm_to_datetime(mjd, mpm)
    fig.suptitle(f"{dateObs.isoformat()},std:{std:.2E},mean:{mean:.2E},rms:{rms:.2E},sn:{dynamicrange:.2E}")
    plotfile = filename.replace(".oims",f"-{i+chunkt}.png").replace(".pims",f"-{i+chunkt}.png").replace(':','-')
    plt.savefig(plotfile)
    print("saving file as ",plotfile)
    plt.close()
