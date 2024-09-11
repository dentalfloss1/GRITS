import matplotlib.pyplot as plt
import numpy as np
from lsl.common.mcs import mjdmpm_to_datetime
from GRITS.catalogs import cat3c,VLSSr50jy
from GRITS.imagetools import angsep, hdrtoworld,hdrtopixmask,pbcorr
from GRITS.readimage import readoims, readpims
import matplotlib.colors as colors
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
def plotsource(index, plotfile, filename,  ra, dec, rms, corrfluxlist,channel,vmax=8000):
    rms = rms[index,channel]

    if "oims" in filename:
        ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata,station = readoims(filename) 
    elif "pims" in filename:
        ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata,station = readpims(filename) 
    else:
        print("Unknown filetype")
        exit()
    mycorr = corrfluxlist[channel] 
    imagedata = imagedata[:,channel,:,:,:]
    data = data[:,channel,:,:,:]
    data  = np.copy(data[:,:,:,:])*mycorr
    imagedata  = np.copy(imagedata[:,:,:,:])*mycorr
    srcloc = hdrtopixmask(hdrlist[index], ngrid, sRad, ra,dec,station)
    avx,avy = srcloc[0]
    extent = SetExtent(sRad,data)
    fig, axs = plt.subplots(3,5,figsize=(30,15))
    nchan = data.shape[1]
    currimage = data[index,0,:,:]
    # Stokes I image
    axs[0,0].imshow(currimage,vmin=0,vmax=vmax*50,origin='lower',extent=extent)
    currimage = imagedata[index,0,:,:]
    # Stokes V image
    axs[0,1].imshow(currimage,vmin=-vmax,vmax=vmax,origin='lower',extent=extent,cmap="coolwarm")
    mintime = max(0,index-2)
    maxtime = min(ints-1,index+2)
    # Series of Stokes I diff image
    for timenum, t in enumerate(range(mintime,maxtime+1)):
        currimage = data[t,0,:,:]
        axs[1,timenum].imshow(currimage,vmin=-vmax,vmax=vmax,origin='lower')
        axs[1,timenum].axvline(avx,color='black',alpha=0.25)
        axs[1,timenum].axhline(avy,color='black',alpha=0.25)
    # Series of Stokes V image
    for timenum, t in enumerate(range(mintime,maxtime+1)):
        currimage = data[t,3,:,:]
        axs[2,timenum].imshow(currimage,vmin=-vmax,vmax=vmax,origin='lower',cmap='coolwarm')
        axs[2,timenum].axvline(avx,color='black',alpha=0.25)
        axs[2,timenum].axhline(avy,color='black',alpha=0.25)

    hdr = hdrlist[index]
    mjd = int(hdr['start_time'])
    mpm = int((hdr['start_time'] - mjd)*86400.0*1000.0)
    dateObs = mjdmpm_to_datetime(mjd, mpm)
    fig.suptitle(f"{dateObs.isoformat()},rms:{rms:.2E}")
    plt.savefig(plotfile)
    plt.close()
     


    

    
def plotresults(i,channel,detstokes,filename, rms,islands,imagedata,data,hdr,sRad,station,catsep=5,chunkt=0):
    vmax = rms*3
    ints = data.shape[0]
    ngrid = data.shape[-1]
    skip = False
    extent = SetExtent(sRad,data)
    chantoplot = 0
    fig, axs = plt.subplots(ncols=5,nrows=6,figsize=(40,35))
    ytextloc = 0.9
    lastlabel = -1
    mintime = max(1,i-2)
    maxtime = min(ints-1,mintime+4)
    for stokesno,sname in enumerate(['I','Q','U','V']):
        for timenum, t in enumerate(range(mintime,maxtime+1)):
            # if (i==mintime):
            #     currimage = imagedata[t,stokesno,:,:]-imagedata[maxtime+1,stokesno,:,:]
            # else:
            #     currimage = imagedata[t,stokesno,:,:]-imagedata[mintime-1,stokesno,:,:]
            if stokesno>0:
                currimage = np.abs(imagedata[t,stokesno,:,:])-np.abs(imagedata[t-1,stokesno,:,:])
            else:
                currimage = imagedata[t,stokesno,:,:]-imagedata[t-1,stokesno,:,:]
            if stokesno>0:
                cmap='coolwarm'
            else:
                cmap='viridis'
            axs[stokesno+2,timenum].imshow(currimage,vmin=-vmax,vmax=vmax,origin='lower',cmap=cmap)
            axs[stokesno+2,timenum].tick_params(axis='both',which='both',bottom=False,top=False,labelbottom=False)
            if t==i:
                axs[stokesno+2,timenum].set_title(f'{sname} diff')
            # axs[0,timenum].imshow(data[t,chantoplot,0,:,:],vmin=0,vmax=vmax/100,origin='lower',extent=extent)
    uniquelabel =1
    for s in [0,1,2,3]:
        for label in np.unique(islands[i,s,:,:][(~np.isnan(islands[i,s,:,:])) & ( islands[i,s,:,:] >0)]).astype(int):
            yloc, xloc = np.where(islands[i,s,:,:]==label)
            if yloc.size==0:
                continue
            for x,y in zip(xloc,yloc):
                if data[i,s,y,x]==np.amax(data[i,s,yloc,xloc]):
                    avx = x
                    avy = y 
                    break
            radec = hdrtoworld(hdr,ngrid,sRad,avx,avy,station)

            XX,YY = pbcorr(hdr,ngrid,channel,hdr['psize'], station)
            pbdata = data[i,0,:,:]/((XX+YY)/2)
            measI = pbdata[avy,avx]
            measQ, measU, measV = data[i,1:4,int(avy),int(avx)]
            resulttext =  f"{channel},{s},{uniquelabel}, pos:{np.round(radec,2)},{(round(avx,0),round(avy,0))} I:{np.nan_to_num(measI):.2e},Q:{int(round(100*measQ/measI))}%,U:{int(round(100*measU/measI))}%,V:{int(round(100*measV/measI))}%"
            axs[0,2].text(0,ytextloc,resulttext)
            ytextloc-=0.15
            for anum in range(2,6,1):
                for imind in range(5):
                    axs[anum,imind].annotate(str(uniquelabel),(avx+5,avy+5),alpha=0.25)
                    circle = plt.Circle((avx,avy),5,fill=False)
                    axs[anum,imind].add_artist(circle)
            uniquelabel+=1
    for axnum in range(2,5):
        axs[0,axnum].set_axis_off()

    islandplot = islands[i,0,:,:]
    axs[1,0].imshow(islandplot,vmin=0,vmax=1,origin='lower',cmap='coolwarm')
    axs[1,0].set_title('I src')
    islandplot = islands[i,1,:,:]
    axs[1,1].imshow(islandplot,vmin=0,vmax=1,origin='lower',cmap='coolwarm')
    axs[1,1].set_title('Q src')
    islandplot = islands[i,2,:,:]
    axs[1,2].imshow(islandplot,vmin=0,vmax=1,origin='lower',cmap='coolwarm')
    axs[1,2].set_title('U src')
    islandplot = islands[i,3,:,:]
    axs[1,3].imshow(islandplot,vmin=0,vmax=1,origin='lower',cmap='coolwarm')
    axs[1,3].set_title('V src')
    currimage = imagedata[i,0,:,:]
    axs[0,0].imshow(currimage,origin='lower',extent=extent)
    #axs[0,0].imshow(currimage,vmin=0,vmax=vmax*5,origin='lower',extent=extent)
    currimage = imagedata[i,3,:,:]
    #axs[0,1].imshow(currimage,vmin=-vmax,vmax=vmaxy,origin='lower',extent=extent,cmap="coolwarm")
    axs[0,1].imshow(currimage,origin='lower',extent=extent,cmap="coolwarm")
    mjd = int(hdr['start_time'])
    mpm = int((hdr['start_time'] - mjd)*86400.0*1000.0)
    # determine if header is in seconds or days
    if round(hdr['int_len']) > 0:
        tInt = hdr['int_len']
    else:
        tInt = hdr['int_len']*86400.0
    dateObs = mjdmpm_to_datetime(mjd, mpm)
    nomaskimdiff = imagedata[i] - imagedata[i-1]
    s2n = np.amax(np.abs(nomaskimdiff[detstokes]))/rms
    fig.suptitle(f"{dateObs.isoformat()},rms:{rms:.2E},max S2N:{s2n}")
    plotfile = filename.split('/')[-1].replace(".oims",f"-t{i+chunkt}-c{channel}.png").replace(".pims",f"-t{i+chunkt}.png").replace(':','-')
    plt.savefig(plotfile)
    plt.close()


def plotLC(filename,plotdata,zoomplotdata,dettimechan,srcid,timeind,tdet,ra,dec):
    plotfile = filename.split('/')[-1].replace(".oims",f"-srcid{srcid}_t{timeind}_LC.png").replace(".pims",f"-src{srcid}_t{timeind}_LC.png").replace(':','-')
    nchan = len(np.unique(plotdata['c']))
    if nchan==1:
        fig,axs = plt.subplots(nrows=1,ncols=2, figsize=(50,16))
        unflag = plotdata[plotdata['flagged']==False]
        flagged = plotdata[plotdata['flagged']==True]
        statmatched = plotdata[(plotdata['statmatch']==True)]
        axs[0].scatter(unflag['hrpst'],unflag['flux'],color='black',marker='s')
        axs[0].scatter(flagged['hrpst'],flagged['flux'],color='black',marker='.',label="flagged",alpha=0.25)
        if statmatched.size >0:
            unflag = statmatched[statmatched['flagged']==False]
            flagged = statmatched[statmatched['flagged']==True]
            axs[0].scatter(unflag['hrpst'],unflag['flux'],color='black',marker='*',label='Matching measurement',s=100)
            axs[0].scatter(flagged['hrpst'],flagged['flux'],color='black',marker='*',label='Matching measurement',s=100,alpha=0.25)
        axs[0].legend()
        if unflag['flux'].size > 0:
            ymin = unflag['flux'].min()
            ymax = unflag['flux'].max()
        else:
            ymin = plotdata['flux'].min()
            ymax = plotdata['flux'].max()
        axs[0].set_ylim(ymin,ymax)
        statmatched = zoomplotdata[(zoomplotdata['statmatch']==True) ]
        unflag = zoomplotdata[zoomplotdata['flagged']==False]
        flagged = zoomplotdata[zoomplotdata['flagged']==True]
        axs[1].scatter(unflag['hrpst'],unflag['flux'],color='black',marker='s')
        axs[1].scatter(flagged['hrpst'],flagged['flux'],color='black',marker='.',label="flagged",alpha=0.25)
        if statmatched.size >0:
            unflag = statmatched[statmatched['flagged']==False]
            flagged = statmatched[statmatched['flagged']==True]
            axs[1].scatter(unflag['hrpst'],unflag['flux'],color='black',marker='*',label='Matching measurement',s=100)
            axs[1].scatter(flagged['hrpst'],flagged['flux'],color='black',marker='*',label='Matching measurement',s=100,alpha=0.25)
        axs[1].legend()
        for ax in axs:
            for d in dettimechan:
                detloc = plotdata['hrpst'][plotdata['index']==d['index']][0]
                ax.axvline(detloc,alpha=0.5,color='black')
        axs[0].set_xlabel("Hours past midnight")
        axs[1].set_xlabel("Hours past midnight")
        axs[0].set_ylabel("Approximate Flux (Jy)")
        fig.suptitle(f"source at {ra:5.2f}, {dec:5.2f}\ndetected at {tdet}")
        plt.tight_layout()
        plt.savefig(plotfile)
        plt.close()
    else:
        fig, axs = plt.subplots(nrows=nchan+1,ncols=2,figsize=(50,12*nchan))
        for c,ax in zip(np.unique(plotdata['c']),axs.flatten()[:(2*nchan):2]):
            freq = plotdata['freq'][plotdata['c']==c][0]
            curdata = plotdata[plotdata['c']==c]
            try:
                fluxdat = curdata['flux'] # - np.nanmedian(curdata['flux'])
            except RuntimeWarning:
                continue
            unflag = curdata[curdata['flagged']==False]
            flagged = curdata[curdata['flagged']==True]
            statmatched = plotdata[(plotdata['statmatch']==True) & (plotdata['c']==c)]
            ax.scatter(unflag['hrpst'],unflag['flux'],color='black',marker='s')
            ax.scatter(flagged['hrpst'],flagged['flux'],color='black',marker='.',label='flagged',alpha=0.25)
            if statmatched.size >0:
                unflag = statmatched[statmatched['flagged']==False]
                flagged = statmatched[statmatched['flagged']==True]
                ax.scatter(unflag['hrpst'],unflag['flux'],color='black',marker='*',label='Matching measurement',s=100)
                ax.scatter(flagged['hrpst'],flagged['flux'],color='black',marker='*',label='Matching measurement',s=100,alpha=0.25)
            ax.legend()
            ax.set_xlim(plotdata['hrpst'].min(), plotdata['hrpst'].max())
            if unflag['flux'].size > 0:
                ymin = unflag['flux'].min()
                ymax = unflag['flux'].max()
            else:
                ymin = plotdata['flux'].min()
                ymax = plotdata['flux'].max()
            ax.set_ylim(ymin,ymax)
            ax.set_title(f"{freq:5.2f} MHz")
            if ax==axs.flatten()[-4]:
                ax.set_xlabel(f"Hours past midnight")
                # ax.set_ylabel("Flux above median (Jy)")
                ax.set_ylabel("Approx. Flux (Jy)")
            curdets = dettimechan[dettimechan['c']==c]
            for c in curdets:
                if c['index'] in plotdata['index']:
                    detloc = plotdata['hrpst'][plotdata['index']==c['index']][0]
                    ax.axvline(detloc,alpha=0.5,color='black')
        for c,ax in zip(np.unique(zoomplotdata['c']),axs.flatten()[1:(2*nchan):2]):
            freq = zoomplotdata['freq'][zoomplotdata['c']==c][0]
            curdata = zoomplotdata[zoomplotdata['c']==c]
            unflag = curdata[curdata['flagged']==False]
            flagged = curdata[curdata['flagged']==True]
            xdata = unflag['hrpst']
            ydata = unflag['flux']
            ax.scatter(xdata,ydata,color='black',marker='s')
            xdata = flagged['hrpst']
            ydata = flagged['flux']
            ax.scatter(xdata,ydata,color='black',marker='.',label='flagged',alpha=0.25)
            statmatched = zoomplotdata[(zoomplotdata['statmatch']==True) & (zoomplotdata['c']==c)]
            if statmatched.size >0:
                unflag = statmatched[statmatched['flagged']==False]
                flagged = statmatched[statmatched['flagged']==True]
                ax.scatter(unflag['hrpst'],unflag['flux'],color='black',marker='*',label='Matching measurement',s=100)
                ax.scatter(flagged['hrpst'],flagged['flux'],color='black',marker='*',label='Matching measurement',s=100,alpha=0.25)
            ax.legend()
            ax.set_xlim(zoomplotdata['hrpst'].min(), zoomplotdata['hrpst'].max())
            ax.set_title(f"{freq:5.2f} MHz")
            if ax==axs.flatten()[-3]:
                ax.set_xlabel(f"Hours past midnight")
                # ax.set_ylabel("Flux above median (Jy)")
                ax.set_ylabel("Approx. Flux (Jy)")
            curdets = dettimechan[dettimechan['c']==c]
            for c in curdets:
                if c['index'] in zoomplotdata['index']:
                    detloc = zoomplotdata['hrpst'][zoomplotdata['index']==c['index']][0]
                    ax.axvline(detloc,alpha=0.5,color='black')
        ax = axs.flatten()[-2]
        for d in np.unique(dettimechan['index']):
            curdata = plotdata[plotdata['index']==d]
            ax.plot(curdata['freq'],curdata['flux'],label=d,marker='x')
        ax.set_xlabel('Freq (MHz)')
        ax.set_ylabel('Approx. Flux (Jy)')
        ax.legend()
        ax = axs.flatten()[-1]
        
        Z = np.zeros((len(np.unique(zoomplotdata['hrpst'])),nchan))
        sortuhrpsts = np.sort(np.unique(zoomplotdata['hrpst']))
        for i,d in enumerate(sortuhrpsts):
            for c in np.unique(zoomplotdata['c']):
                Z[i,c] = zoomplotdata['flux'][(zoomplotdata['hrpst']==d) & (zoomplotdata['c']==c)][0]
    #             try:
    #                 Z[i,c] = np.log10(thisZ) # - np.nanmedian(thisZ)
    #             except RuntimeWarning:
    #                 continue
    #     Z[np.isnan(Z)] = 0
    #     # setvscale = 3*np.median(Z[Z>0])
    #     # setvscale = np.amax(np.abs(Z))
        waterfall = ax.imshow(Z.transpose(),cmap='viridis',origin='lower')
        cbar = fig.colorbar(waterfall,ax=ax, orientation='horizontal', cmap='coolwarm',fraction=0.1)
    #     cbar.set_label('Approx. Log10 Flux (Jy)')
        plt.subplots_adjust(hspace=0.3)
        # locs,labels = plt.xticks()
        # print(locs,labels)

        detind =  np.unique(dettimechan['index'])[0]
        dettime =  zoomplotdata['hrpst'][zoomplotdata['index']==detind][0]

       #  labels[detind].text = 'detection'
        labels = []
        locs = []
        for tnum,t in enumerate(np.sort(np.unique(zoomplotdata['hrpst']))):
           if (tnum%4==0) or ((t-dettime)==0):
               locs.append(tnum)
               labels.append( f'{(t-dettime)*3600:5.0f}')
        ax.set_xticks(locs,labels)
        ax.set_xlabel("seconds since detection")
        fig.suptitle(f"source at {ra:5.2f}, {dec:5.2f}\ndetected at {tdet}")
        plt.savefig(plotfile)
        plt.close()
     



