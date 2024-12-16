#!/usr/bin/env python
import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal as signal 
import glob
from tqdm import tqdm
from GRITS.catalogs import cat3c,VLSSr50jy,atnf
from GRITS.imagetools import angsep
import imageio
for fno,filename in enumerate(glob.glob("*fluxfile*npz")):
    alldata = np.load(filename)
    dat1 = alldata['imageflux1']
    diffdat1 = alldata['diffFLUX1']
    dat2 = alldata['imageflux2']
    diffdat2 = alldata['diffFLUX2']
    pixmeas1 = alldata['pixmeas1']
    pixmeas2 = alldata['pixmeas2']
    srcarray = alldata['srcarray']
    station1 = srcarray['station'][0]
    station2 = srcarray['station'][srcarray['station']!=station1][0]
    detstokes = srcarray['stokes'][0]
    for sno,(dat, diffdat,pixmeas,stationname) in enumerate(zip([dat1,dat2],[diffdat1,diffdat2],[pixmeas1,pixmeas2],[station1,station2])):
        dettime = srcarray['mjd'][srcarray['station']==stationname]
        plotfile = filename.replace(".npz",f"_station{sno+1}_multiplot.png")
        nchan = len(np.unique(diffdat['freq']))
        if nchan==1:
           fig,axs = plt.subplots(nrows=4,ncols=3, figsize=(50,16))
           fig.suptitle(f"{diffdat['freq'][0]} MHz")
           for stokesno, (stokes,marker) in enumerate(zip(['I','Q','U','V'],['s','o','*','D'])):
               xdata = dat['mjd'][dat['stokes']==stokesno]
               ydata = dat['f0'][dat['stokes']==stokesno]
               yerr = dat['f0err'][dat['stokes']==stokesno]
               axs[stokesno,0].errorbar(xdata,ydata,yerr=yerr,marker=marker,ls='none',color='black',label=stokes)
               detpos = diffdat[['ra','dec']][np.isin(diffdat['mjd'],dettime) & (diffdat['stokes']==stokesno)]
               xdiffdata = diffdat['mjd'][diffdat['stokes']==stokesno]
               ydiffdata = diffdat['f0'][diffdat['stokes']==stokesno]
               ydifferr = diffdat['f0err'][diffdat['stokes']==stokesno]
               axs[stokesno,1].errorbar(xdiffdata,ydiffdata,yerr=ydifferr,marker=marker,ls='none',color='black',label=stokes)
               xpixdata = pixmeas['mjd'][pixmeas['stokes']==stokesno]
               ypixdata = pixmeas['f0'][pixmeas['stokes']==stokesno]
               ypixerr = pixmeas['f0err'][pixmeas['stokes']==stokesno]
               axs[stokesno,2].scatter(xpixdata,ypixdata,marker=marker,color='black')
               for d in dettime:
                   if d in xdata:
                       colno=0
                       axs[stokesno,colno].axvline(d)
                   if d in xdiffdata:
                       colno=1
                       axs[stokesno,colno].axvline(d)
                   colno=2
                   axs[stokesno,colno].axvline(d)
               axs[stokesno,2].title.set_text(f"Pixel values stokes {stokes}")
               axs[stokesno,1].title.set_text(f"Diff stokes {stokes} {detpos}")
               axs[stokesno,0].title.set_text(f"Image stokes {stokes}")
               axs[stokesno,0].set_ylabel("Flux (Jy)")
               axs[stokesno,1].set_ylabel("Diff. Flux (Jy)")
               axs[stokesno,2].set_ylabel("Pixel Flux (Jy)")
               axs[stokesno,0].legend()
               axs[stokesno,1].legend()
           axs[3,0].set_xlabel("date (MJD)")
           axs[3,1].set_xlabel("date (MJD)")
           axs[3,2].set_xlabel("date (MJD)")
           plt.tight_layout()
           plt.savefig(plotfile)
           plt.close()
        else:
            for stokesno, (stokes,marker) in enumerate(zip(['I','Q','U','V'],['s','o','*','D'])):
                fig, axs = plt.subplots(nrows=nchan+1,ncols=3,figsize=(50,12*nchan))
                for c,ax in zip(np.unique(dat['freq']),axs.flatten()[:(3*nchan):3]):
                    freq = c
                    curdat = dat[(dat['freq']==freq) ]
                    xdata = curdat['mjd'][curdat['stokes']==stokesno]
                    ydata = curdat['f0'][curdat['stokes']==stokesno]
                    yerr = curdat['f0err'][curdat['stokes']==stokesno]
                    ax.errorbar(xdata,ydata,yerr=yerr,marker=marker,ls='none',color='black',label=stokes)
                    ax.set_title(f"{freq:5.2f} MHz stokes {stokes} Image Flux")
                    ax.legend()
                    for d in dettime:
                        if d in xdata:
                            for colno in range(3):
                                ax.axvline(d)
                    if ax==axs.flatten()[-4]:
                        ax.set_xlabel(f"Date (MJD)")
                        # ax.set_ylabel("Flux above median (Jy)")
                        ax.set_ylabel("Flux (Jy)")
    
                for c,ax in zip(np.unique(diffdat['freq']),axs.flatten()[1:(3*nchan):3]):
                    freq = c
                    curdat = diffdat[(diffdat['freq']==freq)]

                    detpos = curdat[['ra','dec']][np.isin(curdat['mjd'],dettime) & (curdat['stokes']==stokesno)]
                    xdata = curdat['mjd'][curdat['stokes']==stokesno]
                    ydata = curdat['f0'][curdat['stokes']==stokesno]
                    yerr = curdat['f0err'][curdat['stokes']==stokesno]
                    ax.errorbar(xdata,ydata,yerr=yerr,marker=marker,ls='none',color='black',label=stokes)
                    ax.set_title(f"{freq:5.2f} MHz Diff Flux {detpos}")
                    ax.legend()
                    for d in dettime:
                        if d in xdata:
                            for colno in range(3):
                                ax.axvline(d)
                    if ax==axs.flatten()[-4]:
                        ax.set_xlabel(f"Date (MJD)")
                        # ax.set_ylabel("Flux above median (Jy)")
                        ax.set_ylabel("Diff Flux (Jy)")

                for c,ax in zip(np.unique(pixmeas['freq']),axs.flatten()[2:(3*nchan):3]):
                    freq = c
                    curdat = pixmeas[(pixmeas['freq']==freq)]
                    xdata = curdat['mjd'][curdat['stokes']==stokesno]
                    ydata = curdat['f0'][curdat['stokes']==stokesno]
                    ax.scatter(xdata,ydata,marker=marker,color='black',label=stokes)
                    ax.set_title(f"{freq:5.2f} MHz Pixel Flux")
                    ax.legend()
                    for d in dettime:
                        for colno in range(3):
                            ax.axvline(d)
                    if ax==axs.flatten()[-4]:
                        ax.set_xlabel(f"Date (MJD)")
                        # ax.set_ylabel("Flux above median (Jy)")
                        ax.set_ylabel("Pixel Flux (Jy)")
    
                ax = axs.flatten()[-3]
                Z = np.zeros((len(np.unique(dat['mjd'])),nchan))
                sortuhrpsts = np.sort(np.unique(dat['mjd'][dat['mjd']!=0]))
                for i,d in enumerate(sortuhrpsts):
                    for chno, c in enumerate(np.unique(np.sort(dat['freq']))):
                        try:
                            Z[i,chno] = dat['f0'][(dat['mjd']==d) & (dat['freq']==c) & (dat['stokes']==stokesno)][0]
                        except: 
                            pass 
    #                     try:
    #                         Z[i,c] = np.log10(thisZ) # - np.nanmedian(thisZ)
    #                     except RuntimeWarning:
    #                         continue
    #             Z[np.isnan(Z)] = 0
    #             # setvscale = 3*np.median(Z[Z>0])
    #             # setvscale = np.amax(np.abs(Z))
                waterfall = ax.imshow(Z.transpose(),cmap='viridis',origin='lower',aspect='auto')
                cbar = fig.colorbar(waterfall,ax=ax, orientation='horizontal', cmap='coolwarm',fraction=0.1)
    #             cbar.set_label('Approx. Log10 Flux (Jy)')
                plt.subplots_adjust(hspace=0.3)
                # locs,labels = plt.xticks()
                # print(locs,labels)
                # fig.suptitle(f"source at {ra:5.2f}, {dec:5.2f}\ndetected at {tdet}")
                ax = axs.flatten()[-2]
                Z = np.zeros((len(np.unique(diffdat['mjd'])),nchan))
                sortuhrpsts = np.sort(np.unique(diffdat['mjd'][diffdat['mjd']!=0]))
                for i,d in enumerate(sortuhrpsts):
                    for chno, c in enumerate(np.unique(np.sort(diffdat['freq']))):
                        try:
                            Z[i,chno] = diffdat['f0'][(diffdat['mjd']==d) & (diffdat['freq']==c) & (diffdat['stokes']==stokesno)][0]
                        except: 
                            pass 
    #                     try:
    #                         Z[i,c] = np.log10(thisZ) # - np.nanmedian(thisZ)
    #                     except RuntimeWarning:
    #                         continue
    #             Z[np.isnan(Z)] = 0
    #             # setvscale = 3*np.median(Z[Z>0])
    #             # setvscale = np.amax(np.abs(Z))
                waterfall = ax.imshow(Z.transpose(),cmap='viridis',origin='lower',aspect='auto')
                cbar = fig.colorbar(waterfall,ax=ax, orientation='horizontal', cmap='coolwarm',fraction=0.1)
    #             cbar.set_label('Approx. Log10 Flux (Jy)')
                plt.subplots_adjust(hspace=0.3)
                # locs,labels = plt.xticks()
                # print(locs,labels)
                # fig.suptitle(f"source at {ra:5.2f}, {dec:5.2f}\ndetected at {tdet}")
                ax = axs.flatten()[-1]
                Z = np.zeros((len(np.unique(pixmeas['mjd'])),nchan))
                sortuhrpsts = np.sort(np.unique(pixmeas['mjd'][pixmeas['mjd']!=0]))
                for i,d in enumerate(sortuhrpsts):
                    for chno, c in enumerate(np.unique(np.sort(pixmeas['freq']))):
                        try:
                            Z[i,chno] = pixmeas['f0'][(pixmeas['mjd']==d) & (pixmeas['freq']==c) & (pixmeas['stokes']==stokesno)][0]
                        except: 
                            pass 
    #                     try:
    #                         Z[i,c] = np.log10(thisZ) # - np.nanmedian(thisZ)
    #                     except RuntimeWarning:
    #                         continue
    #             Z[np.isnan(Z)] = 0
    #             # setvscale = 3*np.median(Z[Z>0])
    #             # setvscale = np.amax(np.abs(Z))
                waterfall = ax.imshow(Z.transpose(),cmap='viridis',origin='lower',aspect='auto')
                cbar = fig.colorbar(waterfall,ax=ax, orientation='horizontal', cmap='coolwarm',fraction=0.1)
    #             cbar.set_label('Approx. Log10 Flux (Jy)')
                plt.subplots_adjust(hspace=0.3)
                plt.savefig(plotfile.replace(".png",f"-stokes{stokes}.png"))
                plt.close()

        imagedat = dat
        detposra = diffdat['ra'][np.isin(diffdat['mjd'],dettime) & (diffdat['stokes']==detstokes)]
        detposdec = diffdat['dec'][np.isin(diffdat['mjd'],dettime) & (diffdat['stokes']==detstokes)]
        if sno==0:
            meanpos = diffdat['ra'].mean(), diffdat['dec'].mean()
        min3C = cat3c[angsep(meanpos[0],meanpos[1],cat3c['ra'],cat3c['dec'])==angsep(meanpos[0],meanpos[1],cat3c['ra'],cat3c['dec']).min()]
        minVLSSr = VLSSr50jy[angsep(meanpos[0],meanpos[1],VLSSr50jy['ra'],VLSSr50jy['dec'])==angsep(meanpos[0],meanpos[1],VLSSr50jy['ra'],VLSSr50jy['dec']).min()]
        minpulsar = atnf[angsep(meanpos[0],meanpos[1],atnf['ra'],atnf['dec'])==angsep(meanpos[0],meanpos[1],atnf['ra'],atnf['dec']).min()]
        fig = plt.figure()
        plt.scatter(detposra,detposdec,label="t=tdet",marker='P',color='red')
        plt.scatter(diffdat['ra'],diffdat['dec'],alpha=0.1,color='gray')
        plt.scatter(diffdat['ra'].mean(), diffdat['dec'].mean(),marker='x',color='black')
        if angsep(min3C['ra'],min3C['dec'],meanpos[0],meanpos[1])[0] < 5:
            plt.scatter(min3C['ra'],min3C['dec'],marker='s',color='black',label=f"3C{min3C['3C'][0]}")
        if angsep(minVLSSr['ra'],minVLSSr['dec'],meanpos[0],meanpos[1])[0] < 5:
            plt.scatter(minVLSSr['ra'],minVLSSr['dec'],marker='*',color='black',label=f"{minVLSSr['name'][0]}")
        if angsep(minpulsar['ra'],minpulsar['dec'],meanpos[0],meanpos[1])[0] < 5:
            plt.scatter(minpulsar['ra'],minpulsar['dec'],marker='X',color='black',label=f"{minpulsar['name'][0]}")
        plt.legend()
        ax = plt.gca()
        xlim = (meanpos[0] - 5, meanpos[0] + 5)
        ylim = (meanpos[1] - 5, meanpos[1] + 5)
        ax.set_xlim(xlim[0],xlim[1]) 
        ax.set_ylim(ylim[0],ylim[1]) 
        plt.savefig(plotfile.replace("multiplot.png",f"diff_fitpos.png"))
        plt.close()

        detposra = imagedat['ra'][np.isin(imagedat['mjd'],dettime)]
        detposdec = imagedat['dec'][np.isin(imagedat['mjd'],dettime)]
        fig = plt.figure()
        plt.scatter(detposra,detposdec,label="t=tdet",marker='P',color='red')
        plt.scatter(imagedat['ra'],imagedat['dec'],alpha=0.1,color='gray')
        plt.scatter(imagedat['ra'].mean(), imagedat['dec'].mean(),marker='x',color='blue')
        if angsep(minVLSSr['ra'],minVLSSr['dec'],meanpos[0],meanpos[1])[0] < 5:
            plt.scatter(minVLSSr['ra'],minVLSSr['dec'],marker='*',color='black',label=f"{minVLSSr['name'][0]}")
        if angsep(min3C['ra'],min3C['dec'],meanpos[0],meanpos[1])[0] < 5:
            plt.scatter(min3C['ra'],min3C['dec'],marker='s',color='black',label=f"3C{min3C['3C'][0]}")
        if angsep(minpulsar['ra'],minpulsar['dec'],meanpos[0],meanpos[1])[0] < 5:
            plt.scatter(minpulsar['ra'],minpulsar['dec'],marker='X',color='black',label=f"{minpulsar['name'][0]}")
        plt.legend()
        ax = plt.gca()
        ax.set_xlim(xlim[0],xlim[1]) 
        ax.set_ylim(ylim[0],ylim[1]) 
        plt.savefig(plotfile.replace("multiplot.png",f"image_fitpos.png"))
        plt.close()
