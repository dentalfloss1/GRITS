import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal as signal 
import glob
from tqdm import tqdm
from GRITS.catalogs import cat3c,VLSSr50jy
from GRITS.imagetools import angsep
for fno,filename in enumerate(glob.glob("*fluxfile*npz")):
    alldata = np.load(filename)
    dat1 = alldata['imageflux1']
    diffdat1 = alldata['diffFLUX1']
    dat2 = alldata['imageflux2']
    diffdat2 = alldata['diffFLUX2']
    for sno,(dat, diffdat) in enumerate(zip([dat1,dat2],[diffdat1,diffdat2])):

        # print(dat)
        # for ufreq in np.sort(np.unique(dat[:,3])):
        #     subdat = dat[dat[:,3]==ufreq]
        #     fig = plt.figure(figsize=(30,20))
        #     plt.scatter(subdat[:,2],subdat[:,0],color='black')
        #     plt.errorbar(subdat[:,2],subdat[:,0],yerr=subdat[:,1],ls='none',color='black')
        #     ax = plt.gca()
        #     ax.set_xlabel("date (MJD)")
        #     ax.set_ylabel("Flux (Jy)")
        #     plt.savefig(f"freq{ufreq}MHzLC.png")
        #     plt.close()
        #     
        plotfile = filename.replace(".npz",f"_station{sno}_multiplot.png")
        fig,axs = plt.subplots(nrows=1,ncols=2, figsize=(50,16))
        for stokesno, (stokes,marker) in enumerate(zip(['I','Q','U','V'],['s','o','*','D'])):
            xdata = dat['mjd'][dat['stokes']==stokesno]
            ydata = dat['f0'][dat['stokes']==stokesno]
            yerr = dat['f0err'][dat['stokes']==stokesno]
            axs[0].errorbar(xdata,ydata,yerr=yerr,marker=marker,ls='none',color='black',label=stokes)
            xdiffdata = diffdat['mjd'][diffdat['stokes']==stokesno]
            ydiffdata = diffdat['f0'][diffdat['stokes']==stokesno]
            ydifferr = diffdat['f0err'][diffdat['stokes']==stokesno]
            axs[1].errorbar(xdiffdata,ydiffdata,yerr=ydifferr,marker=marker,color='black',ls='none',label=stokes)
        axs[0].set_xlabel("date (MJD)")
        axs[1].set_xlabel("date (MJD)")
        axs[0].set_ylabel("Flux (Jy)")
        axs[1].set_ylabel("Diff. Flux (Jy)")
        axs[0].legend()
        axs[1].legend()
        plt.tight_layout()
        plt.savefig(plotfile)
        plt.close()

        imagedat = dat
        meanpos = diffdat['ra'].mean(), diffdat['dec'].mean()
        min3C = cat3c[angsep(meanpos[0],meanpos[1],cat3c['ra'],cat3c['dec'])==angsep(meanpos[0],meanpos[1],cat3c['ra'],cat3c['dec']).min()]
        minVLSSr = VLSSr50jy[angsep(meanpos[0],meanpos[1],VLSSr50jy['ra'],VLSSr50jy['dec'])==angsep(meanpos[0],meanpos[1],VLSSr50jy['ra'],VLSSr50jy['dec']).min()]
        fig = plt.figure()
        plt.scatter(diffdat['ra'],diffdat['dec'],alpha=0.4)
        plt.scatter(diffdat['ra'].mean(), diffdat['dec'].mean(),marker='x',color='black')
        if angsep(min3C['ra'],min3C['dec'],meanpos[0],meanpos[1])[0] < 5:
            plt.scatter(min3C['ra'],min3C['dec'],marker='s',color='black',label=f"3C{min3C['3C'][0]}")
        if angsep(minVLSSr['ra'],minVLSSr['dec'],meanpos[0],meanpos[1])[0] < 5:
            plt.scatter(minVLSSr['ra'],minVLSSr['dec'],marker='*',color='black',label=f"{minVLSSr['name'][0]}")
        plt.legend()
        plt.savefig(plotfile.replace("multiplot.png",f"diff_fitpos.png"))
        plt.close()
        fig = plt.figure()
        meanpos = imagedat['ra'].mean(), imagedat['dec'].mean()
        plt.scatter(imagedat['ra'],imagedat['dec'],alpha=0.4)
        
        plt.scatter(imagedat['ra'].mean(), imagedat['dec'].mean(),marker='x',color='black')
        if angsep(minVLSSr['ra'],minVLSSr['dec'],meanpos[0],meanpos[1])[0] < 5:
            plt.scatter(minVLSSr['ra'],minVLSSr['dec'],marker='*',color='black',label=f"{minVLSSr['name'][0]}")
        if angsep(min3C['ra'],min3C['dec'],meanpos[0],meanpos[1])[0] < 5:
            plt.scatter(min3C['ra'],min3C['dec'],marker='s',color='black',label=f"3C{min3C['3C'][0]}")
        plt.legend()
        plt.savefig(plotfile.replace("multiplot.png",f"image_fitpos.png"))
        plt.close()
