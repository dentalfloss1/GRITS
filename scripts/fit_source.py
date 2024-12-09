#!/usr/bin/env python
import numpy as np 
from GRITS.imagetools import sourcefitting,pbcorr,hdrtopixmask
from GRITS.GPUimagestats import getminrms
from GRITS.readimage import readpims,readoims
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from lsl.common.stations import lwasv as lsllwasv
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.coordinates import SkyCoord,get_body,EarthLocation
from astropy.wcs import WCS
from astropy.time import Time
import astropy.units as u 
import os
import glob
import imageio
filename = glob.glob('*.pims')[0]
fluxfile = glob.glob('*fluxfile*.npz')
dettimes = []
for ffile in fluxfile:
    dettimes.extend(list(np.load(fluxfile[0])['srcarray']['mjd']))
dettimes = np.array(dettimes)
alldata = np.load(fluxfile[0])
diffFLUX1 = alldata['diffFLUX1']
diffFLUX2 = alldata['diffFLUX2']
detra = list(diffFLUX2['ra'][np.isin(diffFLUX2['mjd'],dettimes)]) + list(diffFLUX1['ra'][np.isin(diffFLUX1['mjd'],dettimes)])
detdec = list(diffFLUX2['dec'][np.isin(diffFLUX2['mjd'],dettimes)]) + list(diffFLUX1['dec'][np.isin(diffFLUX1['mjd'],dettimes)])
ra= np.average(detra)
dec = np.average(detdec)
corrflux = alldata['corrfluxlist2']
rmsdata = alldata['minrms2']
ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata,station = readpims(filename)
sRad2 = sRad
hdrlist2 = hdrlist
detindex2 = alldata['srcarray']['index'][alldata['srcarray']['station']==station.decode()][0]
beampatterns = []
allflaggedtimes = []
unpbdata2 = np.copy(data)
for c in range(nchan):
    XX,YY = pbcorr(hdrlist[0],ngrid,c,psize,station)
    beampatterns.append((XX,YY))
    imagedata[:,c,0,:,:]/=((XX+YY)/2)
    data[:,c,0,:,:]/=((XX+YY)/2)
    for s in range(4):
        allflaggedtimes.append((0,c,s))
rmsdata2 = alldata['minrms2']
for c in range(nchan):
    imagedata[:,c,:,:,:] = (imagedata[:,c,:,:,:] )*corrflux[c]
    data[:,c,:,:,:] = (data[:,c,:,:,:] )*corrflux[c]
    unpbdata2[:,c,:,:,:] = (unpbdata2[:,c,:,:,:] )*corrflux[c]
imagedata[:,:,1,:,:] = np.sqrt( imagedata[:,:,1,:,:]**2 + imagedata[:,:,2,:,:]**2)
imagedata[:,:,2,:,:] = np.sqrt( imagedata[:,:,1,:,:]**2 + imagedata[:,:,2,:,:]**2 + imagedata[:,:,3,:,:]**2)
data[:,:,1,:,:] = np.sqrt( data[:,:,1,:,:]**2 + data[:,:,2,:,:]**2)
data[:,:,2,:,:] = np.sqrt( data[:,:,1,:,:]**2 + data[:,:,2,:,:]**2 + data[:,:,3,:,:]**2)
if not os.path.exists("FULLmeasuredsource.npz"):
    imagefit2, diffFIT2, colnames2, coldtypes2, pixmeas2 = sourcefitting(data, imagedata, allflaggedtimes, ra, dec, hdrlist, sRad,station,detindex2,1.5,fitrange=721,constraint=2)
station2 = station
w = WCS(naxis=2)
w.wcs.crpix = [ngrid/2 + 0.5 ,ngrid/2  + 1.809 ]
cdelt = psize
w.wcs.cdelt = np.array([-cdelt*(1.0154771519556305/(130/128)),cdelt*(1.018/(130/128))]) 
hdr = hdrlist[detindex2]
centerRA = hdr['center_ra']
centerDec = hdr['center_dec']
w.wcs.crval = [centerRA,centerDec]
w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
w.wcs.radesys = 'FK5'
w.wcs.specsys = 'TOPOCENT'
data2 = np.copy(data)
filename = glob.glob('*.oims')[0]
corrflux = alldata['corrfluxlist1']
rmsdata = alldata['minrms1']
rmsdata1 = alldata['minrms1']
ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata,station = readoims(filename)
hdrlist1= hdrlist
sRad1 = sRad
detindex1 = alldata['srcarray']['index'][alldata['srcarray']['station']==station.decode()][0]
beampatterns = []
allflaggedtimes = []
unpbdata1 = np.copy(data)
for c in range(nchan):
    XX,YY = pbcorr(hdrlist[0],ngrid,c,psize,station)
    beampatterns.append((XX,YY))
    imagedata[:,c,0,:,:]/=((XX+YY)/2)
    data[:,c,0,:,:]/=((XX+YY)/2)
    for s in range(4):
        allflaggedtimes.append((0,c,s))

for c in range(nchan):
    imagedata[:,c,:,:,:] = (imagedata[:,c,:,:,:] )*corrflux[c]
    data[:,c,:,:,:] = (data[:,c,:,:,:] )*corrflux[c]
    unpbdata1[:,c,:,:,:] = (unpbdata1[:,c,:,:,:] )*corrflux[c]
station1 = station
imagedata[:,:,1,:,:] = np.sqrt( imagedata[:,:,1,:,:]**2 + imagedata[:,:,2,:,:]**2)
imagedata[:,:,2,:,:] = np.sqrt( imagedata[:,:,1,:,:]**2 + imagedata[:,:,2,:,:]**2 + imagedata[:,:,3,:,:]**2)
data[:,:,1,:,:] = np.sqrt( data[:,:,1,:,:]**2 + data[:,:,2,:,:]**2)
data[:,:,2,:,:] = np.sqrt( data[:,:,1,:,:]**2 + data[:,:,2,:,:]**2 + data[:,:,3,:,:]**2)
data1 = np.copy(data)
if not os.path.exists("FULLmeasuredsource.npz"):
    imagefit1, diffFIT1, colnames1, coldtypes1, pixmeas1 = sourcefitting(data, imagedata, allflaggedtimes, ra, dec, hdrlist, sRad,station,detindex1,1.5,fitrange=721, constraint=2)
    imageflux1 = np.array(imagefit1,dtype=[(name,dtype) for name,dtype in zip(colnames1,coldtypes1)])
    diffFLUX1 = np.array(diffFIT1,dtype=[(name,dtype) for name,dtype in zip(colnames1,coldtypes1)])
    pixmeas1 = np.array(pixmeas1,dtype=[(name,dtype) for name,dtype in zip(colnames1,coldtypes1)])
    imageflux2 = np.array(imagefit2,dtype=[(name,dtype) for name,dtype in zip(colnames2,coldtypes2)])
    diffFLUX2 = np.array(diffFIT2,dtype=[(name,dtype) for name,dtype in zip(colnames2,coldtypes2)])
    pixmeas2 = np.array(pixmeas2,dtype=[(name,dtype) for name,dtype in zip(colnames2,coldtypes2)])
    uniqueid = 0
    formattedoutput = []
    np.savez("FULLmeasuredsource",
             imageflux1=imageflux1,
             diffFLUX1=diffFLUX1,
             imageflux2=imageflux2,
             diffFLUX2=diffFLUX2,
             pixmeas1=pixmeas1,
             pixmeas2=pixmeas2)
else:
    alldata =np.load("FULLmeasuredsource.npz")
    imageflux1 = alldata['imageflux1']
    diffFLUX1 = alldata['diffFLUX1']
    imageflux2 = alldata['imageflux2']
    diffFLUX2 = alldata['diffFLUX2']
    pixmeas1 = alldata['pixmeas1']
    pixmeas2 = alldata['pixmeas2']
for stokesno in range(4):
    dat = imageflux1
    diffdat = diffFLUX1
    pixmeas = pixmeas1
    print(np.unique(diffdat['freq']))
    hourmjd = np.linspace(hdrlist2[detindex2]['start_time'],hdrlist[1]['start_time'],num=1000)
    t = Time(hourmjd[detindex2],format='mjd')
    moon1 = get_body("moon", t,location=EarthLocation.of_site("lwa1"))
    fig = plt.figure()
    ax = plt.gca()
    stat1times = diffFLUX1['mjd']
    stat1times = (stat1times - np.floor(stat1times))*24
    tmid = (np.average(dettimes) - np.floor(dettimes.min()))*24
    trange = [tmid - (5/60), tmid + (5/60)]
    detcond1 = np.abs(diffFLUX1['f0']/(diffFLUX1['f0err']**2)) >3
    detcond2 = np.abs(diffFLUX2['f0']/np.sqrt(diffFLUX2['f0err']**2)) >3
    x1 = list(diffFLUX1['ra'][(diffFLUX1['stokes']==stokesno) & detcond1])
    y1 = list(diffFLUX1['dec'][(diffFLUX1['stokes']==stokesno)& detcond1])
    ax.scatter(x1,y1,label=station1,marker='x')
    stat2times = diffFLUX2['mjd']
    stat2times = (stat2times - np.floor(stat2times))*24
    x2 = list(diffFLUX2['ra'][(diffFLUX2['stokes']==stokesno) & detcond2 ])
    y2 = list(diffFLUX2['dec'][(diffFLUX2['stokes']==stokesno) & detcond2])
    ax.scatter(x2,y2,label=station2,marker='s')
    avx = np.nanmean(x1+x2)
    avy = np.nanmean(y1+y2)
    print(x1+x2)
    print(avx)
    print(y1+y2)
    print(avy)
    sc = SkyCoord(avx,avy,unit='deg')
    if moon1.separation(sc).deg < 5:
        ax.scatter(moon1.ra.deg,moon1.dec.deg,marker='o',color='red',label='moon')
        ellipse = Ellipse((moon1.ra.deg,moon1.dec.deg), width=0.25, height=0.25,angle=0,facecolor='none',ls='-',edgecolor='gray',label="Moon")
        ax.add_patch(ellipse)
    ellipse = Ellipse((avx,avy), width=2, height=2,angle=0,facecolor='none',edgecolor='gray',label="1 deg")
    ax.add_patch(ellipse)
    ellipse = Ellipse((avx,avy), width=4, height=4,angle=0,facecolor='none',edgecolor='gray',label='2 deg',ls=':')
    ax.add_patch(ellipse)
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    # circle = SphericalCircle((avx*u.deg,avy*u.deg), 1*u.deg, facecolor='none',edgecolor='gray',transform=ax.get_transform('fk5'),label='1 deg',alpha=0.5)
    # ax.add_patch(circle)
    # circle = SphericalCircle((avx*u.deg,avy*u.deg), 2*u.deg, facecolor='none',edgecolor='gray',transform=ax.get_transform('fk5'),ls=':',label='2 deg',alpha=0.5)
    # ax.add_patch(circle)
    # xpix,ypix = w.wcs_world2pix([(avx,avy)],0)[0]
    # ax.axvline(xpix)
    # ax.axhline(ypix)
    sc = SkyCoord(avx,avy,unit='deg')
    gal = sc.transform_to('galactic')
    ax.set_title(f"Ecliptic: {avx:.2f},{avy:.2f} Galactic: {gal.l.deg:.2f},{gal.b.deg:.2f}")
    # lon = ax.coords[0]
    # lat = ax.coords[1]
    # lon.set_axislabel("Right Ascension")
    # lat.set_axislabel("Declination")
    # lon.set_major_formatter("d.dd")
    # lat.set_major_formatter("d.dd")
    plt.legend()
    plt.savefig(f"Fullpositions_stokes{stokesno}.png")
    plt.close()
start = max(0,detindex2-10)
stop = min(len(data2),detindex2+11)

for c in range(data2.shape[1]):
    images = []
    for i in range(start,stop):
        pixcrd = hdrtopixmask(hdrlist2[i], data2.shape[-1], sRad2, avx, avy, station2)
        fig, axs = plt.subplots(ncols=2,nrows=2)
        currimage = unpbdata2[i,c,0]
        vmax = rmsdata2[i,c,0]*5
        axs[0,0].imshow(currimage,vmin=-vmax,vmax=vmax,origin='lower',cmap='coolwarm')
        currimage = data2[i,c,1]
        vmax = np.sqrt(rmsdata2[i,c,1]**2 + rmsdata2[i,c,2]**2)*5
        axs[0,1].imshow(currimage,vmin=0,vmax=vmax,origin='lower',cmap='viridis')
        currimage = data2[i,c,2]
        vmax = np.sqrt(rmsdata2[i,c,1]**2 + rmsdata2[i,c,2]**2 + rmsdata2[i,c,3]**2)*5
        axs[1,0].imshow(currimage,vmin=0,vmax=vmax,origin='lower',cmap='viridis')
        currimage = data2[i,c,3]
        vmax = rmsdata2[i,c,3]*5
       
        for ax in axs.flatten():
            circle = plt.Circle((pixcrd[0][0],pixcrd[0][1]),5,fill=False)
            ax.add_artist(circle)
        axs[1,1].imshow(currimage,vmin=-vmax,vmax=vmax,origin='lower',cmap='coolwarm')
        fig.suptitle(f"index from detection {i-detindex2}")
        plt.savefig(f"station2-{i}-{c}.png")
        plt.close()
        images.append(f"station2-{i}-{c}.png")
    with imageio.get_writer(f"station2-channel{c}.mp4", mode='I', fps=3) as writer:
        for im in images:
            image = imageio.imread(im)
            writer.append_data(image)
    for im in images:
        myfile = im
        os.remove(myfile)
start = max(0,detindex1-10)
stop = min(len(data1),detindex1+11)
for c in range(data1.shape[1]):
    images = []
    for i in range(start,stop):
        pixcrd = hdrtopixmask(hdrlist1[i], data1.shape[-1], sRad1, avx, avy, station1)
        fig, axs = plt.subplots(ncols=2,nrows=2)
        currimage = unpbdata1[i,c,0]
        vmax = rmsdata1[i,c,0]*5
        axs[0,0].imshow(currimage,vmin=-vmax,vmax=vmax,origin='lower',cmap='coolwarm')
        currimage = data1[i,c,1]
        vmax = rmsdata1[i,c,1]*5
        axs[0,1].imshow(currimage,vmin=0,vmax=vmax,origin='lower',cmap='viridis')
        currimage = data1[i,c,2]
        vmax = rmsdata1[i,c,2]*5
        axs[1,0].imshow(currimage,vmin=0,vmax=vmax,origin='lower',cmap='viridis')
        currimage = data1[i,c,3]
        vmax = rmsdata1[i,c,3]*5
        axs[1,1].imshow(currimage,vmin=-vmax,vmax=vmax,origin='lower',cmap='coolwarm')
        for ax in axs.flatten():
            circle = plt.Circle((pixcrd[0][0],pixcrd[0][1]),5,fill=False)
            ax.add_artist(circle)
        fig.suptitle(f"index from detection {i-detindex1}")
        plt.savefig(f"station1-{i}-{c}.png")
        plt.close()
        images.append(f"station1-{i}-{c}.png")
    with imageio.get_writer(f"station1-channel{c}.mp4", mode='I', fps=3) as writer:
        for im in images:
            image = imageio.imread(im)
            writer.append_data(image)
    for im in images:
        myfile = im
        os.remove(myfile)
hrdettimes = (dettimes - np.floor(dettimes))*24
fig,axs = plt.subplots(nrows=4,ncols=3, figsize=(50,16))
fig.suptitle(f"{diffFLUX2['freq'][0]} MHz")
for stokesno, (stokes,marker) in enumerate(zip(['I','Linear','Total','V'],['s','o','*','D'])):
    xdata = imageflux2['mjd'][imageflux2['stokes']==stokesno]
    xdata = (xdata - np.floor(xdata))*24
    ydata = imageflux2['f0'][imageflux2['stokes']==stokesno]
    yerr = imageflux2['f0err'][imageflux2['stokes']==stokesno]
    axs[stokesno,0].errorbar(xdata,ydata,yerr=yerr,marker=marker,ls='none',color='black',label=stokes)
    xdiffdata = diffFLUX2['mjd'][diffFLUX2['stokes']==stokesno]
    xdiffdata = (xdiffdata - np.floor(xdiffdata))*24
    ydiffdata = diffFLUX2['f0'][diffFLUX2['stokes']==stokesno]
    ydifferr = diffFLUX2['f0err'][diffFLUX2['stokes']==stokesno]
    axs[stokesno,1].errorbar(xdiffdata,ydiffdata,yerr=ydifferr,marker=marker,ls='none',color='black',label=stokes)
    xpixdata = pixmeas2['mjd'][pixmeas2['stokes']==stokesno]
    xpixdata = (xpixdata - np.floor(xpixdata))*24
    ypixdata = pixmeas2['f0'][pixmeas2['stokes']==stokesno]
    ypixerr = pixmeas2['f0err'][pixmeas2['stokes']==stokesno]
    axs[stokesno,2].scatter(xpixdata,ypixdata,marker=marker,color='black')
    axs[stokesno,2].title.set_text(f"Pixel values stokes {stokes}")
    axs[stokesno,1].title.set_text(f"Diff stokes {stokes} ")
    axs[stokesno,0].title.set_text(f"Image stokes {stokes}")
    axs[stokesno,0].set_ylabel("Flux (Jy)")
    axs[stokesno,1].set_ylabel("Diff. Flux (Jy)")
    axs[stokesno,2].set_ylabel("Pixel Flux (Jy)")
    axs[stokesno,0].legend()
    axs[stokesno,1].legend()
   #  miny = np.amin(ydata[np.isin(xdata,hrdettimes)])
   #  maxy = np.amax(ydata[np.isin(xdata,hrdettimes)])
    for ax in axs.flatten():
        ax.grid(visible=True)
       #  ax.set_ylim(miny,maxy)
        for d in hrdettimes: 
            ax.axvline(d)
            
axs[3,0].set_xlabel("date (MJD)")
axs[3,1].set_xlabel("date (MJD)")
axs[3,2].set_xlabel("date (MJD)")
for ax in axs.flatten():
    print(trange[0],trange[1])
    # ax.set_xlim(trange[0],trange[1])
plt.tight_layout()
plt.savefig("PASI full data")
plt.close()

dat = imageflux1
diffdat = diffFLUX1
pixmeas = pixmeas1
print(np.unique(diffdat['freq']))
for stokesno, (stokes,marker) in enumerate(zip(['I','Linear','Total','V'],['s','o','*','D'])):
    fig, axs = plt.subplots(nrows=nchan+1,ncols=3,figsize=(50,12*nchan))
    for c,ax in zip(np.unique(dat['freq']),axs.flatten()[:(3*nchan):3]):
        freq = c 
        curdat = dat[(dat['freq']==freq)]
        xdata = curdat['mjd'][curdat['stokes']==stokesno]
        xdata = (xdata - np.floor(xdata))*24
        ydata = curdat['f0'][curdat['stokes']==stokesno]
        yerr = curdat['f0err'][curdat['stokes']==stokesno]
        ax.errorbar(xdata,ydata,yerr=yerr,marker=marker,ls='none',color='black',label=stokes)
        ax.set_title(f"{freq:5.2f} MHz stokes {stokes} Image Flux")
        ax.legend()
#         miny = np.amin(ydata[np.isin(xdata,hrdettimes)])
#         maxy = np.amax(ydata[np.isin(xdata,hrdettimes)])
        for ax in axs.flatten():
            ax.grid(visible=True)
            for d in hrdettimes: 
                ax.axvline(d)
        if ax==axs.flatten()[-4]:
            ax.set_xlabel(f"Date (MJD)")
            # ax.set_ylabel("Flux above median (Jy)")
            ax.set_ylabel("Flux (Jy)")

    for c,ax in zip(np.unique(diffdat['freq']),axs.flatten()[1:(3*nchan):3]):
        freq = c 
        curdat = diffdat[(diffdat['freq']==freq) ]

        xdata = curdat['mjd'][curdat['stokes']==stokesno]
        xdata = (xdata - np.floor(xdata))*24
        ydata = curdat['f0'][curdat['stokes']==stokesno]
        yerr = curdat['f0err'][curdat['stokes']==stokesno]
        ax.errorbar(xdata,ydata,yerr=yerr,marker=marker,ls='none',color='black',label=stokes)
        ax.set_title(f"{freq:5.2f} MHz Diff Flux")
       #  miny = np.amin(ydata[np.isin(xdata,hrdettimes)])
       #  maxy = np.amax(ydata[np.isin(xdata,hrdettimes)])
        for ax in axs.flatten():
            ax.grid(visible=True)
            for d in hrdettimes: 
                ax.axvline(d)
        ax.legend()
        if ax==axs.flatten()[-4]:
            ax.set_xlabel(f"Date (MJD)")
            # ax.set_ylabel("Flux above median (Jy)")
            ax.set_ylabel("Diff Flux (Jy)")

    for c,ax in zip(np.unique(pixmeas['freq']),axs.flatten()[2:(3*nchan):3]):
        freq = c 
        curdat = pixmeas[(pixmeas['freq']==freq) ]
        xdata = curdat['mjd'][curdat['stokes']==stokesno]
        xdata = (xdata - np.floor(xdata))*24
        ydata = curdat['f0'][curdat['stokes']==stokesno]
        ax.scatter(xdata,ydata,marker=marker,color='black',label=stokes)
        ax.set_title(f"{freq:5.2f} MHz Pixel Flux")
       #  miny = np.amin(ydata[np.isin(xdata,hrdettimes)])
       #  maxy = np.amax(ydata[np.isin(xdata,hrdettimes)])
        for ax in axs.flatten():
            ax.grid(visible=True)
            for d in hrdettimes: 
                ax.axvline(d)
        ax.legend()
        if ax==axs.flatten()[-4]:
            ax.set_xlabel(f"Date (MJD)")
            # ax.set_ylabel("Flux above median (Jy)")
            ax.set_ylabel("Pixel Flux (Jy)")
    # for ax in axs.flatten()[:-3]:
        # ax.set_xlim(trange[0],trange[1])
    ax = axs.flatten()[-3]
    Z = np.zeros((len(np.unique(dat['mjd'])),nchan))
    sortuhrpsts = np.sort(np.unique(dat['mjd'][dat['mjd']!=0]))
    for i,d in enumerate(sortuhrpsts):
        for chno, c in enumerate(np.unique(np.sort(dat['freq']))):
            try:
                Z[i,chno] = dat['f0'][(dat['mjd']==d) & (dat['freq']==c) & (dat['stokes']==stokesno)][0]
            except: 
                pass 
      # setvscale = 3*np.median(Z[Z>0])
      # setvscale = np.amax(np.abs(Z))
    try: 
        vmax = dat['f0'][(dat['mjd'] > (np.average(dettimes) - 1/60/24)) & (dat['mjd'] < (np.average(dettimes) + 1/60/24))].max()
        vmin = dat['f0'][(dat['mjd'] > (np.average(dettimes) - 1/60/24)) & (dat['mjd'] < (np.average(dettimes) + 1/60/24))].min()
        waterfall = ax.imshow(Z.transpose(),cmap='viridis',origin='lower',aspect='auto',vmin=vmin,vmax=vmax)
    except:
        waterfall = ax.imshow(Z.transpose(),cmap='viridis',origin='lower',aspect='auto')
    cbar = fig.colorbar(waterfall,ax=ax, orientation='horizontal', cmap='coolwarm',fraction=0.1)
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
      # setvscale = 3*np.median(Z[Z>0])
      # setvscale = np.amax(np.abs(Z))
    try:
        vmax = diffdat['f0'][(diffdat['mjd'] > (np.average(dettimes) - 1/60/24)) & (diffdat['mjd'] < (np.average(dettimes) + 1/60/24))].max()
        vmin = diffdat['f0'][(diffdat['mjd'] > (np.average(dettimes) - 1/60/24)) & (diffdat['mjd'] < (np.average(dettimes) + 1/60/24))].min()
        waterfall = ax.imshow(Z.transpose(),cmap='viridis',origin='lower',aspect='auto',vmin=vmin,vmax=vmax)
    except: 
        waterfall = ax.imshow(Z.transpose(),cmap='viridis',origin='lower',aspect='auto')
    cbar = fig.colorbar(waterfall,ax=ax, orientation='horizontal', cmap='coolwarm',fraction=0.1)
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
      # setvscale = 3*np.median(Z[Z>0])
      # setvscale = np.amax(np.abs(Z))
    vmax = pixmeas['f0'][(pixmeas['mjd'] > (np.average(dettimes) - 1/60/24)) & (pixmeas['mjd'] < (np.average(dettimes) + 1/60/24))].max()
    vmin = pixmeas['f0'][(pixmeas['mjd'] > (np.average(dettimes) - 1/60/24)) & (pixmeas['mjd'] < (np.average(dettimes) + 1/60/24))].min()
    waterfall = ax.imshow(Z.transpose(),cmap='viridis',origin='lower',aspect='auto',vmin=vmin,vmax=vmax)
    cbar = fig.colorbar(waterfall,ax=ax, orientation='horizontal', cmap='coolwarm',fraction=0.1)
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(f"fullorvilledata-stokes{stokes}.png")
    plt.close()


