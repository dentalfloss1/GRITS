import os 

import numpy as np 
from datetime import timedelta 

from lsl.common.paths import DATA as dataPath
from lsl.common.mcs import mjdmpm_to_datetime
from lsl.common.stations import lwasv as lsllwasv
from lsl.common.stations import lwana as lsllwana

from bifrost import ndarray as BFArray
from bifrost import map as BFMap 

from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from astropy.time import Time
from astropy.io import fits as astrofits
import astropy.units as u

from lsl_toolkits.OrvilleImage.wcs import getSVwcs, getGENERICwcs

from scipy.optimize import curve_fit

def gaussian_nobackground(xy, amplitude, xo, yo, sigma, theta):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    sigma_x = sigma
    sigma_y = sigma
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()

def angsep(ra1deg,dec1deg,ra2deg,dec2deg):
    ra1 = ra1deg*np.pi/180
    dec1 = dec1deg*np.pi/180
    ra2 = ra2deg*np.pi/180
    dec2 = dec2deg*np.pi/180
    try:
        sep =  np.arccos(np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2))
    except Exception as e:
        if round((np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2)),1)==1.0:
            sep = 0
    return 180*sep/np.pi

def maskhorizon(data,hdrlist,ngrid,sRad,pbpercent=0.667):
        
    # return a mask for the part of the image beyond 60 degrees from the center
    x = np.arange(ngrid) + 0.5
    y = np.arange(ngrid) + 0.5
    x,y = np.meshgrid(x,y)
    gtboundary = np.where( ((x-ngrid/2.0)**2 + (y-ngrid/2.0)**2) > ((pbpercent*sRad)**2) )
    cboundary =  np.where( np.isclose(((x-ngrid/2.0)**2 + (y-ngrid/2.0)**2) , ((pbpercent*sRad)**2),rtol=0.01 ))
    return gtboundary, cboundary
def calcbeamprops(az,alt,header,freq):

    # az and alt need to be the same shape as the image we will correct

    i = 0
    beamDict = np.load(os.path.join(dataPath, 'lwa1-dipole-emp.npz'))
    polarpatterns = []
    for beamCoeff in (beamDict['fitX'], beamDict['fitY']):
        alphaE = np.polyval(beamCoeff[0,0,:],freq )
        betaE =  np.polyval(beamCoeff[0,1,:],freq )
        gammaE = np.polyval(beamCoeff[0,2,:],freq )
        deltaE = np.polyval(beamCoeff[0,3,:],freq )
        alphaH = np.polyval(beamCoeff[1,0,:],freq )
        betaH =  np.polyval(beamCoeff[1,1,:],freq )
        gammaH = np.polyval(beamCoeff[1,2,:],freq )
        deltaH = np.polyval(beamCoeff[1,3,:],freq )
        corrFnc = None

        def compute_beam_pattern(az, alt, corr=corrFnc):
            zaR = np.pi/2 - alt*np.pi / 180.0
            azR = az*np.pi / 180.0

            c = 1.0
            if corrFnc is not None:
                c = corrFnc(alt*np.pi / 180.0)
                c = np.where(np.isfinite(c), c, 1.0)

            pE = (1-(2*zaR/np.pi)**alphaE)*np.cos(zaR)**betaE + gammaE*(2*zaR/np.pi)*np.cos(zaR)**deltaE
            pH = (1-(2*zaR/np.pi)**alphaH)*np.cos(zaR)**betaH + gammaH*(2*zaR/np.pi)*np.cos(zaR)**deltaH

            return c*np.sqrt((pE*np.cos(azR))**2 + (pH*np.sin(azR))**2)
        # Calculate the beam
        pattern = compute_beam_pattern(az, alt)
        polarpatterns.append(pattern)
        i += 1
    beamDict.close()
    return polarpatterns[0], polarpatterns[1]

def pbcorr(header,imSize,chan,pScale,station):
    sRad   = 360.0/pScale/np.pi / 2
    w = getGENERICwcs(header, imSize)
    x = np.arange(imSize) - 0.5
    y = np.arange(imSize) - 0.5
    x,y = np.meshgrid(x,y)
    maskpix  = ((x-imSize/2.0)**2 + (y-imSize/2.0)**2) > ((0.95*sRad)**2)
    x[maskpix] = imSize/2
    y[maskpix] = imSize/2
    sc = pixel_to_skycoord(x,y,wcs=w,mode='wcs')
    # Need date and location for converting to altaz
    mjd = int(header['start_time'])
    mpm = int((header['start_time'] - mjd)*86400.0*1000.0)
    tInt = header['int_len']*86400.0
    dateObs = mjdmpm_to_datetime(mjd, mpm)
    time = Time(dateObs.strftime("%Y-%m-%dT%H:%M:%S"),format="isot")
    if station==b'LWASV':
        GEOx,GEOy,GEOz = lsllwasv.geocentric_location
        lwasv = EarthLocation.from_geocentric(x=GEOx,y=GEOy,z=GEOz,unit='m')
        aa = AltAz(location=lwasv, obstime=time)
    elif station==b'LWA1':
        lwa1 = EarthLocation.of_site('lwa1')
        aa = AltAz(location=lwa1, obstime=time)
    elif station==b'LWANA':
        GEOx,GEOy,GEOz = lsllwana.geocentric_location
        lwana = EarthLocation.from_geocentric(x=GEOx,y=GEOy,z=GEOz,unit='m')
        aa = AltAz(location=lwana, obstime=time)
    else:
        print(station,"unrecognized")
    myaltaz = sc.transform_to(aa)
    alt = myaltaz.alt.deg
    az = myaltaz.az.deg
    # Keep alt between 0 and 90, adjust az accordingly
    negalt = alt < 0
    alt[negalt] += 90
    az[negalt] + 180
    freq = (int(header['start_freq'])  + ((int(chan)+1)*int(header['bandwidth'])/2))
    XX,YY = calcbeamprops(az,alt,header,freq)
    return XX,YY


def hdrtopixmask(hdr,imSize,sRad,ra,dec,station):
    cdelt = hdr['psize']
    w = WCS(naxis=2)
    ngrid = imSize
    if (station==b"LWA1") and ("startTime" in hdr):     
        w.wcs.crpix = [ngrid/2 + 0.5,ngrid/2  + 1.809]
        w.wcs.cdelt = np.array([-cdelt*(1.0154771519556305/(130/128)),cdelt*(1.018/(130/128))]) 
        centerRA = hdr['center_ra']
        centerDec = hdr['center_dec']
    elif (station==b"LWASV") and ("startTime" in hdr):
        w.wcs.crpix = [ngrid/2 + 0.8663,ngrid/2  + 1.343]
        w.wcs.cdelt = np.array([-cdelt*(1.0154771519556305/(130/128)),cdelt*(1.018/(130/128))]) 
        centerRA = hdr['center_ra']
        centerDec = hdr['center_dec']
    elif (station==b"LWASV") and ("startTime" not in hdr):
        w.wcs.crpix = [ngrid/2 + 1.422, ngrid/2  + 0.5]
        w.wcs.cdelt = np.array([-cdelt*(1.013/(130/128)),cdelt*(1.018/(130/128))]) 
        ha=358.25
        centerDec=33.35763888888889
        SV = EarthLocation(lat=34.348358*u.deg, lon=-106.885783*u.deg, height=1477.8*u.m)
        t_obs = Time(hdr['start_time'], format='mjd', scale='utc',location=SV)
        LST = t_obs.sidereal_time("apparent").degree
        centerRA = (LST - ha) %360
        # print(hdr['center_ra'],centerRA, hdr['center_dec'],centerDec)
    else:
        w.wcs.crpix = [ngrid/2 + 0.5, ngrid/2 + 0.5]
        w.wcs.cdelt = np.array([-cdelt,cdelt])
    w.wcs.crval = [centerRA,centerDec]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.radesys = 'FK5'
    w.wcs.specsys = 'TOPOCENT'
    try: 
        pixcrd = [(r,d) for r,d in zip(ra,dec)]
    except TypeError:
        pixcrd = [(ra,dec)]
    return w.wcs_world2pix(pixcrd, 0)

def hdrtoworld(hdr,imSize,sRad,x,y,station):
    cdelt = hdr['psize']
    w = WCS(naxis=2)
    ngrid = imSize
    if (station==b"LWA1") and ("startTime" in hdr):     
        w.wcs.crpix = [ngrid/2 + 0.5,ngrid/2  + 1.809]
        w.wcs.cdelt = np.array([-cdelt*(1.0154771519556305/(130/128)),cdelt*(1.018/(130/128))]) 
        centerRA = hdr['center_ra']
        centerDec = hdr['center_dec']
    elif (station==b"LWASV") and ("startTime" in hdr):
        w.wcs.crpix = [ngrid/2 + 0.8663,ngrid/2  + 1.343]
        w.wcs.cdelt = np.array([-cdelt*(1.0154771519556305/(130/128)),cdelt*(1.018/(130/128))]) 
        centerRA = hdr['center_ra']
        centerDec = hdr['center_dec']
    elif (station==b"LWASV") and ("startTime" not in hdr):
        w.wcs.crpix = [ngrid/2 + 1.422, ngrid/2  + 0.5]
        w.wcs.cdelt = np.array([-cdelt*(1.013/(130/128)),cdelt*(1.018/(130/128))]) 
        ha=358.25
        centerDec=33.35763888888889
        SV = EarthLocation(lat=34.348358*u.deg, lon=-106.885783*u.deg, height=1477.8*u.m)
        t_obs = Time(hdr['start_time'], format='mjd', scale='utc',location=SV)
        LST = t_obs.sidereal_time("apparent").degree
        centerRA = (LST - ha) %360
        # print(hdr['center_ra'],centerRA, hdr['center_dec'],centerDec)
    else:
        w.wcs.crpix = [ngrid/2 + 0.5, ngrid/2 + 0.5]
        w.wcs.cdelt = np.array([-cdelt,cdelt])
    w.wcs.crval = [centerRA,centerDec]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.radesys = 'FK5'
    w.wcs.specsys = 'TOPOCENT'
    try:
        pixcrd = [(xp,yp) for xp,yp in zip(x,y)]
    except TypeError:
        pixcrd = [(x,y)]
    radec = w.wcs_pix2world(pixcrd, 0)
    if np.isnan(radec[0][0]):
        raise Exception(f"Invalid transform: {x},{y},{w},{station}")
    return radec

def maskislands(imagedata, data, rmsdata,  ints, nchan, ngrid, flaggedtimes, nsigma=3, maxDR=300):
    dataislands = np.copy(data)
    islandmasks = np.zeros(data.shape,dtype=bool)
    flaggedtimes = []
    for i in range(1,ints):
        for j in range(nchan):
            k=0
            islandcond = data[i,j,k] > (rmsdata[i,j,k]*nsigma)
            islandmasks[i,j,k] = islandmasks[i,j,k] | islandcond 
            for k in range(1,4):
                islandcond = ((data[i,j,k] > (rmsdata[i,j,k]*nsigma)) & (imagedata[i,j,k] > 0) & (data[i,j,k] > 0))| ((data[i,j,k] < (-rmsdata[i,j,k]*nsigma)) & (imagedata[i,j,k] < 0) & (data[i,j,k] < 0))
                islandmasks[i,j,k] = islandmasks[i,j,k] | islandcond 
    return flaggedtimes, islandmasks.astype(int)

def oimsfitsout(imagedata, diffdata, index, hdrlist, station, outfile,chan):
    hdulist = astrofits.HDUList()
    ints = len(imagedata)
    ngrid = imagedata.shape[-1]
    minind = max(0,index-2)
    maxind = min(ints-1,index+2)
    numims = maxind - minind + 1
    imagedata = imagedata[minind:maxind+1]
    diffdata = diffdata[minind:maxind+1]
    hdrlist = hdrlist[minind:maxind+1]
    for myint in range(len(imagedata)):
        hdr = hdrlist[myint]
        imdata = imagedata[myint,:,:,:]
        ddata = diffdata[myint,:,:,:]
        imSize = ngrid    
        psize = hdr['pixel_size']
        pScale = psize
        sRad   = 360.0/pScale/np.pi / 2
        x = np.arange(imagedata.shape[-2]) - 0.5
        y = np.arange(imagedata.shape[-1]) - 0.5
        x,y = np.meshgrid(x,y)
        mjd = int(hdr['start_time'])
        mpm = int((hdr['start_time'] - mjd)*86400.0*1000.0)
        # determine if header is in seconds or days
        if round(hdr['int_len']) > 0:
            tInt = hdr['int_len']
        else:
            tInt = hdr['int_len']*86400.0
        dateObs = mjdmpm_to_datetime(mjd, mpm)
        dateEnd = dateObs + timedelta(seconds=int(tInt), microseconds=int((tInt-int(tInt))*1000000))
        for curdata in [imdata,ddata]:
            ## Create the FITS HDU and fill in the header information
            hdu = astrofits.ImageHDU(data=curdata)
            hdu.header['TELESCOP'] = station.decode()
            hdu.header['EXPTIME'] = tInt
            ### Coordinates - sky
            hdu.header['NAXIS'] = 3
            hdu.header['CTYPE1'] = 'RA---SIN'
            hdu.header['CRPIX1'] = imSize/2 + 1 + 0.5 * ((imSize+1)%2)
            hdu.header['CDELT1'] = -360.0/(2*sRad)/np.pi
            hdu.header['CRVAL1'] = hdr['center_ra']
            hdu.header['CUNIT1'] = 'deg'
            hdu.header['CTYPE2'] = 'DEC--SIN'
            hdu.header['CRPIX2'] = imSize/2 + 1 + 0.5 * ((imSize+1)%2)
            hdu.header['CDELT2'] = 360.0/(2*sRad)/np.pi
            hdu.header['CRVAL2'] = hdr['center_dec']
            hdu.header['CUNIT2'] = 'deg'
            ### Coordinates - Stokes parameters
            hdu.header['CTYPE3'] = 'STOKES'
            hdu.header['CRPIX3'] = 1
            hdu.header['CDELT3'] = 1
            hdu.header['CRVAL3'] = 1
            midfreq = (hdr['start_freq']  + ((chan)*hdr['bandwidth']) + (hdr['bandwidth']/2))
            beamSize = 2.2*74e6/midfreq
            ### LWA1 approximate beam size
        #     hdu.header['BMAJ'] = beamSize/psize
        #     hdu.header['BMIN'] = beamSize/psize
        #     hdu.header['BPA'] = 0.0
            hdu.header['COMMENT'] = "beamsize = 2.2*74e6/freq"
            ### Frequency
            hdu.header['RESTFREQ'] = midfreq
            hdu.header['RESTFRQ'] = midfreq
            hdu.header['RESTBW'] = hdr['bandwidth']
            
            ## Write it to disk
            hdulist.append(hdu)
    hdulist.writeto(outfile, overwrite=True)

def pimsfitsout(imagedata, diffdata, index, hdrlist, station, outfile,chan):
    hdulist = astrofits.HDUList()
    ints = len(imagedata)
    ngrid = imagedata.shape[-1]
    minind = max(0,index-2)
    maxind = min(ints-1,index+2)
    numims = maxind - minind + 1
    imagedata = imagedata[minind:maxind+1]
    diffdata = diffdata[minind:maxind+1]
    hdrlist = hdrlist[minind:maxind+1]
    for myint in range(len(imagedata)):
        hdr = hdrlist[myint]
        imdata = imagedata[myint,:,:,:]
        ddata = diffdata[myint,:,:,:]
        imSize = ngrid    
        psize = hdr['xPixelSize']
        pScale = psize
        sRad   = 360.0/pScale/np.pi / 2
        x = np.arange(imagedata.shape[-2]) - 0.5
        y = np.arange(imagedata.shape[-1]) - 0.5
        x,y = np.meshgrid(x,y)
        mjd = int(hdr['startTime'])
        mpm = int((hdr['startTime'] - mjd)*86400.0*1000.0)
        # determine if header is in seconds or days
        if round(hdr['int_len']) > 0:
            tInt = hdr['intLen']
        else:
            tInt = hdr['intLen']*86400.0
        dateObs = mjdmpm_to_datetime(mjd, mpm)
        dateEnd = dateObs + timedelta(seconds=int(tInt), microseconds=int((tInt-int(tInt))*1000000))
        chan=0
        for curdata in [imdata,ddata]:
            ## Create the FITS HDU and fill in the header information
            hdu = astrofits.ImageHDU(data=curdata)
            hdu.header['TELESCOP'] = station.decode()
            hdu.header['EXPTIME'] = tInt
            ### Coordinates - sky
            hdu.header['NAXIS'] = 3
            hdu.header['CTYPE1'] = 'RA---SIN'
            hdu.header['CRPIX1'] = imSize/2 + 1 + 0.5 * ((imSize+1)%2)
            hdu.header['CDELT1'] = -360.0/(2*sRad)/np.pi
            hdu.header['CRVAL1'] = hdr['zenithRA']
            hdu.header['CUNIT1'] = 'deg'
            hdu.header['CTYPE2'] = 'DEC--SIN'
            hdu.header['CRPIX2'] = imSize/2 + 1 + 0.5 * ((imSize+1)%2)
            hdu.header['CDELT2'] = 360.0/(2*sRad)/np.pi
            hdu.header['CRVAL2'] = hdr['zenithDec']
            hdu.header['CUNIT2'] = 'deg'
            ### Coordinates - Stokes parameters
            hdu.header['CTYPE3'] = 'STOKES'
            hdu.header['CRPIX3'] = 1
            hdu.header['CDELT3'] = 1
            hdu.header['CRVAL3'] = 1
            hdu.header['LONPOLE'] = 180.0
            hdu.header['LATPOLE'] = 90.0
            hdu.header['DATE-OBS'] = dateObs.strftime("%Y-%m-%dT%H:%M:%S")
            hdu.header['END_UTC'] = dateEnd.strftime("%Y-%m-%dT%H:%M:%S")
            hdu.header['EXPTIME'] = tInt
            ### LWA1 approximate beam size
            midfreq = (hdr['start_freq']  + ((chan)*hdr['bandwidth']) + (hdr['bandwidth']/2))
            beamSize = 2.2*74e6/midfreq
            hdu.header['BMAJ'] = beamSize/psize
            hdu.header['BMIN'] = beamSize/psize
            hdu.header['BPA'] = 0.0
            ### Frequency
            hdu.header['RESTFREQ'] = midfreq
            hdu.header['RESTFRQ'] = midfreq
            hdu.header['RESTBW'] = hdr['bandwidth']
            
            ## Write it to disk
            hdulist.append(hdu)
    hdulist.writeto(outfile, overwrite=True)


def exportcrtf(ra,dec,outfile):
    contents=f"""#CRTFv0 CASA Region Text Format version 0
circle [[{ra}deg, {dec}deg], 15775.2806arcsec] coord=ICRS, corr=[I, Q, U, V], linewidth=2, linestyle=-, symsize=1, symthick=1, color=2ee6d6, font=Helvetica, fontsize=10, fontstyle=bold, usetex=false"""
    with open(outfile,'w') as f:
        f.write(contents)


def sourcefitting(data,imagedata,flaggedtimes,srcRA,srcDEC,hdrlist,sRad,station,targetindex,statsep):
    ngrid = data.shape[-1]
    nstokes = data.shape[2]
    nchan = data.shape[1]
    ints = data.shape[0]
    imagefit = []
    diffFIT = []
    for c in range(nchan):
        XX,YY = pbcorr(hdrlist[0],ngrid,c,hdrlist[0]['psize'],station)
        imagedata[:,c,0,:,:]/=((XX+YY)/2)
        data[:,c,0,:,:]/=((XX+YY)/2)
    # Source fit the images
    xarr = np.linspace(0,data.shape[-1]-1,data.shape[-1])
    yarr = np.linspace(0, data.shape[-1]-1,data.shape[-1])
    x, y = np.meshgrid(xarr, yarr)
    subxarr = np.linspace(0,10,11)
    subyarr = np.linspace(0, 10,11)
    subx, suby = np.meshgrid(subxarr, subyarr)
    data[1:] = np.where(np.isnan(data[1:]),imagedata[1:] - imagedata[:-1],data[1:])
    pixmeas = []
    for i in range(ints):
        targetx, targety = hdrtopixmask(hdrlist[i],ngrid,sRad,srcRA,srcDEC,station)[0]
        avx = targetx
        avy = targety
        for j in range(nchan):
            for k in range(nstokes):
                if (i,j,k) in flaggedtimes:
                    continue
                hdr = hdrlist[i]
                try:
                    midfreq = (hdr['freq']  + j*hdr['bandwidth'] + (hdr['bandwidth']/2))
                except KeyError:
                    midfreq = (hdr['start_freq']  + j*hdr['bandwidth'] + (hdr['bandwidth']/2))
                bsizepix = (2.2*74e6/midfreq/2)/hdr['psize']
                fsize_gaussian = lambda xy, amplitude, xo, yo: gaussian_nobackground(xy, amplitude, xo, yo, bsizepix,0)
                # Fit image data
                f0 = imagedata[i,j,k,int(round(avy)),int(round(targetx))]
                x0 = avx - round(avx) + 5
                y0 = avy - round(avy) + 5
                initial_guess = [f0,x0,y0]
                if k==0:
                    if f0<0:
                        continue
                    initbounds = [(f0*0.1,f0*10),(x0-(bsizepix/2),x0+(bsizepix/2)),(y0-(bsizepix/2),y0+(bsizepix/2))]
                else:
                    initbounds = [(-np.abs(f0)*10,np.abs(f0)*10),(x0-(bsizepix/2),x0+(bsizepix/2)),(y0-(bsizepix/2),y0+(bsizepix/2))]
                bounds0 = tuple([b[0] for b in initbounds])
                bounds1 = tuple([b[1] for b in initbounds])
                bounds = [bounds0,bounds1]
                fitdata = np.transpose(imagedata[i,j,k,(int(round(avy)) - 5):(int(round(avy)) + 6),(int(round(avx)) - 5):(int(round(avx)) + 6)]).ravel()
                try:
                    
                    popt, pcov, infodict, mesg, ier = curve_fit(fsize_gaussian, (subx, suby), fitdata, p0=initial_guess,bounds=bounds,full_output=True,ftol=1e-3)
                    xOUT = popt[1] + round(avx) - 5
                    yOUT = popt[2] + round(avy) - 5
                    # TIGHT position restrictions
                    if np.sqrt((xOUT - avx)**2 + (yOUT - avy)**2) < 1:
                        raOUT, decOUT = hdrtoworld(hdrlist[i],ngrid,sRad,xOUT,yOUT,station)[0]
                        row = []
                        for pind in range(len(popt)):
                            param = popt[pind]
                            perr = np.absolute(pcov[pind][pind])**0.5
                            row.append(param)
                            row.append(perr)
                        row.append(hdrlist[i]['start_time'])
                        row.append(midfreq/1e6)
                        row.append(k)
                        row.append(raOUT)
                        row.append(decOUT)
                        imagefit.append(tuple(row))
                    else: 
                        raise Exception("fit too far away")
                except Exception as e:
                    # print(e)
                    # print(popt,pcov,infodict,mesg,ier)
                    pass

                # fit diff data
                f0 = data[i,j,k,int(round(avy)),int(round(avx))]
                pixmeas.append((f0,0,avx,0,avy,0,hdrlist[i]['start_time'],midfreq/1e6,k,srcRA,srcDEC))
                x0 = avx - round(avx) + 5
                y0 = avy - round(avy) + 5
                initial_guess = [f0,x0,y0]
                initbounds = [(-np.abs(f0)*10,np.abs(f0)*10),(x0-(bsizepix/2),x0+(bsizepix/2)),(y0-(bsizepix/2),y0+(bsizepix/2))]
                bounds0 = tuple([b[0] for b in initbounds])
                bounds1 = tuple([b[1] for b in initbounds])
                bounds = [bounds0,bounds1]
                fitdata = np.transpose(data[i,j,k,(int(round(avy)) - 5):(int(round(avy)) + 6),(int(round(avx)) - 5):(int(round(avx)) + 6)]).ravel()
                try:
                    popt, pcov, infodict, mesg, ier = curve_fit(fsize_gaussian, (subx, suby), fitdata, p0=initial_guess,bounds=bounds,full_output=True,ftol=1e-3)
                    xOUT = popt[1] + round(avx) - 5
                    yOUT = popt[2] + round(avy) - 5
                    # TIGHT position restrictions
                    if np.sqrt((xOUT - avx)**2 + (yOUT - avy)**2) < 1:
                        raOUT, decOUT = hdrtoworld(hdrlist[i],ngrid,sRad,xOUT,yOUT,station)[0]
                        row = []
                        for pind in range(len(popt)):
                            param = popt[pind]
                            perr = np.absolute(pcov[pind][pind])**0.5
                            row.append(param)
                            row.append(perr)
                        row.append(hdrlist[i]['start_time'])
                        row.append(midfreq/1e6)
                        row.append(k)
                        row.append(raOUT)
                        row.append(decOUT)
                        diffFIT.append(tuple(row))
                    else:
                        raise Exception("fit too far away")
                except Exception as e:
                    pass
                  #   print(e)
                  #   print(popt,pcov,infodict,mesg,ier)
    colnames = ["f0","f0err","x0","x0err","y0","y0err","mjd","freq","stokes","ra","dec"]
    coldtypes = ["f8","f8","f8","f8","f8","f8","f8","f8","i4","f8","f8"]
    return imagefit,diffFIT,colnames,coldtypes,pixmeas


