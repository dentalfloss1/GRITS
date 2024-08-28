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

from lsl_toolkits.OrvilleImage.getwcs import getSVwcs, getGENERICwcs

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
    if (station==b'LWASV') and ("startTime" not in header):
        w = getSVwcs(header, imSize)
    else:
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
        x,y,z = lsllwasv.geocentric_location
        lwasv = EarthLocation.from_geocentric(x=x,y=y,z=z,unit='m')
        aa = AltAz(location=lwasv, obstime=time)
    elif station==b'LWA1':
        lwa1 = EarthLocation.of_site('lwa1')
        aa = AltAz(location=lwa1, obstime=time)
    elif station==b'LWANA':
        x,y,z = lsllwana.geocentric_location
        lwana = EarthLocation.from_geocentric(x=x,y=y,z=z,unit='m')
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
    if (station==b'LWASV') and ("startTime" not in hdr):
        w = getSVwcs(hdr, imSize)
    elif ('startTime' in hdr) and (station==b"LWA1"):
        cdelt = hdr['psize']
        w = WCS(naxis=2)
        ngrid = imSize
        # 130 degrees is what is visible to the dipoles
        HA = 0
        if station==b'LWASV':
            w.wcs.crpix = [ngrid/2 + 0.5,ngrid/2  + 0.5]
            x,y,z = lsllwasv.geocentric_location
            stationLOC = EarthLocation.from_geocentric(x=x,y=y,z=z,unit='m')
        elif station==b'LWA1':
            w.wcs.crpix = [ngrid/2 + 0.5,ngrid/2  + 1.809]
            stationLOC = EarthLocation.of_site('lwa1')
        t_obs = Time(hdr['startTime'],format='mjd',scale='utc',location=stationLOC)
        LST = t_obs.sidereal_time('apparent').degree
        center_RA = LST-HA
        center_dec = stationLOC.lat.degree
        corcdelt = cdelt*1.0023384615384616 
        w.wcs.cdelt = np.array([-corcdelt,corcdelt]) 
        w.wcs.crval = [center_RA,center_dec]
        w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
        w.wcs.radesys = 'FK5'
        w.wcs.specsys = 'TOPOCENT'
        w.wcs.equinox = t_obs.jyear
        
    else:
        w = getGENERICwcs(hdr, imSize)
    try: 
        pixcrd = [(r,d) for r,d in zip(ra,dec)]
    except TypeError:
        pixcrd = [(ra,dec)]
    return w.wcs_world2pix(pixcrd, 0)
def hdrtoworld(hdr,imSize,sRad,x,y,station):
    if (station==b'LWASV') and ("startTime" not in hdr):
        w = getSVwcs(hdr, imSize)
    elif ('startTime' in hdr) and (station==b"LWA1"):
        cdelt = hdr['psize']
        w = WCS(naxis=2)
        ngrid = imSize
        # 130 degrees is what is visible to the dipoles
        HA = 0
        if station==b'LWASV':
            w.wcs.crpix = [ngrid/2 + 0.5,ngrid/2  + 0.5]
            x,y,z = lsllwasv.geocentric_location
            stationLOC = EarthLocation.from_geocentric(x=x,y=y,z=z,unit='m')
        elif station==b'LWA1':
            w.wcs.crpix = [ngrid/2 + 0.5,ngrid/2  + 1.809]
            stationLOC = EarthLocation.of_site('lwa1')
        t_obs = Time(hdr['startTime'],format='mjd',scale='utc',location=stationLOC)
        LST = t_obs.sidereal_time('apparent').degree
        center_RA = LST-HA
        center_dec = stationLOC.lat.degree
        corcdelt = cdelt*1.0023384615384616 
        w.wcs.cdelt = np.array([-corcdelt,corcdelt]) 
        w.wcs.crval = [center_RA,center_dec]
        w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
        w.wcs.radesys = 'FK5'
        w.wcs.specsys = 'TOPOCENT'
        w.wcs.equinox = t_obs.jyear
        
    else:
        w = getGENERICwcs(hdr, imSize)
    try:
        pixcrd = [(xp,yp) for xp,yp in zip(x,y)]
    except TypeError:
        pixcrd = [(x,y)]
    return w.wcs_pix2world(pixcrd, 0)

def maskislands(data, rmsdata,  ints, nchan, ngrid,nsigma=3, maxDR=300):
    dataislands = np.copy(data)
    islandmasks = np.zeros(data.shape,dtype=bool)
    flaggedtimes = []
    for i in range(1,ints):
        for j in range(nchan):
            linearsigma = maxDR
            k = 1
            islandcond = data[i,j,k] > (rmsdata[i,j]*linearsigma)
            brightQ = np.sum(islandcond)
            k = 2
            islandcond = data[i,j,k] > (rmsdata[i,j]*linearsigma)
            brightU = np.sum(islandcond)

                # print(np.amax(data[i,j,k]),print(rmsdata[i,:].max()*3),i,j,k)

            # If detected at 100 sigma in linear polarization, skip due to likely RFI/bright sun

            if (brightQ==0) and (brightU==0):
                k = 0 
                islandcond = data[i,j,k] > (rmsdata[i,j]*nsigma)
                islandmasks[i,j,k] = islandmasks[i,j,k] | islandcond 
                k = 3
                islandcond = (data[i,j,k]) > (rmsdata[i,j]*nsigma)
                islandmasks[i,j,k] = islandmasks[i,j,k] | islandcond 
            else:
                flaggedtimes.append((i,j))
    sumisland = np.sum(islandmasks,axis=1) 
    sumisland = BFArray(sumisland,space='cuda',dtype='f32')
    islanddata = BFArray(islandmasks, space='system', dtype='u16')
    islanddata = islanddata.copy('cuda')
    # MULTI Channel detections only
    BFMap("""if (sumisland(time,stokes,i,j) < (5/6)) isldata(time,chan,stokes,i,j) = 0;""",{'sumisland':sumisland,'isldata':islanddata},
        axis_names=('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))
    islanddata = islanddata.copy('system')
    return flaggedtimes, islanddata

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
