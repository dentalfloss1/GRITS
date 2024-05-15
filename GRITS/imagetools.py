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

def angsep(ra1deg,dec1deg,ra2deg,dec2deg):
    ra1 = ra1deg*np.pi/180
    dec1 = dec1deg*np.pi/180
    ra2 = ra2deg*np.pi/180
    dec2 = dec2deg*np.pi/180

    sep =  np.arccos(np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2))
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
    w = WCS(naxis=2)
    w.wcs.crpix = [imSize/2 + 1 + 0.5 * ((imSize+1)%2),imSize/2 + 1 + 0.5 * ((imSize+1)%2)]
    w.wcs.cdelt = np.array([-360.0/(2*sRad)/np.pi, 360.0/(2*sRad)/np.pi])
    w.wcs.crval = [header['center_ra'], header['center_dec']]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    x = np.arange(imSize) - 0.5
    y = np.arange(imSize) - 0.5
    x,y = np.meshgrid(x,y)
    maskpix  = ((x-imSize/2.0)**2 + (y-imSize/2.0)**2) > ((0.98*sRad)**2)
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

def hdrtopixmask(hdr,imSize,sRad,ra,dec):
    w = WCS(naxis=2)
    w.wcs.crpix = [imSize/2 + 1 + 0.5 * ((imSize+1)%2),imSize/2 + 1 + 0.5 * ((imSize+1)%2)]
    w.wcs.cdelt = np.array([-360.0/(2*sRad)/np.pi, 360.0/(2*sRad)/np.pi])
    w.wcs.crval = [hdr['center_ra'], hdr['center_dec']]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    try: 
        pixcrd = [(r,d) for r,d in zip(ra,dec)]
    except TypeError:
        pixcrd = [(ra,dec)]
    return w.wcs_world2pix(pixcrd, 0)
def hdrtoworld(hdr,imSize,sRad,x,y):
    w = WCS(naxis=2)
    w.wcs.crpix = [imSize/2 + 1 + 0.5 * ((imSize+1)%2),imSize/2 + 1 + 0.5 * ((imSize+1)%2)]
    w.wcs.cdelt = np.array([-360.0/(2*sRad)/np.pi, 360.0/(2*sRad)/np.pi])
    w.wcs.crval = [hdr['center_ra'], hdr['center_dec']]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    try:
        pixcrd = [(xp,yp) for xp,yp in zip(x,y)]
    except TypeError:
        pixcrd = [(x,y)]
    return w.wcs_pix2world(pixcrd, 0)

def maskislands(data, stdarray, meanarray, ints, nchan, ngrid,nsigma=3):
    dataislands = np.copy(data)
    islandmasks = np.zeros(data.shape,dtype=bool)
    for i in range(ints):
        for j in range(nchan):
            for k in [0,3]:
                islandcond = np.nan_to_num(data[i,j,k]) < (meanarray[i,j,0] + stdarray[i,j,0]*nsigma)
                islandmasks[i,j,k] = islandmasks[i,j,k] | ~islandcond 
    sumisland = np.sum(islandmasks,axis=1) 
    sumisland = BFArray(sumisland,space='cuda',dtype='f32')
    islanddata = BFArray(islandmasks, space='system', dtype='i8')
    islanddata = islanddata.copy('cuda')
    # MULTI Channel detections only
    BFMap("""if (sumisland(time,stokes,i,j) < (5/6)) isldata(time,chan,stokes,i,j) = 0;""",{'sumisland':sumisland,'isldata':islanddata},
        axis_names=('time','chan','stokes','i','j'), shape=(ints,nchan,4,ngrid,ngrid))
    return islanddata

def oimsfitsout(imagedata, diffdata, index, hdrlist, station, outfile):    
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
        imdata = imagedata[myint,:,:,:,:]
        ddata = diffdata[myint,:,:,:,:]
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
        chan=0
        for curdata in [imdata,ddata]:
            ## Create the FITS HDU and fill in the header information
            hdu = astrofits.ImageHDU(data=curdata)
            hdu.header['TELESCOP'] = station.decode()
            hdu.header['EXPTIME'] = tInt
            ### Coordinates - sky
            hdu.header['NAXIS'] = 4
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
            hdu.header['CTYPE4'] = 'FREQ'
            hdu.header['CRPIX4'] = 1
            hdu.header['CDELT4'] = hdr['bandwidth']
            hdu.header['CRVAL4'] = hdr['start_freq']
            hdu.header['LONPOLE'] = 180.0
            hdu.header['LATPOLE'] = 90.0
            hdu.header['DATE-OBS'] = dateObs.strftime("%Y-%m-%dT%H:%M:%S")
            hdu.header['END_UTC'] = dateEnd.strftime("%Y-%m-%dT%H:%M:%S")
            hdu.header['EXPTIME'] = tInt
            ### LWA1 approximate beam size
            midfreq = (hdr['start_freq']  + ((chan)*hdr['bandwidth']) + (hdr['bandwidth']/2))
            beamSize = 2.2*74e6/midfreq
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

def pimsfitsout(imagedata, diffdata, index, hdrlist, station, outfile):    
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
        imdata = imagedata[myint,:,:,:,:]
        ddata = diffdata[myint,:,:,:,:]
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
            hdu.header['RESTFREQ'] = hdr['freq']
            hdu.header['RESTFRQ'] = hdr['freq']
            hdu.header['RESTBW'] = hdr['bandwidth']
            
            ## Write it to disk
            hdulist.append(hdu)
    hdulist.writeto(outfile, overwrite=True)


def exportcrtf(ra,dec,outfile):
    contents=f"""#CRTFv0 CASA Region Text Format version 0
circle [[{ra}deg, {dec}deg], 15775.2806arcsec] coord=ICRS, corr=[I, Q, U, V], linewidth=2, linestyle=-, symsize=1, symthick=1, color=2ee6d6, font=Helvetica, fontsize=10, fontstyle=bold, usetex=false"""
    with open(outfile,'w') as f:
        f.write(contents)
