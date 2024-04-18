import os 

import numpy as np 

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

def angsep(ra1deg,dec1deg,ra2deg,dec2deg):
    ra1 = ra1deg*np.pi/180
    dec1 = dec1deg*np.pi/180
    ra2 = ra2deg*np.pi/180
    dec2 = dec2deg*np.pi/180

    sep =  np.arccos(np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2))
    return 180*sep/np.pi

def maskhorizon(data,hdrlist,ngrid,sRad):
    # return a mask for the part of the image beyond 60 degrees from the center
    x = np.arange(ngrid) + 0.5
    y = np.arange(ngrid) + 0.5
    x,y = np.meshgrid(x,y)
    gtboundary = np.where( ((x-ngrid/2.0)**2 + (y-ngrid/2.0)**2) > ((2*sRad/3)**2) )
    cboundary =  np.where( np.isclose(((x-ngrid/2.0)**2 + (y-ngrid/2.0)**2) , ((2*sRad/3)**2),rtol=0.01 ))
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

    
