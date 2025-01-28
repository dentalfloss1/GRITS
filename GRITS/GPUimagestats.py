import numpy as np 
from bifrost import ndarray as BFArray
from bifrost.ndarray import copy_array
from bifrost import map as BFMap 
from GRITS.ateamtools import getsunjup, getcasa, getcyga, getcrab, getvira
from GRITS.constants import CYG_A, CAS_A, SGR_A, TAU_A, VIRGO_A, HERC_A, floatmax
from GRITS.imagetools import hdrtopixmask,hdrtoworld
import datetime

# Pre-calibration RMS measurement. 
# IN: imagedata, sRad
# OUT: rmsdata np.array
# Set up center region for minrms calculation
# We want a box in the center that does not go beyond the horizon. Also 
# We need to account for the extra one pixel perimeter in the computations.
def getminrms(sRad, imagedata, ints, nchan, ngrid, station, chunking=400):
    boxhalflen = np.sin(np.pi/4)*sRad - 2
    fltmax = np.finfo('f4').max
    imagedata[np.isnan(imagedata)] = fltmax
    gpurmsdata = BFArray(np.full((chunking,nchan,imagedata.shape[2],ngrid,ngrid),floatmax,dtype='f4'),dtype='f32',space='cuda')
    BFimagedata = BFArray(imagedata.astype('f8'),dtype='f64',space='system')
    minrms = np.zeros((ints,nchan,imagedata.shape[2]),dtype='f4')
    for chunk in range(int(np.ceil(ints/chunking))):
        if chunk==0:
            start = 0
            end = min(chunking,ints)
        else:
            start=chunk*chunking 
            end=min(chunk*chunking + chunking ,ints)
        idata = BFimagedata[start:end].copy(space='cuda')
        subints = end-start
        rmsbox =   int(round(30*(ngrid/128)**(1.7)))
        BFMap("""if ((i>%i) && (j>%i) && (i<%i) &&(j<%i)){
                     auto intermediate = 0.0;
                     auto nx=%i;
                     auto ny=%i;
                     for (auto offi=-(nx/2); offi<=(nx/2); offi++) { 
                         for (auto offj=-(ny/2); offj<=(ny/2); offj++){
                             intermediate+=dat(time,chan,stokes,i+offi,j+offj)*dat(time,chan,stokes,i+offi,j+offj);
                         }
                     }
                     rmsdat(time,chan,stokes,i,j) = sqrt(intermediate/((float) nx*ny)) ;
                 }
                 """ % (rmsbox,rmsbox,ngrid - rmsbox,ngrid-rmsbox,rmsbox,rmsbox,),
                 {"dat":idata, "rmsdat":gpurmsdata},
                 axis_names=('time','chan','stokes','i','j'), shape=(subints,nchan,imagedata.shape[2],ngrid,ngrid))
        try:
            copy_array(cpurmsdata, gpurmsdata)
        except NameError:
            cpurmsdata = gpurmsdata.copy('system')
        minrms[start:end] = cpurmsdata.min(axis=(-1,-2))[0:subints]
    return  minrms
def fluxcalmask(sRad, data, imagedata, hdrlist, minrms, sunpos, juppos, nchan, ngrid, station, filename, positioncheck=False, keepateam=False):
    # we can calibrate off of cyg a, cas a, virgo a, or tau a for now
    # We do all the masking of regions we don't want to detect transients from here
    # Also we are getting the max flux from the aforementioned a team sources for 
    # flux calibration
    cygaflux = [] 
    casaflux = []
    viraflux = []
    tauaflux = []
    completemask = np.zeros((len(data),ngrid,ngrid),dtype=bool)
    masksize = 14
    xgrid = np.arange(ngrid) + 0.5
    ygrid = np.arange(ngrid) + 0.5
    xgrid,ygrid = np.meshgrid(xgrid,ygrid)
    xgrid = np.tile(xgrid, len(data)).reshape((len(data),ngrid,ngrid))
    y = np.tile(ygrid, len(data)).reshape((len(data),ngrid,ngrid))
    t0 = datetime.datetime.now()
    allpixcoord = np.zeros((len(data),4,2)) 
    allcygacoord = np.zeros((len(data),2))
    allcasacoord = np.zeros((len(data),2))
    allviracoord = np.zeros((len(data),2))
    alltauacoord = np.zeros((len(data),2))
    for i in range(len(data)):
        src_ra = [src[0] for src in [SGR_A,HERC_A,sunpos[i],juppos[i],CYG_A,CAS_A,VIRGO_A,TAU_A]]
        src_dec = [src[1] for src in [SGR_A,HERC_A,sunpos[i],juppos[i],CYG_A,CAS_A,VIRGO_A,TAU_A]]
        srccoords = hdrtopixmask(hdrlist[i],ngrid,sRad,src_ra,src_dec,station)
        for srcno,(x,y) in enumerate(srccoords[0:4]):
            allpixcoord[i,srcno,...] = (x,y)
        allcygacoord[i,...] = srccoords[4]
        allcasacoord[i,...] = srccoords[5]
        allviracoord[i,...] = srccoords[6]
        alltauacoord[i,...] = srccoords[7]
    for srcno in range(4):
        currcond = ((xgrid-np.tile(allpixcoord[:,srcno,0],ngrid*ngrid).reshape(len(data),ngrid,ngrid))**2 + (ygrid-np.tile(allpixcoord[:,srcno,1],ngrid*ngrid).reshape(len(data),ngrid,ngrid))**2) < (masksize**2)
        if keepateam:
            if srcno in [2,3]:
                completemask = completemask | currcond
        else:
            completemask = completemask | currcond
    noncalmask = np.copy(completemask)
    x = np.arange(ngrid) + 0.5
    y = np.arange(ngrid) + 0.5
    x,y = np.meshgrid(x,y)
    casaposerr = []
    for i in range(len(data)):
        ateammask = np.zeros((ngrid,ngrid),dtype=bool)
        pixcoord = allpixcoord[i]
        for p in pixcoord:
            if p[0]: 
                currcond = ((x-p[0])**2 + (y-p[1])**2) < (masksize**2)
                if not keepateam:
                    completemask[i] = completemask[i] | currcond
                    ateammask = ateammask | currcond
        if positioncheck:
            ctrx, ctry  = ngrid/2, ngrid/2
            p = allcygacoord[i]
            if ~np.isnan(p[0]): 
                currcond = ((x-p[0])**2 + (y-p[1])**2) < (masksize**2)
                pixdist = np.sqrt( (p[0] - ctrx)**2 + (p[1] - ctry)**2)
                distratio = pixdist/sRad
                if (np.sum(currcond) > 0) and (distratio < 0.8):
                    for c in range(nchan):
                        dety, detx = np.where(imagedata[i,c,0,:,:]==imagedata[i,c,0][currcond].max())
                        poserr = np.sqrt((p[0] - detx)**2 + (p[1] - dety)**2)
                        with open("cygoffsets.csv","a") as posfile:
                            posfile.write(f"{filename},{distratio},{poserr[0]}\n")

                ateammask = ateammask | currcond
                if not keepateam:
                    completemask[i] = completemask[i] | currcond
                cygmask = np.where(currcond)
                if np.sum(currcond) >0:
                    for c in range(nchan):
                        cygaflux.append((imagedata[i,c,0,cygmask[0],cygmask[1]].max() - minrms[i,c],c))
            p = allcasacoord[i]
            if ~np.isnan(p[0]): 
                currcond = ((x-p[0])**2 + (y-p[1])**2) < (masksize**2)
                pixdist = np.sqrt( (p[0] - ctrx)**2 + (p[1] - ctry)**2)
                distratio = pixdist/sRad
                if (np.sum(currcond) > 0) and (distratio < 0.8):
                    for c in range(nchan):
                        dety, detx = np.where(imagedata[i,c,0,:,:]==imagedata[i,c,0][currcond].max())
                        poserr = np.sqrt((p[0] - detx)**2 + (p[1] - dety)**2)
                        with open("casoffsets.csv","a") as posfile:
                            posfile.write(f"{filename},{distratio},{poserr[0]}\n")
                ateammask = ateammask | currcond
                if not keepateam:
                    completemask[i] = completemask[i] | currcond
                casmask = np.where(currcond)
                if np.sum(currcond) >0:
                    for c in range(nchan):
                        casaflux.append((imagedata[i,c,0,casmask[0],casmask[1]].max() - minrms[i,c],c))
        else:
            p = allcygacoord[i]
            if ~np.isnan(p[0]): 
                currcond = ((x-p[0])**2 + (y-p[1])**2) < (masksize**2)
                ateammask = ateammask | currcond
                if not keepateam:
                    completemask[i] = completemask[i] | currcond
                cygmask = np.where(currcond)
                if np.sum(currcond) >0:
                    for c in range(nchan):
                        cygaflux.append((imagedata[i,c,0,cygmask[0],cygmask[1]].max() - minrms[i,c],c))
            p = allcasacoord[i]
            if ~np.isnan(p[0]): 
                currcond = ((x-p[0])**2 + (y-p[1])**2) < (masksize**2)
                ateammask = ateammask | currcond
                if not keepateam:
                    completemask[i] = completemask[i] | currcond
                casmask = np.where(currcond)
                if np.sum(currcond) >0:
                    for c in range(nchan):
                        casaflux.append((imagedata[i,c,0,casmask[0],casmask[1]].max() - minrms[i,c],c))
        p =allviracoord[i]
        if ~np.isnan(p[0]): 
            currcond = ((x-p[0])**2 + (y-p[1])**2) < (masksize**2)
            ateammask = ateammask | currcond
            if not keepateam:
                completemask[i] = completemask[i] | currcond
            virmask = np.where(currcond)
            if np.sum(currcond) >0:
                for c in range(nchan):
                    viraflux.append((imagedata[i,c,0,virmask[0],virmask[1]].max() - minrms[i,c],c))
        p =alltauacoord[i]
        if ~np.isnan(p[0]): 
            currcond = ((x-p[0])**2 + (y-p[1])**2) < (masksize**2)
            ateammask = ateammask | currcond
            if not keepateam:
                completemask[i] = completemask[i] | currcond
            taumask = np.where(currcond)
            if np.sum(currcond) >0:
                for c in range(nchan):
                    tauaflux.append((imagedata[i,c,0,taumask[0],taumask[1]].max() - minrms[i,c],c))
        invalid = np.where( completemask[i] )

        for stokesno in range(data.shape[2]):
            data[i,:,stokesno,invalid[0], invalid[1]] = np.nan

    freq = lambda c: (int(hdrlist[0]['start_freq'])  + ((c+1)*int(hdrlist[0]['bandwidth'])/2))*1e-6
    allcorr = []
    for cyg,c in cygaflux:
        if cyg!=0:
            allcorr.append((c,getcyga(freq(c))/cyg))
    for cas,c in casaflux:
        if cas!=0:
                allcorr.append((c,getcasa(freq(c))/cas))
    for vir,c in viraflux:
        if vir!=0:
            allcorr.append((c,getvira(freq(c))/vir))
    for tau,c in tauaflux:
        if tau!=0:
            allcorr.append((c,getcrab(freq(c))/tau))
    allcorr = np.array(allcorr,dtype=[('channel','i8'),('corrfac','f8')])
    allcorr = allcorr[allcorr["corrfac"]>0]
    calchans = np.unique(allcorr['channel'])
    removechan = []
    for c in range(nchan):
        if c not in calchans:
            print("Channel",c,"not calibrated. Removing this channel.")
            if nchan==1:
                print("No good channels")
                exit()
            removechan.append(c)
    if removechan:
        removechan = tuple(removechan)
        imagedata[:,removechan,:,:,:]=0
        data[:,removechan,:,:,:]=0
     
    if nchan == 0 :
        print("No channels calibrated. Skipping file.")
        return False, False
    #### flux correct, 
    corrfluxlist = []
    for c in range(nchan):
        try:
            corrflux = np.nanmean(allcorr['corrfac'][(allcorr['channel']==c)])
        except RuntimeWarning:
            print("Invalid corrflux for channel ",c)
            corrflux = 0
        corrfluxlist.append(corrflux)
        imagedata[:,c,:,:,:] = (imagedata[:,c,:,:,:]) *corrflux
        data[:,c,:,:,:] = data[:,c,:,:,:]*corrflux
    return imagedata, data, corrfluxlist 


def getmultistat(sRad, imagedata, ints, nchan, ngrid):
    # Get rms, mean, and std from clibrated data
    # Set up center region for minrms calculation
    # We want a box in the center that does not go beyond the horizon. Also 
    # We need to account for the extra one pixel perimeter in the computations.
    boxhalflen = np.sin(np.pi/4)*sRad - 2
    
    calimdata = BFArray(np.zeros(imagedata.shape),dtype='f32',space='system')
    calimdata[...] = imagedata
    calimdata = calimdata.copy('cuda')
    nstokes = imagedata.shape[2]
    calrmsdata = BFArray(np.zeros((ints,nchan,nstokes,ngrid,ngrid)),dtype='f32',space='cuda')
    immean = BFArray(np.zeros((ints,nchan,nstokes,ngrid,ngrid)), dtype='f32',space='cuda')
    imstd = BFArray(np.zeros((ints,nchan,nstokes,ngrid,ngrid)), dtype='f32',space='cuda')
    BFMap("""auto boxhalflen = %f;
             auto center = %f/2;
             auto x = (float) i - center + 0.5 ;
             auto y = (float) j - center + 0.5 ;
             auto mean = 0.0;
             auto intermediaterms = 0.0;
             auto intermediatemean = 0.0;
             auto intermediatestd = 0.0;
             if ((x > -boxhalflen) && (x < boxhalflen) && (y > -boxhalflen) && (y < boxhalflen)){
                 for (auto offi=-2; offi<3; offi++) { 
                     for (auto offj=-2; offj<3; offj++){
                         intermediaterms+= dat(time,chan,0,i+offi,j+offj)*dat(time,chan,0,i+offi,j+offj);
                         intermediatemean+=dat(time,chan,0,i+offi,j+offj);
                     }
                 }
                 rmsdat(time,chan,stokes,i,j) = sqrt(intermediaterms/25.0);
                 mean = intermediatemean/25.0;
                 meandat(time,chan,stokes,i,j) = mean;
                 for (auto offi=-2; offi<3; offi++) { 
                     for (auto offj=-2; offj<3; offj++){
                         intermediatestd+=(dat(time,chan,0,i+offi,j+offj) - mean)*(dat(time,chan,0,i+offi,j+offj) - mean);
                     }
                 }
                 std(time,chan,stokes,i,j) = sqrt(intermediatestd/25.0) ;
             }
             else {
                 rmsdat(time,chan,stokes,i,j) = 9999;
                 }""" % (boxhalflen,float(ngrid),),
                 {"dat":calimdata,"rmsdat":calrmsdata,'meandat':immean,'std':imstd},
                 axis_names=('time','chan','stokes','i','j'), shape=(ints,nchan,nstokes,ngrid,ngrid))
    calrmsdata = calrmsdata.copy('system')
    calimdata = calimdata.copy('system')
    rmsdata = []
    sndata = []
    for avdat,im in zip(calrmsdata[:,:,0,:,:],imagedata[:,:,:,:,:]):
        currms = np.amin(avdat[avdat!=0])
        rmsdata.append(currms)
        sndata.append(np.amax(im)/currms)
    immean= np.array(immean.copy('system'))
    imstd = np.array(imstd.copy('system'))
    meanstokeep = []
    stdtokeep = []
    meanarray = np.zeros((ints,nchan,nstokes))
    stdarray = np.zeros((ints,nchan,nstokes))
    for i in range(ints):
        for c in range(nchan):
            for s in range(nstokes):
                meanarr = immean[i,c,s]
                stdarr = imstd[i,c,s]
                nonzero = meanarr!=0
                valid1 = meanarr!=9999
                meanarr = meanarr[nonzero & valid1]
                stdarr = stdarr[nonzero & valid1]
                if np.sum(nonzero & valid1)==0:
                    print("all mean values are zero")
                    print(i,c,s)
                    meanarray[i,c,s] = floatmax
                    stdarray[i,c,s] = floatmax
                else:
                    minabsmean = np.abs(meanarr)
                    minmean = np.amin(minabsmean)
                    mincond = minabsmean==minmean
                    meanarray[i,c,s] = meanarr[mincond][0]
                    stdarray[i,c,s] = stdarr[mincond][0]
    return sndata, rmsdata, stdarray, meanarray
def getdiffstats(data,sRad,ints,nchan,ngrid):
    # Set up center region for mean and std calculations of diff ims
    # We want a box in the center that does not go beyond the horizon. Also 
    # We need to account for the extra one pixel perimeter in the computations.
    data = np.where(data==np.nan,floatmax,data)
    boxhalflen = np.sin(np.pi/4)*sRad - 1
    diffdata = BFArray(np.zeros(data.shape),dtype='f32',space='system')
    diffdata[...] = data
    diffdata = diffdata.copy('cuda')
    meandata = BFArray(np.zeros((ints,nchan,ngrid,ngrid)),dtype='f32',space='cuda')
    stddata = BFArray(np.zeros((ints,nchan,ngrid,ngrid)),dtype='f32',space='cuda')
    BFMap("""auto boxhalflen = %f;
             auto center = %f/2;
             auto x = (float) i - center + 0.5 ;
             auto y = (float) j - center + 0.5 ;
             auto mean = 0.0;
             if ((x > -boxhalflen) && (x < boxhalflen) && (y > -boxhalflen) && (y < boxhalflen)){
                 auto tl = dat(time,chan,0,i-1,j+1);
                 auto tp = dat(time,chan,0,i,j+1);
                 auto tr = dat(time,chan,0,i+1,j+1);
                 auto lt = dat(time,chan,0,i-1,j);
                 auto mid = dat(time,chan,0,i-1,j);
                 auto rt = dat(time,chan,0,i+1,j);
                 auto bl = dat(time,chan,0,i-1,j-1);
                 auto bm = dat(time,chan,0,i,j-1);
                 auto br = dat(time,chan,0,i+1,j-1);
                 mean = (tl + tp + tr + lt + mid + rt + bl + bm + br)/9.0 ;
                 meandat(time,chan,i,j) = mean;
                 std(time,chan,i,j) = sqrt(((tl-mean)*(tl-mean) + (tp-mean)*(tp-mean) + (tr-mean)*(tr-mean) + (lt-mean)*(lt-mean) + (mid-mean)*(mid-mean) + (rt-mean)*(rt-mean) + (bl-mean)*(bl-mean) + (bm-mean)*(bm-mean) + (br-mean)*(br-mean))/9.0) ;
             }
             else {
                 meandat(time,chan,i,j) = 9999;
                 std(time,chan,i,j) = 9999;
                 }""" % (boxhalflen,float(ngrid),),
                 {"dat":diffdata, "meandat":meandata,"std":stddata},
                 axis_names=('time','chan','i','j'), shape=(ints,nchan,ngrid,ngrid))
    diffmean= np.array(meandata.copy('system'))
    diffstd = np.array(stddata.copy('system'))
    meanstokeep = []
    stdtokeep = []
    for i in range(ints):
        meanarr = diffmean[i]
        stdarr = diffstd[i]
        nonzero = meanarr!=0
        valid1 = meanarr!=9999
        valid2 = ~np.isnan(meanarr)
        toeval = np.where(nonzero & valid1 & valid2)
        meanarr = meanarr[toeval[0],toeval[1],toeval[2]]
        stdarr = stdarr[toeval[0],toeval[1],toeval[2]]
        minabsmean = np.abs(meanarr)
        minmean = np.amin(minabsmean)
        wheremin = np.where(minabsmean==minmean)
        meanstokeep.append(meanarr[wheremin[0]][0])
        stdtokeep.append(stdarr[wheremin[0]][0])
    meanofdiff = np.array(meanstokeep)
    stdofdiff = np.array(stdtokeep)
    return meanofdiff, stdofdiff
