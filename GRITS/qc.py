from math import isclose
import numpy as np
from astropy.coordinates import SkyCoord
from GRITS.imagetools import hdrtoworld

def restrictnumsrc(islanddata,width,ints,nchan,maxsrcs=2):
    numsrcs = np.zeros(ints,dtype=int)
    for i in range(ints):
        for label in range(0,np.amax(islanddata[i,:,:,:,:])+1):
            for c in range(nchan):
                for s in [0,3]:
                    srcwid = width[i,c,s,label-1]
                    if srcwid > 0:
                        numsrcs[i] += 1
    numsrccond = numsrcs <= maxsrcs
    return numsrccond

def restrictangvel(whereislands, islanddata, hdrlist, length, width,nchan,ngrid,sRad,minangvel=1,rtol=0.2):
    
    srcinfo = []
    srcid=0
    for i in np.sort(np.unique(whereislands[0])):
        for label in range(0,np.amax(islanddata[i,:,:,:,:])+1):
            for c in range(nchan):
                for s in [0,3]:
                    srclen = length[i,c,s,0,label-1,0]
                    srcwid = width[i,c,s,label-1]
                    srclocs = np.argwhere(islanddata[i,c,s,:,:]==label)
                    if ((srclen > 0) and (srcwid >0)):
                        avx, avy = np.average(srclocs+0.5,axis=0)
                        radec = hdrtoworld(hdrlist[i],ngrid,sRad,avx,avy)
                        srcinfo.append((i,srcid,radec[0][0],radec[0][1])) 
                        srcid+=1
    srcarray = np.array(srcinfo, dtype=[('imid','i8'),('srcid','i8'),('ra','f8'),('dec','f8')])
    trackno = 0
    tracklist= []
    lasttrack = False
    justfinishedtrack = False
    nummeas = 3
    for i in range(len(srcarray['srcid'])-nummeas):
        idlist = []
        datelist = []
        sclist = []
        for j in range(nummeas):
            idlist.append(srcarray['srcid'][i+j])
        for id in idlist:
            src = srcarray[srcarray['srcid']==id]
            datelist.append(hdrlist[srcarray['imid'][srcarray['srcid']==id][0]]["start_time"])
            sclist.append(SkyCoord(src['ra'],src['dec'],unit='deg'))
        datediff = (np.amax(datelist) - np.amin(datelist))*3600*24
        posangle1 = sclist[0].position_angle(sclist[1]).degree
        posangle2 = sclist[1].position_angle(sclist[2]).degree
        datecond = (datediff < 120)
        poscond = isclose(posangle1,posangle2,rel_tol=rtol) 
        if poscond and datecond:
            for id,sc,d in zip(idlist,sclist,datelist):
                imid = srcarray['imid'][srcarray['srcid']==id][0]
                tracklist.append([trackno,imid,sc,d])
            justfinishedtrack=True
        else:
            if tracklist and justfinishedtrack:
                trackno += 1
            justfinishedtrack=False

    for i,t1 in enumerate(tracklist[:-1]):
        j = i + 1 
        t2 = tracklist[j]
        if t1[0]==t2[0]:
            angle = abs(t1[2].separation(t2[2]).degree)
            tracklist[i].append(angle[0])
            tracklist[j].append(angle[0])
            time = abs(t1[3] - t2[3])*3600*24
            angvel = angle/time
            tracklist[i].append(angvel[0])
            tracklist[j].append(angvel[0])
    tracks = np.array([(t[0],t[1],t[5]) for t in tracklist], dtype=[('track','i8'),('imid','i8'),('angvel','f8')])
    condlist = []
    avangvel = []
    print(len(np.unique(tracks['track'])), "tracks found")
    for trackno in np.unique(tracks['track']):
        t= tracks[tracks['track']==trackno]
        print("found track")
        print(t)

    for im in range(len(islanddata)):
        if im in np.unique(tracks['imid']):
            thistrack = tracks['track'][tracks['imid']==im][0]
            avangvel = np.average(tracks['angvel'][tracks['track']==thistrack])
            if avangvel > minangvel:
                condlist.append(True)
            else:
                condlist.append(False)
        else:
            condlist.append(True)
    angvelcond = np.array(condlist,dtype=bool) 
    return angvelcond
