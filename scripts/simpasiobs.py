import datetime
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.units
from astropy.time import Time
import argparse
from tqdm import tqdm
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("--nogaps",action='store_true',help='simulate an optimistic situation where every image is made')
parser.add_argument('--daytimegaps', action='store_true', help='simulate a situation where pasi stops between 2 PM and 9 PM')
parser.add_argument('--coolingissues', action='store_true',help='simulate a situation where pasi stops between 2 PM and 9 PM and stops for the hottest 36.5 days out of the year')
parser.add_argument('--createregions', nargs='*',help='Create region npy files. Requires an input of obs npy files.')
args = parser.parse_args()


start_epoch = datetime.datetime(1858, 11, 17, 00, 00, 00, 00)
begin = 56013
now = (datetime.datetime.now() - start_epoch).total_seconds()/3600/24
# PASI makes 5 second images, so get largest possible number of images
totalobs=3600*24*(now - begin)/5 
nobs = int(np.floor(totalobs))
lwa1 = EarthLocation.of_site('lwa1')
rng = np.random.default_rng()
obssens = 2e+3
obssig = 0.5e+3 
fov = 55/2

def starttime(i):
    return begin + i*(5/3600/24)
def endtime(i):
    return begin + (i+1)*(5/3600/24)
def getra(i):
    curtime =  begin + i*(5/3600/24)
    obs_time = Time(curtime,scale='utc',location=lwa1,format='mjd')
    raoff = -3.34328732
    ra = obs_time.sidereal_time('mean').to('deg').deg - raoff
    return ra
def getdec(i):
    decoff = 1.22055275
    dec = lwa1.lat.deg - decoff
    return dec
def sens(i):
    return rng.normal(obssens, obssig,size=i.shape)
def donogaps(divisor,chunkindex, chunk):
    i = np.arange(chunk,int(chunk+nobs//divisor))
    array = np.stack([starttime(i), endtime(i),sens(i),getra(i),np.full(i.size,getdec(i)),np.full(i.size,0.),np.full(i.size,fov)],axis=0)
    print('chunk',chunkindex)
    print(array.shape)
    np.save(f'chunkobs{chunkindex}.npy',array)
def computeregioninfo(p):
    timeind = np.array([np.amax(np.argwhere((pointFOV[:,0] == p[0]) & (pointFOV[:,1] ==p[1]))), np.amin(np.argwhere((pointFOV[:,0] == p[0]) & (pointFOV[:,1] ==p[1])))])
    matchregion = (regions['ra']==p[0]) & (regions['dec']==p[1]) & (regions['area']==(4*np.pi*np.sin(p[2]*(np.pi/180/2))**2*(180/np.pi)**2))
    return matchregion, timeind
if __name__=='__main__':
    if args.nogaps:
        divisor = 70
        print( [(divisor, chunkindex, chunk) for chunkindex, chunk in enumerate(range(0, nobs, int(nobs//divisor)))])
        with Pool() as pool:
            pool.starmap(donogaps, [(divisor, chunkindex, chunk) for chunkindex, chunk in enumerate(range(0, nobs, int(nobs//divisor)))])
        print('catching leftovers')
        allchunks = np.arange(0, nobs, int(nobs//divisor))
        chunk = allchunks.max()
        i = np.arange(int(chunk+nobs//divisor),((chunk + nobs//divisor) - 1 - nobs))
        print(len(i), ' leftovers')
        chunkindex  = len(allchunks)
        if len(i) > 0:
            array = np.stack([starttime(i), endtime(i),sens(i),getra(i),np.full(i.size,getdec(i)),np.full(i.size,0.),np.full(i.size,fov)],axis=0)
            np.save(f'chunkobs{chunkindex}.npy',array)

    
    
    elif args.daytimegaps:
        with open('daytimegapslwa1.txt', 'w') as f:
            for i in tqdm(range(nobs)):
                dailyobs = int(24*3600/5)
                turnoff = 10080
                turnon = 15120
                if not ( ((i % dailyobs) > turnoff) and ((i % dailyobs) < turnon)):
                    f.write(f'{starttime(i)},{endtime(i)},{sens(i)},{getra(i)},{getdec(i)},False,{fov}\n')
    elif args.coolingissues:
        with open('coolingissuesgapslwa1.txt', 'w') as f:
            for i in tqdm(range(nobs)):
                dailyobs = int(24*3600/5)
                turnoff = 10080
                turnon = 15120
                yearlyobs = int(365*24*3600/5)
                latejune = int(yearlyobs*0.45)
                latejuly = int(yearlyobs*0.55)
                dailyturnoff = not ( ((i % dailyobs) > turnoff) and ((i % dailyobs) < turnon))
                yearlyturnoff = not ( ((i % yearlyobs) > latejune) and ((i % yearlyobs) < latejuly))
                if dailyturnoff or yearlyturnoff:
                    f.write(f'{starttime(i)},{endtime(i)},{sens(i)},{getra(i)},{getdec(i)},False,{fov}\n')
    elif args.createregions:
        fullarr = np.vstack([np.load(f) for f in args.createregions])
        tstart = fullarr[0,:]
        tstop = fullarr[1,:]
        duration = tstop - tstart

        pointFOV = np.transpose(fullarr[(3,4,6),:])
        uniquepoint = np.unique(pointFOV,axis=0)
        uniquesky = SkyCoord(ra=uniquepoint[:,0],dec=uniquepoint[:,1], unit='deg', frame='fk5')
        numrgns = len(uniquepoint) 
        # Not doing overlap
        regions = np.zeros(np.uint32(len(uniquepoint)), dtype={'names': ('ra', 'dec','identity', 'area', 'timespan', 'stop', 'start'), 'formats': ('f8','f8','U32','f8', 'f8', 'f8', 'f8')})
        for i in range(numrgns): # Label the individual pointings. These regions are for example region 1 NOT 2 and 2 NOT 1 
            regions['identity'][i] = str(i)
            regions['ra'][i] = uniquepoint[i,0]
            regions['dec'][i] = uniquepoint[i,1]
            regions['area'][i] = (4*np.pi*np.sin(uniquepoint[i,2]*(np.pi/180/2))**2*(180/np.pi)**2) # Assumes single circular regions, for multiple pointings or other shapes this needs altering
            leftoff = i + 1
            obssubsection = []
        chunklength =  5000
        for i in tqdm(range(0,len(uniquepoint),chunklength)):
            with Pool() as pool:
                results = pool.map(computeregioninfo,uniquepoint[i:i+chunklength])
            for matchregion,timeind in results:
                regions['timespan'][matchregion] = (tstart[timeind[0]] + duration[timeind[0]] - tstart[timeind[1]])
                regions['stop'][matchregion] = tstart[timeind[0]] + duration[timeind[0]]
                regions['start'][matchregion] = tstart[timeind[1]]
                obssubsection.append([timeind[1],timeind[0],regions['identity'][matchregion][0]])

        print(regions[regions['identity'] != ''])
        np.save('allregions.npy',regions[regions['identity'] != ''])
        np.save('obssubsection', obssubsection)
