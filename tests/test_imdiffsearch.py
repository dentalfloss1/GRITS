import unittest 
import numpy as np 
from bifrost import map as BFMAP
from bifrost import ndarray as BFArray
from GRITS.GPUgrouping import *
import matplotlib.pyplot as plt
import os
import sys
import glob
import subprocess
import tempfile
import shutil
currentDir = os.path.abspath(os.getcwd())
if os.path.exists(os.path.join(currentDir, 'tests', 'test_imdiffsearch.py')):
    MODULE_BUILD = currentDir
else:
    MODULE_BUILD = None

oimsFile = os.path.join(os.path.dirname(__file__), 'data', 'test.oims')
pimsFile = os.path.join(os.path.dirname(__file__), 'data', 'test.pims')
configFile = os.path.join(os.path.dirname(__file__), 'data', 'config.ini')
pimsrmsFile = os.path.join(os.path.dirname(__file__), 'data', 'pimsrms.csv')
pimssrcFile = os.path.join(os.path.dirname(__file__), 'data', 'pimsrc.csv')
pimsflagFile = os.path.join(os.path.dirname(__file__), 'data', 'pimsflags.csv')
class imdiffsearch_tests(unittest.TestCase):
    """A unittest.TestCase collection of unit tests for the 
       imdiffsearch.py script"""
    
    testPath = None

    def setUp(self):
        """Turn off all numpy warnings and create the temporary file directory."""

        np.seterr(all='ignore')
        self.testPath = tempfile.mkdtemp(prefix='test-imdiffsearch-', suffix='.tmp')
        
    def run_imdiffsearch(self, *args, print_on_failure=True):
        status = 1
        with open('imdiffsearch.log', 'w') as logfile:
            try:
                cmd = [sys.executable, os.path.join(MODULE_BUILD, 'scripts/imdiffsearch.py')]
                cmd.extend(args)
                
                status = subprocess.check_call(cmd, stdout=logfile)
            except subprocess.CalledProcessError:
                pass
                
        if status == 1 and print_on_failure:
            with open('imdiffsearch.log', 'r') as logfile:
                print(logfile.read())
                
        os.unlink('imdiffsearch.log')
        
        return status
    def test_default_pims(self):
        """Run imdiffsearch on a pimsfile with default settings"""
        cwd = os.getcwd()
        os.chdir(self.testPath) 

        shutil.copy(configFile,self.testPath)
        shutil.copy(pimsrmsFile,self.testPath)
        shutil.copy(pimssrcFile,self.testPath)
        shutil.copy(pimsflagFile,self.testPath)
        
        status = self.run_imdiffsearch(pimsFile,"--noplot")
        self.assertEqual(status,0)
        fluxFile = np.sort(glob.glob(os.path.join(self.testPath, '*fluxfile*')))
        self.assertEqual(len(fluxFile),1)
        dat = np.load(fluxFile[0])
        knownrms = np.loadtxt(pimsrmsFile, delimiter=',')
        for stokes in range(4):
            testdata = np.sort(dat['minrms'][:,0,stokes])
            curknown = np.sort(knownrms[:,stokes])
            np.testing.assert_allclose(testdata,curknown,rtol=1e-1)
        knownsrc = np.loadtxt(pimssrcFile, delimiter=',', dtype=[('id', '<i8'), ('x', '<f8'), ('y', '<f8'), ('ra', '<f8'), ('dec', '<f8'), ('mjd', '<f8'), ('isodate', '<U32'), ('3c', '<U9'), ('VLSSr50', '<U9'), ('index', '<i8'), ('filename', '<U256'), ('channel', '<i8'), ('nchan', '<i8'), ('stokes', '<i8'), ('fpk', '<f8'), ('ftot', '<f8')])
        for key in ['x','y','ra','dec','mjd','fpk','ftot']:
            np.testing.assert_allclose(np.sort(dat['srcarray'][key]),np.sort(knownsrc[key]),rtol=0.1)
        np.testing.assert_allclose(dat['corrfluxlist'],np.array([0.01018831]),rtol=0.1)
        knownflags = np.loadtxt(pimsflagFile, delimiter=',', dtype='i8')
        np.testing.assert_equal(dat['flaggedtimes'],knownflags)
        for ff in fluxFile:
            os.remove(ff)
        os.chdir(cwd)
    def test_default_oims(self):
        """Run imdiffsearch on a pimsfile with default settings"""
        cwd = os.getcwd()
        os.chdir(self.testPath) 

        shutil.copy(configFile,self.testPath)
        
        status = self.run_imdiffsearch(oimsFile,"--noplot")
        self.assertEqual(status,0)
        fluxFile = np.sort(glob.glob(os.path.join(self.testPath, '*fluxfile*')))
        for ff in fluxFile:
            os.remove(ff)
        os.chdir(cwd)







