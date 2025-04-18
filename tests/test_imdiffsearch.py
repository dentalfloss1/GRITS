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
        
        status = self.run_imdiffsearch(pimsFile,"--noplot")
        self.assertEqual(status,0)
        fluxFile = np.sort(glob.glob(os.path.join(self.testPath, '*fluxfile*')))
        print(fluxFile)
        os.chdir(cwd)







