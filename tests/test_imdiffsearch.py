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
currentDir = os.path.abspath(os.getcwd())
if os.path.exists(os.path.join(currentDir, 'tests', 'test_OIMS2fits.py')):
    MODULE_BUILD = currentDir
else:
    MODULE_BUILD = None
    
run_scripts_tests = False
if MODULE_BUILD is not None:
    run_scripts_tests = True

oimsFile = os.path.join(os.path.dirname(__file__), 'data', 'test.oims')

@unittest.skipUnless(run_scripts_tests, "cannot determine correct script path to use")
class imdiffsearch_tests(unittest.TestCase):
    """A unittest.TestCase collection of unit tests for the 
       imdiffsearch.py script"""
    
    testPath = None

    def setUp(self):
        """Turn off all numpy warnings and create the temporary file directory."""

        np.seterr(all='ignore')
        self.testPath = tempfile.mkdtemp(prefix='test-imdiffsearch-', suffix='.tmp')
        
    def run_default(self, *args, print_on_failure=True):

