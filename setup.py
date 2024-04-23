import glob

from setuptools import setup, Extension, Distribution, find_packages


setup(name="GRITS",
      version="0.0.1",
      description="GPU-based Radio Image Transient Search",
      author="Sarah Chastain",
      author_email="sarahichastain@gmail.com",
      packages=["GRITS"],
      scripts = glob.glob('scripts/*.py'),
      python_requires ='>3.6',
      install_requires = ['numpy','matplotlib','lsl','scipy','astropy','datetime','bifrost','ephem','tqdm','lsl-toolkits-pasiimage','paramiko'],
      test_suite="tests")


