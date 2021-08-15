from setuptools import setup
from Cython.Build import cythonize


setup(
	ext_modules = cythonize(["seed.pyx", "sibnetcnt_v2.pyx"], language='c++')
)

