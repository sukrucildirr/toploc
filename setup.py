from setuptools import setup, Extension
import os

cpp_extension = Extension(
    'toploc.C.ndd',
    sources=[
        os.path.join('toploc', 'C', 'csrc', 'ndd.cpp'),
        os.path.join('toploc', 'C', 'csrc', 'utils.cpp')
    ],
    include_dirs=[os.path.join('toploc', 'C', 'csrc')],
    language='c++'
)

setup()
#setup(
#    ext_modules=[cpp_extension],
#)
