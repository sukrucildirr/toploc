from setuptools import setup, find_packages
import os
from torch.utils.cpp_extension import BuildExtension, CppExtension

CSRC_DIR = os.path.join('toploc', 'C', 'csrc')

extensions = [
    CppExtension(
        name='toploc.C.csrc.ndd',
        sources=[os.path.join(CSRC_DIR, 'ndd.cpp')],
        extra_compile_args=['-O3'],
    ),
    CppExtension(
        name='toploc.C.csrc.utils',
        sources=[os.path.join(CSRC_DIR, 'utils.cpp')],
        extra_compile_args=['-O3'],
    ),
]

setup(
    name='toploc',
    ext_modules=extensions,
    packages=['toploc', 'toploc.C.csrc'],
    package_data={
        'toploc.C.csrc': ['*.pyi'],  # Include .pyi files
    },
    cmdclass={'build_ext': BuildExtension},
)
