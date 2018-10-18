from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

extensions = [
    Extension("fit_granger_k", ["fit_granger_k.pyx"],
        include_dirs=[np.get_include()],)
]

setup(
      ext_modules=cythonize(extensions)
    )