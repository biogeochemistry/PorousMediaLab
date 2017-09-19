from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

modules = [Extension('*',
                     sources=['src/*.py'],
                     extra_compile_args=[
                         '-shared', '-pthread', '-fPIC', '-fwrapv', '-O3', '-Wall', '-fno-strict-aliasing'])]
# Extension('OdeSolver',
#           sources=['src/OdeSolver.py'],
#           extra_compile_args=[
#               '-shared', '-pthread', '-fPIC', '-fwrapv', '-O3', '-Wall', '-fno-strict-aliasing'])]
for m in modules:
    m.cython_c_in_temp = True

setup(
    name='Column app',
    ext_modules=cythonize(modules),
    cmdclass={'build_ext': build_ext}
)
