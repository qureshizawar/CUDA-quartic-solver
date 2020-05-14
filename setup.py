import os
import re
import sys
import platform
import subprocess

from os import path

from setuptools import setup, Extension, Command
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += [
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        if gpu_build == 'True':
            print('building with GPU enabled libraries...')
            cmake_args.append('-DGPU_build=true')

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] +
                              cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] +
                              build_args, cwd=self.build_temp)


if __name__ == "__main__":
    # read the contents of README file
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

    gpu_build = 'False'
    if '--GPU_build' in sys.argv:
        index = sys.argv.index('--GPU_build')
        sys.argv.pop(index)  # Removes the '--GPU_build'
        # Returns the element after the '--GPU_build'
        gpu_build = sys.argv.pop(index)

    setup(
        name='QuarticSolver',
        version='0.1.5',
        description='A CPU/GPU library for finding the minimum of a quartic function',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='Zawar Qureshi',
        author_email='qureshizawar@gmail.com',
        url='https://github.com/qureshizawar/CUDA-quartic-solver',
        keywords=['CUDA', 'QUARTIC', 'OPTIMISATION'],
        install_requires=[
             'numpy',
        ],
        classifiers=[
            # "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Topic :: Software Development',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.6',
        ],
        ext_modules=[CMakeExtension('QuarticSolver')],
        cmdclass=dict(build_ext=CMakeBuild),
        zip_safe=False,
    )
