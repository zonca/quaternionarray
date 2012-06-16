from setuptools import setup


setup(
    name="quaternionarray",
    version="0.6.2dev",

    packages=['quaternionarray'],

    # metadata for upload to PyPI
    author="Andrea Zonca",
    author_email="code@andreazonca.com",
    description="Python package for fast quaternion arrays math",
    license="GPL3",
    keywords="quaternion, nlerp, rotate",
    url="http://andreazonca.com/software/quaternion-array/",   # project home page, if any
    classifiers=[
          'Development Status :: 3 - Alpha',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License (GPL)',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Programming Language :: Python',
          'Topic :: Office/Business',
          'Topic :: Scientific/Engineering :: Physics',
          ],
)
