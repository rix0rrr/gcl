from setuptools import setup, find_packages  # Always prefer setuptools over distutils
from codecs import open  # To use a consistent encoding
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file (only on devhost)
try:
  with open(path.join(here, 'README.md'), encoding='utf-8') as f:
      long_description = f.read()
except IOError:
  long_description = ''

#https://packaging.python.org/en/latest/distributing.html
setup(
    name='gcl',
    version='0.6.10',
    description='Generic Configuration Language',
    long_description=long_description,
    url='https://github.com/rix0rrr/gcl',
    author='Rico Huijbers',
    author_email='rix0rrr@gmail.com',
    license='MIT',

    entry_points = {
        'console_scripts' : [
            'gcl-print=gcl.printer:main',
            'gcl-doc=gcl.doc:main',
            'gcl2json=gcl.to_json:main',
        ],
    },

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Interpreters',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],

    # What does your project relate to?
    keywords='configuration',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),

    # List run-time dependencies here.  These will be installed by pip when your
    # project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['pyparsing>=2.0.3'],
)
