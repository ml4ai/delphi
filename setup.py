""" setuptools-based setup module. """

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='delphi',

    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    #
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='2.0.1',  # Required

    description='A framework for assembling probabilistic models from text.',  # Required

    url='https://ml4ai.github.io/delphi/',
    author='ML4AI',
    author_email='adarsh@email.arizona.edu',

    classifiers=[  # Optional
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='assembling models from text',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=[
            'tqdm',
            'networkx',
            'numpy',
            'scipy',
            'matplotlib',
            'pandas',
            'future',
            'networkx',
            'indra',
            ],

    python_requires='>=3.6',
    setup_requires=['cython'],
    # List additional groups of dependencies here (e.g. development
    # dependencies). Users will be able to install these using the "extras"
    # syntax, for example:
    #
    #   $ pip install sampleproject[dev]
    #
    # Similar to `install_requires` above, these must be valid existing
    # projects.
    extras_require={  # Optional
        'dev': ['check-manifest'],
        'test': ['pytest', 'mypy'],
        'docs' : [
            'sphinxcontrib-bibtex',
            'sphinxcontrib-trio',
            'sphinx-autodoc-typehints'
            ]
    },

    dependency_links=['git+https://github.com/sorgerlab/indra.git#egg=indra']
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
    # entry_points={  # Optional
        # 'console_scripts': [
            # 'rundelphi=delphi.commands:cli',
        # ],
    # },
)
