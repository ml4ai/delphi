""" setuptools-based setup module. """

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='delphi',
    version='2.1.0',
    description='A framework for assembling probabilistic models from text.',
    url='https://ml4ai.github.io/delphi/',
    author='ML4AI',
    author_email='adarsh@email.arizona.edu',
    classifiers=[
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
    extras_require={
        'dev': ['check-manifest'],
        'test': ['pytest', 'mypy'],
        'docs' : [
            'sphinxcontrib-bibtex',
            'sphinxcontrib-trio',
            'sphinx-autodoc-typehints'
            ]
    },
    dependency_links=['git+https://github.com/sorgerlab/indra.git#egg=indra'],
)
