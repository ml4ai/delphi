""" setuptools-based setup module. """

from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

setup(
    name="delphi",
    version="3.0.0",
    description="A framework for assembling probabilistic models from text.",
    url="https://github.com/ml4ai/delphi",
    author="ML4AI",
    author_email="adarsh@email.arizona.edu",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
    ],
    keywords="assembling models from text",
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    install_requires=[
        "indra",
        "tqdm",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn>=0.9.0",
        "pandas",
        "future",
        "networkx",
        "pygraphviz",
        "cython",
        "dataclasses",
        "flask",
        "SQLAlchemy",
        "flask-sqlalchemy",
        "fuzzywuzzy[speedup]",
        "jupyter",
        "jupyter-contrib-nbextensions",
        "python-dateutil",
        "salib",
        "tangent",
        "torch",
        "ruamel.yaml",
    ],
    python_requires=">=3.6",
    extras_require={
        "dev": [
            "check-manifest",
            "rise",
            "shapely",
            "pyshp",
            "xlrd",
            "pyjnius",
            "plotly"
            ],
        "test": [
            "pytest>=3.6.0",
            "pytest-cov",
            "pytest-sugar",
            "pytest-xdist",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "sphinxcontrib-bibtex",
            "sphinxcontrib-trio",
            "sphinx-autodoc-typehints",
            "recommonmark",
        ],
    },
    entry_points={"console_scripts": ["delphi = delphi.cli:main"]},
)
