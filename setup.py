""" setuptools-based setup module. """

import os
from setuptools import setup, find_packages
import re
import sys
import platform
from subprocess import check_call, check_output

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

here = os.path.abspath(os.path.dirname(__file__))


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir="", builddir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.builddir = builddir


class CMakeBuild(build_ext):
    def run(self):
        # Check if a compatible version of CMake is installed
        try:
            out = check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        os.makedirs(ext.builddir, exist_ok=True)
        check_call(["make", "extensions"])


EXTRAS_REQUIRE = {
    "aske": [
        "plotly",
        "sympy",
        "flask",
        "flask-WTF",
        "flask-codemirror",
        "salib",
        "torch",
    ],
    "dev": [
        "jupyter",
        "jupyter-contrib-nbextensions",
        "check-manifest",
        "rise",
        "shapely",
        "pyshp",
        "xlrd",
    ],
    "test": ["pytest>=4.4.0", "pytest-cov", "pytest-sugar", "pytest-xdist"],
    "docs": [
        "sphinx",
        "sphinx-rtd-theme",
        "sphinxcontrib-bibtex",
        "sphinxcontrib-trio",
        "recommonmark",
        "breathe",
        "exhale",
    ],
}

EXTRAS_REQUIRE["all"] = list(
    {dep for deps in EXTRAS_REQUIRE.values() for dep in deps}
)

setup(
    name="delphi",
    version="4.0.1",
    description="A framework for assembling probabilistic models from text and software.",
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
    ext_modules=[CMakeExtension("extension", "lib/", "build")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    install_requires=[
        "tqdm",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn>=0.9.0",
        "pandas",
        "future==0.16.0",
        "networkx",
        "pygraphviz",
        "dataclasses",
        "flask",
        "ruamel.yaml",
        "pygments",
        "flask",
        "pybind11",
        "SQLAlchemy",
        "flask-sqlalchemy",
        "flask-executor",
        "python-dateutil",
    ],
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "delphi = delphi.apps.cli:main",
            "delphi_rest_api = delphi.apps.rest_api.run:main",
            "codex = delphi.apps.CodeExplorer.app:main",
        ]
    },
)
