.. delphi documentation master file, created by
   sphinx-quickstart on Tue Feb  6 10:41:42 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to delphi's documentation!
==================================

Delphi is a framework for assembling and executing probabilistic models from textual
evidence.

Requirements
============

* The `eidos reader <https://github.com/clulab/eidos>`_
* The `INDRA package <http://indra.readthedocs.io/en/latest/>`_
* Eidos will be used as a fat JAR through INDRA, so you should do
  ``sbt assembly`` in the ``eidos`` folder, and follow the instructions
  `here <https://gist.github.com/bgyori/37c55681bd1a6e1a2fb6634faf255d60>`_
  to get Eidos and INDRA to work together.


Usage
=====

Clone the directory, install the requirements::

    git clone https://github.com/ml4ai/delphi
    cd delphi
    pip install -r requirements.txt

Add this directory to your ``PYTHONPATH``::

    export PYTHONPATH=`pwd`:$PYTHONPATH

Then, do::

    FLASK_APP=delphi flask run

and navigate your browser to ``http://127.0.0.1:5000/``.

To build the Sphinx API documentation, do::

    make

Disclaimer:
===========

This project is in its early stages, so the documentation is a bit
sparse. We are working on it :)

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
