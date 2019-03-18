.. delphi documentation master file, created by
   sphinx-quickstart on Tue Feb  6 10:41:42 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Home
====

Modeling complex phenomena such as food insecurity requires reasoning
over multiple levels of abstraction and fully utilizing expert
knowledge about multiple disparate domains, ranging from the
environmental to the sociopolitical.

Delphi is a Python library (3.6+) for assembling causal, dynamic, probabilistic
models from information extracted from two sources:

- *Text*: Delphi utilizes causal relations extracted using machine
  reading from text sources such as UN agency reports, news articles,
  and technical papers.
- *Software*: Delphi also incorporates functionality to extract
  abstracted representations of scientific models from code that
  implements them, and convert these into probabilistic models.

For a detailed description of our procedure to convert text to models,
see `this document <http://vision.cs.arizona.edu/adarsh/export/Arizona_Text_to_Model_Procedure.pdf>`_.

Delphi is also part of the
`AutoMATES <https://ml4ai.github.io/automates/>`_ project.

Citing
------

If you use Delphi, please cite the following:

.. code-block:: bibtex

   @misc{Delphi,
       Author = {Adarsh Pyarelal and Paul Hein and Jon Stephens and Pratik
                 Bhandari and Terrence Lim and Saumya Debray and Clayton
                 Morrison},
       Title = {Delphi: A Framework for Assembling Causal Probabilistic 
                Models from Text and Software.},
       doi={10.5281/zenodo.1436915},
   }


.. toctree::
  :maxdepth: 2
  :caption: Contents:

  self
  installation
  usage
  model
  AnalysisGraph_API
  for2py_API
  grfn_spec
  license_and_funding
  CONTRIBUTING


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
