[![Build Status](https://travis-ci.org/ml4ai/delphi.svg?branch=master)](https://travis-ci.org/ml4ai/delphi)
[![Coverage Status](https://codecov.io/gh/ml4ai/delphi/branch/master/graph/badge.svg)](https://codecov.io/gh/ml4ai/delphi)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1436914.svg)](https://doi.org/10.5281/zenodo.1436914)

Complete documentation available at
[ml4ai.github.io/delphi](https://ml4ai.github.io/delphi) (the 'raw' version can
be found in the `docs` directory.)

Modeling complex phenomena such as food insecurity requires reasoning
over multiple levels of abstraction and fully utilizing expert
knowledge about multiple disparate domains, ranging from the
environmental to the sociopolitical.

Delphi is a C++ library for assembling causal, dynamic, probabilistic
models from information extracted from two sources:

- *Text*: Delphi utilizes causal relations extracted using machine
  reading from text sources such as UN agency reports, news articles,
  and technical papers.
- *Software*: Delphi also incorporates functionality to extract
  abstracted representations of scientific models from code that
  implements them, and convert these into probabilistic models.

Delphi builds upon [INDRA](https://indra.bio) and [Eidos](https://github.com/clulab/eidos).

For a detailed description of our procedure to convert text to models,
see [this document](http://vision.cs.arizona.edu/adarsh/Arizona_Text_to_Model_Procedure.pdf).

Delphi is also part of the
[AutoMATES](https://ml4ai.github.io/automates/) project.

Citing
------

If you use Delphi, please cite the following:

```bibtex

  @InProceedings{sharp-EtAl:2019:N19-4,
    author    = {Sharp, Rebecca  and  Pyarelal, Adarsh  and  Gyori, Benjamin
      and  Alcock, Keith  and  Laparra, Egoitz  and  Valenzuela-Esc\'{a}rcega,
      Marco A.  and  Nagesh, Ajay  and  Yadav, Vikas  and  Bachman, John  and
      Tang, Zheng  and  Lent, Heather  and  Luo, Fan  and  Paul, Mithun  and
      Bethard, Steven  and  Barnard, Kobus  and  Morrison, Clayton  and
      Surdeanu, Mihai},
    title     = {Eidos, INDRA, \& Delphi: From Free Text to Executable Causal Models},
    booktitle = {Proceedings of the 2019 Conference of the North American
    Chapter of the Association for Computational Linguistics (Demonstrations)},
    month     = {6},
    year      = {2019},
    address   = {Minneapolis, Minnesota},
    publisher = {Association for Computational Linguistics},
    pages     = {42-47},
    url       = {http://www.aclweb.org/anthology/N19-4008},
    keywords = {demo paper, causal relations, timelines, locations, information extraction},
  }
   @misc{Delphi,
       Author = {Adarsh Pyarelal and Paul Hein and Jon Stephens and Pratik
                 Bhandari and HeuiChan Lim and Saumya Debray and Clayton
                 Morrison},
       Title = {Delphi: A Framework for Assembling Causal Probabilistic 
                Models from Text and Software.},
       doi={10.5281/zenodo.1436915},
   }
```


License and Funding
-------------------

Delphi is licensed under the Apache License 2.0.

The development of Delphi was supported by the Defense Advanced Research
Projects Agency (DARPA) under the World Modelers (grant no. W911NF1810014) and
Automated Scientific Knowledge Extraction (agreement no. HR00111990011)
programs.
