[![Build Status](https://travis-ci.org/ml4ai/delphi.svg?branch=master)](https://travis-ci.org/ml4ai/delphi)
[![Coverage Status](https://codecov.io/gh/ml4ai/delphi/branch/master/graph/badge.svg)](https://codecov.io/gh/ml4ai/delphi)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/ml4ai/delphi/master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1436914.svg)](https://doi.org/10.5281/zenodo.1436914)

<img src="https://raw.githubusercontent.com/ml4ai/delphi/master/docs/delphi_logo.png" width="250">

# Delphi

## Contents
- [Citing](#citing)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

Modeling complex phenomena such as food insecurity requires reasoning
over multiple levels of abstraction and fully utilizing expert
knowledge about multiple disparate domains, ranging from the
environmental to the sociopolitical.

Delphi is a framework for assembling causal, dynamic, probabilistic
models from information extracted from two sources:

- *Text*: Delphi utilizes causal relations extracted using machine
   reading from text sources such as UN agency reports, news articles,
   and technical papers.

- *Software*: Delphi also incorporates functionality to extract
   abstracted representations of scientific models from code that
   implements them, and convert these into probabilistic models.

For a detailed description of our procedure to convert text to models,
see [this
document](http://vision.cs.arizona.edu/adarsh/export/Arizona_Text_to_Model_Procedure.pdf).

For API documentation, see
[ml4ai.github.io/delphi](https://ml4ai.github.io/delphi).

Delphi is also part of the
[AutoMATES](https://ml4ai.github.io/automates/) project.

## Citing

If you use Delphi, please cite the following:

```
@misc{Delphi,
    Author = {Adarsh Pyarelal and Paul Hein and Clayton Morrison},
    Title = {Delphi: A Framework for Assembling Causal Probabilistic Models from Text and Software.},
    doi={10.5281/zenodo.1436915},
}
```
