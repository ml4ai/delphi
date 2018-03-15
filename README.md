# Delphi
Delphi is a framework for visualizing causal analysis graphs (CAGs) for DARPA's
World Modelers program. Here is an example of what it looks like:

![alt text](/sphinx/delphi_screenshot.png?raw=True")

The arrows show the direction of causal influences between entities. Circular
arrowheads indicate a positive correlation, while tee arrowheads indicate an
inverse correlation.

As of now, the command-line interface of Delphi makes it easy to visualize
causal relations extracted from the Eidos machine reading system in the JSON-LD
format. However, since Delphi is built on
[INDRA](http://indra.readthedocs.io/en/latest/), it can be easily modified to
visualize output from other readers, such as the one developed by
[BBN](https://www.raytheon.com/ourcompany/bbn).

# Installation

## Requirements

- Python 3.6 or higher.

The recommended way to install Delphi is to use the `pip` Python package
manager:

```bash
pip install -e git+https://github.com/ml4ai/delphi.git#egg=delphi
```

To make it easier to invoke delphi, you can create an alias. On MacOS, you can
do the following:

```bash
echo "alias delphi=python <delphi_path>/delphi.py" >> ~/.bash_profile
```

where `<delphi_path>` should be replaced by the location where `pip` installed
Delphi (you can find this by invoking `pip show delphi` at the command line).

# Usage

After creating this alias, you can invoke delphi as follows

```bash
delphi <filename>
```
where `<filename>` should be a JSON-LD file produced by Eidos. The example used
for the screenshot above can be found
[here](https://raw.githubusercontent.com/ml4ai/delphi/master/delphi/data/10_Document_Eidos_CAG.jsonld).

This starts up a local webserver, which you can access by navigating your
web browser to `http://127.0.0.1:5000/`.

# Features

## Tooltips
- Clicking on nodes and edges toggles tooltips for them - node tooltips show the
    modifiers attached to the entity.
- Edge tooltips show the provenance of the extraction.


## Grounding

Delphi also supports grounding to Eidos' internal toy ontology. For each entity,
Eidos outputs a list of matching entries in their ontology, with corresponding
scores. You can set a minimum score threshold for grounding with the
`--grounding_threshold` flag. Nodes that are grounded to the same term in the
ontology will be merged together. Grounding is a necessarily lossy
transformation, so node tooltips will no longer reflect interpretable state
information. The example in the screenshot above was generated with the
invokation:

```bash
delphi DocN.jsonld --grounding_threshold=0.5
```

You can also do `./delphi.py -h` to view further options.


# Using Eidos

You do not need any additional software if you are just processing JSON-LD
files and visualizing the resulting CAGs. However, If you want to use Eidos to
process text locally on your computer and produce the JSON-LD Eidos output, you
need the following:

- The [eidos reader](https://github.com/clulab/eidos)
- Eidos will be used as a fat JAR through INDRA, so you should do the following:
  - Run `sbt assembly` in the `eidos` folder.
  - Set the environment variable $EIDOSPATH to point to the location of the
      Eidos fat JAR (typically something like
      `/eidos/target/scala-2.12/eidos-assembly-x.x.x-SNAPSHOT.jar`) where
      `x.x.x` should be replaced by the correct version.
  - Follow the instructions
  [here](http://indra.readthedocs.io/en/latest/installation.html#pyjnius) to set
  up the Pyjnius and `jnius-indra` packages.

# License 

Delphi is licensed under the Apache License 2.0.
