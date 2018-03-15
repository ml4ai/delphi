# delphi
Framework for visualizing Causal Analysis Graphs

![alt text](/sphinx/delphi_screenshot.png?raw=True")

# Requirements
- The [INDRA package](http://indra.readthedocs.io/en/latest/)

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

# Installation

Clone the directory, install delphi:
```bash
git clone https://github.com/ml4ai/delphi
cd delphi
pip install -e .
```

# Usage

While in the directory, do

```bash
./delphi.py <filename>
```

where `<filename>` should be a JSON-LD file produced by Eidos - for an example,
see: [https://raw.githubusercontent.com/clulab/eidos/master/example_output/example_mar6.jsonld].

You can also do `./delphi.py -h` to view further options.

and navigate your browser to `http://127.0.0.1:5000/`.
