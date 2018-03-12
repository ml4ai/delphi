# delphi
Framework for visualizing, assembling and executing probabilistic models from textual
evidence.

# Requirements
- The [eidos reader](https://github.com/clulab/eidos)
- The [INDRA package](http://indra.readthedocs.io/en/latest/)
- Eidos will be used as a fat JAR through INDRA, so you should do `sbt assembly`
    in the `eidos` folder, and follow the instructions
    [here](https://gist.github.com/bgyori/37c55681bd1a6e1a2fb6634faf255d60)
    to get Eidos and INDRA to work together.

# Usage

Clone the directory, install delphi:
```bash
git clone https://github.com/ml4ai/delphi
cd delphi
pip install -e .
```

Then, while in the directory, do

```bash
./delphi.py <filename>
```

where <filename> should be a JSON-LD file produced by Eidos - for an example,
see (here)[https://raw.githubusercontent.com/clulab/eidos/master/example_output/example_mar6.jsonld].

and navigate your browser to `http://127.0.0.1:5000/`.
