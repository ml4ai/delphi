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

where `<filename>` should be a JSON-LD file produced by Eidos - for an example,
see: [https://raw.githubusercontent.com/clulab/eidos/master/example_output/example_mar6.jsonld].

You can also do `./delphi.py -h` to view further options.

and navigate your browser to `http://127.0.0.1:5000/`.

# Disclaimer
March 12, 2018: The current version only works with the
`adding_eidos_groundings` branch of [this fork of INDRA](https://github.com/adarshp/indra/tree/adding_eidos_groundings).
We are actively working on integrating with the [main INDRA repo](https://github.com/sorgerlab/indra). We expect to be done in about a week.



of INDRA, we are waiting on
integration with INDRA
