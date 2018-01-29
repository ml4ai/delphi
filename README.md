# delphi
Framework for assembling and executing probabilistic models from textual
evidence.

# Requirements
- The [eidos reader](https://github.com/clulab/eidos)
- The [INDRA package](http://indra.readthedocs.io/en/latest/)
- Eidos will be used as a fat JAR through INDRA, so you should do `sbt assembly`
    in the `eidos` folder, and follow the instructions
    [here](https://gist.github.com/bgyori/37c55681bd1a6e1a2fb6634faf255d60)
    to get Eidos and INDRA to work together.

# Usage

Install the requirements:
```
pip install -r requirements.txt
```

Add this directory to your `PYTHONPATH`

```
export PYTHONPATH=`pwd`:$PYTHONPATH
```

Then, do

```
FLASK_APP=webapp/app.py flask run
```

To build the Sphinx API documentation, do:

```
make
```

# Disclaimer:

This project is in its early stages, so the documentation is a bit sparse. We
are working on it :)
