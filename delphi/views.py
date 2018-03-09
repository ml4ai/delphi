import os
import json
import click
import pathlib

from delphi.core import *
from delphi.utils import ltake, lfilter, compose, lmap, lzip, repeat
from delphi import app

from typing import List, Optional, Dict
import numpy as np
from delphi.types import State

from flask import render_template, request, redirect
from flask.cli import FlaskGroup

from functools import partial
from glob import glob
from pandas import Series
from itertools import cycle

from indra.statements import Influence
from indra.sources import eidos
from indra.assemblers import CAGAssembler 


import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif", # use serif/main font for text elements
    "text.usetex": True,    # use inline math for ticks
    "font.serif": 'Palatino',
    "figure.figsize": [4,3],
})

mpl.use('Agg')

from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

@app.route('/')
def show_index():
    """ Show the index page. """
    if app.state.statements is None and pathlib.Path('eidos_output.json').exists():
        return redirect('/setupExperiment')
    elif app.state.statements is not None:
        return render_template('layout.html', 
                           text = "Input text to be processed here",
                           state = app.state)


@app.route("/processText")
def process_text():
    """ Process the input text. """

    # Clean up old data.
    for filename in glob('static/*.png')+glob('*.json'):
        os.remove(filename)

    app.state.inputText = request.args.get('textToProcess', '')
    eidos.process_text(app.state.inputText)
    return redirect('/setupExperiment')


@app.route("/setupExperiment")
def setupExperiment():
    """ Set up the experiment, get initial values of parameters. """
    app.state.statements = eidos.process_json_file('eidos_output.json').statements
    cag_assembler = CAGAssembler(app.state.statements)
    app.state.CAG = cag_assembler.make_model()
    app.state.elementsJSON = cag_assembler.export_to_cytoscapejs()
    with open('elements.json', 'w') as f:
        f.write(json.dumps(app.state.elementsJSON, indent=2))

    # Create keys for factors and their time derivatives
    
    app.state.factors = lmap(lambda n: n['data']['id'],
            filter(lambda n: n['data']['simulable'], app.state.elementsJSON['nodes']))
    app.state.escapedFactorNames = lmap(lambda n: n.replace(' ', '_'), app.state.factors)
    app.state.s_index = flatMap(lambda n: (n, f'∂{n}/∂t'), app.state.factors)

    # Set defaults
    app.state.sigmas = dict(lzip(app.state.factors, ltake(len(app.state.factors), repeat(1.0))))
    app.state.s0 = dict(lzip(app.state.s_index, ltake(len(app.state.s_index), cycle([100,0]))))

    app.state.elementsJSONforJinja = json.dumps(app.state.elementsJSON)
    return render_template("layout.html", state = app.state)

@app.route("/runExperiment", methods=["POST"])
def make_histograms():
    """ Make histograms """
    initialValues=dict(lzip(app.state.s_index,request.form.getlist('initialValue')))
    sigmas = dict(lzip(app.state.factors, request.form.getlist('sigma')))

    for f in app.state.factors:
        app.state.sigmas[f]=float(sigmas[f])

    for k in app.state.s_index:
        app.state.s0[k]=float(initialValues[k])

    app.state.n_steps   = int(request.form.get('nsteps'))
    app.state.n_samples = int(request.form.get('nsamples'))
    app.state.Δt = float(request.form.get('Δt'))
    sampled_sequences = sample_sequences(app.state.statements,
            Series(app.state.s0, index = app.state.s0.keys()),
            n_steps = app.state.n_steps, n_samples = app.state.n_samples)

    fig, axes = plt.subplots()

    for i in range(len(app.state.factors)):
        for j in range(app.state.n_steps):
            axes.clear()
            compose(partial(sns.distplot, ax=axes), np.random.normal, np.array)(
                    lfilter(lambda v: abs(v) < 200,
                            map(lambda s: s[j][::2][i], sampled_sequences)))
            axes.set_xlim(0, 200)
            plt.tight_layout()
            fig.savefig(f'static/{app.state.escapedFactorNames[i]}_{j}.png', dpi=150)

    for n in app.state.elementsJSON['nodes']:
        n['data']['backgroundImage'] =  f'static/{n["data"]["id"]}_0.png'

    app.state.elementsJSONforJinja = json.dumps(app.state.elementsJSON)
    app.state.histos_built = True

    return render_template('layout.html', state = app.state)
