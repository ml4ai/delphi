import os
import sys
import json
import pandas as pd
from itertools import cycle
from delphi.core import *
from delphi.helpers import ltake, lfilter, compose, lmap
from typing import List
from tqdm import trange
import numpy as np
import networkx as nx
from flask import Flask, render_template, request, redirect
import subprocess as sp
from functools import partial

from indra.statements import Influence
# if not os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
from indra.sources import eidos

import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif", # use serif/main font for text elements
    "text.usetex": True,    # use inline math for ticks
    "font.serif": 'Palatino',
    "figure.figsize": [4,3],
})

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
plt.style.use('ggplot')

app                      = Flask(__name__)
app.CAG                  = None
app.factors              = None
app.s0                   = None
app.s_index              = None
app.initialValues        = None
app.elementsJSON         = None
app.n_steps              = 10
app.n_samples            = 10000
app.statements           = None
app.elementsJSONforJinja = None
app.histos_built         = False
app.Δt                   = 1


@app.route('/')
def show_index():
    """ Show the index page. """
    return render_template('index.html', 
                           text = "Input text to be processed here",
                           apps = app)

@app.route("/processText")
def process_text():
    """ Process the input text. """
    eidos.process_text(request.args.get('textToProcess', ''))
    return redirect('/setupExperiment')

@app.route("/setupExperiment")
def setupExperiment():
    ep = eidos.process_json_file('eidos_output.json')
    app.statements = ep.statements
    app.CAG = create_causal_analysis_graph(app.statements) 
    app.elementsJSON = export_to_cytoscapejs(app.CAG)
    with open('elements.json', 'w') as f:
        f.write(json.dumps(app.elementsJSON, indent=2))

    # # Create keys for factors and their time derivatives
    
    app.factors = lmap(lambda n: n['data']['id'],
            filter(lambda n: n['data']['simulable'], app.elementsJSON['nodes']))
    app.escapedFactorNames = lmap(lambda n: n.replace(' ', '_'), app.factors)
    app.s_index = flatMap(lambda n: (n, f'∂{n}/∂t'), app.factors)

    # Set defaults
    app.sigmas = dict(lzip(app.factors, ltake(len(app.factors), repeat(1.0))))
    app.s0 = dict(lzip(app.s_index, ltake(len(app.s_index), cycle([100,0]))))

    app.elementsJSONforJinja = json.dumps(app.elementsJSON)
    return render_template("index.html", apps = app)

@app.route("/runExperiment", methods=["POST"])
def make_histograms():
    """ Make histograms """
    initialValues=dict(lzip(app.s_index,request.form.getlist('initialValue')))
    sigmas = dict(lzip(app.factors, request.form.getlist('sigma')))

    for f in app.factors:
        app.sigmas[f]=float(sigmas[f])

    for k in app.s_index:
        app.s0[k]=float(initialValues[k])

    app.n_steps   = int(request.form.get('nsteps'))
    app.n_samples = int(request.form.get('nsamples'))
    app.Δt = float(request.form.get('Δt'))
    sampled_sequences = runExperiment(app.statements,
            pd.Series(app.s0, index = app.s0.keys()),
            n_steps = app.n_steps, n_samples = app.n_samples)

    fig, axes = plt.subplots()

    for i in range(len(app.factors)):
        for j in range(app.n_steps):
            axes.clear()
            compose(partial(sns.distplot, ax=axes), np.random.normal, np.array)(
                    lfilter(lambda v: abs(v) < 200,
                            map(lambda s: s[j][::2][i], sampled_sequences)))
            axes.set_xlim(0, 200)
            plt.tight_layout()
            fig.savefig(f'static/{app.escapedFactorNames[i]}_{j}.png', dpi=150)

    for n in app.elementsJSON['nodes']:
        n['data']['backgroundImage'] =  f'static/{n["data"]["id"]}_0.png'

    app.elementsJSONforJinja = json.dumps(app.elementsJSON)
    app.histos_built = True

    return render_template('index.html', apps = app)

if __name__=="__main__":
    app.run()
