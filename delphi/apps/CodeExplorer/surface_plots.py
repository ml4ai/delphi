import json
import plotly
import plotly.graph_objs as go
import pandas as pd

from flask import flash

BOUNDS = {
    "petpt::msalb_-1": [0, 1],
    "petpt::srad_-1": [1, 20],
    "petpt::tmax_-1": [-30, 60],
    "petpt::tmin_-1": [-30, 60],
    "petpt::xhlai_-1": [0, 20],
    "petasce::doy_-1": [1, 365],
    "petasce::meevp_-1": [0, 1],
    "petasce::msalb_-1": [0, 1],
    "petasce::srad_-1": [1, 30],
    "petasce::tmax_-1": [-30, 60],
    "petasce::tmin_-1": [-30, 60],
    "petasce::xhlai_-1": [0, 20],
    "petasce::tdew_-1": [-30, 60],
    "petasce::windht_-1": [
        0.1,
        10,
    ],  # HACK: has a hole in 0 < x < 1 for petasce__assign__wind2m_1
    "petasce::windrun_-1": [0, 900],
    "petasce::xlat_-1": [3, 12],  # HACK: south sudan lats
    "petasce::xelev_-1": [0, 6000],
    "petasce::canht_-1": [0.001, 3],
}

PRESETS = {
    "petpt::msalb_-1": 0.5,
    "petpt::srad_-1": 10,
    "petpt::tmax_-1": 20,
    "petpt::tmin_-1": 10,
    "petpt::xhlai_-1": 10,
    "petasce::msalb_-1": 0.5,
    "petasce::srad_-1": 15,
    "petasce::tmax_-1": 10,
    "petasce::tmin_-1": -10,
    "petasce::xhlai_-1": 10,
}

COVERS = {
    "petasce::canht_-1": 2,
    "petasce::meevp_-1": "A",
    "petasce::cht_0": 0.001,
    "petasce::cn_4": 1600.0,
    "petasce::cd_4": 0.38,
    "petasce::rso_0": 0.062320,
    "petasce::ea_0": 7007.82,
    "petasce::wind2m_0": 3.5,
    "petasce::psycon_0": 0.0665,
    "petasce::wnd_0": 3.5,
}


def get_grfn_surface_plot(G, presets=PRESETS, num_samples=10):
    try:
        X, Y, Z, x_var, y_var = G.S2_surface((8, 6), BOUNDS, presets,
                num_samples=num_samples)
        z_data = pd.DataFrame(Z, index=X, columns=Y)
        data = [go.Surface(z=z_data.as_matrix(), colorscale="Viridis")]
        layout = json.dumps(
            dict(
                title="Sensitivity surface",
                scene=dict(
                    xaxis=dict(title=x_var.split("::")[1]),
                    yaxis=dict(title=y_var.split("::")[1]),
                    zaxis=dict(title=G.output_node.split("::")[1]),
                ),
                autosize=True,
                width=800,
                height=800,
                margin=dict(l=65, r=50, b=65, t=90),
            )
        )

        graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    except KeyError:
        flash(
            "Bounds information was not found for some variables,"
            "so the S2 sensitivity surface cannot be produced for this "
            "code example."
        )
        graphJSON = "{}"
        layout = "{}"
    return graphJSON, layout

def get_fib_surface_plot(G, covers, num_samples=10):
    X, Y, Z, x_var, y_var = G.S2_surface((8, 6), BOUNDS, PRESETS, covers,
            num_samples=num_samples)
    z_data = pd.DataFrame(Z, index=X, columns=Y)
    data = [go.Surface(z=z_data.as_matrix(), colorscale="Viridis")]
    layout = json.dumps(
        dict(
            title="Sensitivity surface",
            scene=dict(
                xaxis=dict(title=x_var.split("::")[1]),
                yaxis=dict(title=y_var.split("::")[1]),
                zaxis=dict(title=G.output_node.split("::")[1]),
            ),
            autosize=True,
            width=600,
            height=600,
            margin=dict(l=65, r=50, b=65, t=90),
        )
    )
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON, layout
