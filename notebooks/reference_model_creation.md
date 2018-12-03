```python
%load_ext autoreload
%autoreload 2
import json
from tqdm import tqdm
from delphi.AnalysisGraph import AnalysisGraph
from delphi.icm_api.models import *
from delphi.utils.indra import (get_statements_from_json,
                                influence_stmt_from_dict)
from delphi.subgraphs import (get_subgraph_for_concept,
                             get_subgraph_for_concept_pairs,
                             get_subgraph_for_concept_pair)
from delphi.visualization import visualize
from delphi.assembly import is_well_grounded, is_grounded
from delphi.manipulation import merge_nodes
from delphi.assembly import get_valid_statements_for_modeling
```

```python
from delphi.paths import data_dir
with open(data_dir/"3-Indra16k.json", "r") as f:
    json_list = [d for d in json.load(f) if d["type"] == "Influence"]

sts = [influence_stmt_from_dict(d) for d in tqdm(json_list)]
valid_sts_for_modeling = get_valid_statements_for_modeling(sts)
supported_sts = [s for s in valid_sts_for_modeling if len(s.evidence) >= 2]
```

```python
well_grounded_sts = [s for s in supported_sts if is_well_grounded(s, 0.8)]
```

```python
list(map(len, [sts, valid_sts_for_modeling, supported_sts,
               well_grounded_sts]))
```

```python
G = AnalysisGraph.from_statements(well_grounded_sts)
```

```python
G = get_subgraph_for_concept(G, "food_insecurity", 2)
```

```python
visualize(G, layout="dot")
```

```python
import pickle
with open('example_delphi_model.pkl', 'wb') as f:
    pickle.dump(G, f)
```

```python
from delphi.icm_api.models import *
from delphi.icm_api import create_app
from datetime import date
from delphi.random_variables import LatentVar
import numpy as np

G.assemble_transition_model_from_gradable_adjectives()
delphi_model = DelphiModel(id = G.id, model = G)
icm_metadata = ICMMetadata(
        id=G.id,
        created=date.today().isoformat(),
        estimatedNumberOfPrimitives=len(G.nodes) + len(G.edges),
        createdByUser_id=1,
        lastAccessedByUser_id=1,
        lastUpdatedByUser_id=1,
    )
today = date.today().isoformat()
default_latent_var_value = 1.0
causal_primitives = []
for n in G.nodes(data=True):
    n[1]["rv"] = LatentVar(n[0])
    n[1]["update_function"] = G.default_update_function
    rv = n[1]["rv"]
    rv.dataset = [default_latent_var_value for _ in range(G.res)]
    if n[1].get("indicators") is not None:
        for ind in n[1]["indicators"]:
            ind.dataset = np.ones(G.res) * ind.mean

    causal_variable = CausalVariable(
        id=n[1]["id"],
        model_id=G.id,
        units="",
        namespaces={},
        auxiliaryProperties=[],
        label=n[0],
        description=f"Long description of {n[0]}.",
        lastUpdated=today,
        confidence=1.0,
        lastKnownValue={
            "active": "ACTIVE",
            "trend": None,
            "time": today,
            "value": {
                "baseType": "FloatValue",
                "value": n[1]["rv"].dataset[0],
            },
        },
        range={
            "baseType": "FloatRange",
            "range": {"min": 0, "max": 10, "step": 0.1},
        },
    )
    causal_primitives.append(causal_variable)

max_evidences = max(
    [
        sum([len(s.evidence) for s in e[2]["InfluenceStatements"]])
        for e in G.edges(data=True)
    ]
)
max_mean_betas = max(
    [abs(np.median(e[2]["betas"])) for e in G.edges(data=True)]
)
for e in G.edges(data=True):
    causal_relationship = CausalRelationship(
        id=str(uuid4()),
        namespaces={},
        source={"id": G.nodes[e[0]]["id"], "baseType": "CausalVariable"},
        target={"id": G.nodes[e[1]]["id"], "baseType": "CausalVariable"},
        model_id=G.id,
        auxiliaryProperties=[],
        lastUpdated=today,
        types=["causal"],
        description=f"{e[0]} influences {e[1]}.",
        confidence=np.mean(
            [s.belief for s in e[2]["InfluenceStatements"]]
        ),
        label="influences",
        strength=abs(np.median(e[2]["betas"]) / max_mean_betas),
        reinforcement=(
            True
            if np.mean(
                [
                    stmt.subj_delta["polarity"]
                    * stmt.obj_delta["polarity"]
                    for stmt in e[2]["InfluenceStatements"]
                ]
            )
            > 0
            else False
        ),
    )
    causal_primitives.append(causal_relationship)
    
app = create_app()
app.testing = True

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///delphi.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True
with app.app_context():
    db.create_all()
    db.session.add(icm_metadata)
    db.session.add(delphi_model)
    for causal_primitive in causal_primitives:
        db.session.add(causal_primitive)
    db.session.commit()
```
