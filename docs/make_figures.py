from indra.sources import eidos

ep = eidos.process_text 
    "Significantly increased conflict seen in "
    "South Sudan forced many families to flee in 2017."
)
G.assemble_transition_model_from_gradable_adjectives()
G.map_concepts_to_indicators()
G.parameterize(country="South Sudan", year=2017, month=4)
A = G.to_agraph(indicators=True, indicator_values=True)
A.draw("CAG.png", prog="dot")
