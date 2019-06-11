import sys
import pickle
from delphi.AnalysisGraph import AnalysisGraph
from delphi.export import to_agraph

G = AnalysisGraph.from_uncharted_json_file(sys.argv[1])

# TODO Make sure to get indicators from DSSAT
# G.map_concepts_to_indicators()

# G.set_indicator("UN/events/weather/precipitation", "Historical Average Total Daily Rainfall (Maize)", "DSSAT")
# G.set_indicator("UN/events/human/agriculture/food_production",
        # "Historical Production (Maize)", "DSSAT")
# G.set_indicator("UN/entities/human/food/food_security", "IPC Phase Classification", "FEWSNET")
# G.set_indicator("UN/entities/food_availability", "Production, Meat indigenous, total", "FAO")
# G.set_indicator("UN/entities/human/financial/economic/market", "Inflation Rate", "ieconomics.com")
# G.set_indicator("UN/events/human/death", "Battle-related deaths", "WDI")

# G.parameterize(year = 2017, month=4)
G.assemble_transition_model_from_gradable_adjectives()
G.sample_from_prior()
A = to_agraph(G)
A.draw("CauseMos_CAG.png", prog="dot")
A = to_agraph(G, indicators=True, indicator_values = True)
A.draw("CauseMos_CAG_with_indicators.png", prog="dot")

with open(sys.argv[2], 'wb') as f:
    pickle.dump(G, f)
