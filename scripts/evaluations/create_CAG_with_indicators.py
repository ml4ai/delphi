import sys
import pickle


def create_CAG_with_indicators(input, output, filename="CAG_with_indicators.pdf"):
    """ Create a CAG with mapped indicators """
    with open(input, "rb") as f:
        G = pickle.load(f)
    G.map_concepts_to_indicators()
    G.set_indicator("UN/events/weather/precipitation", "Historical Average Total Daily Rainfall (Maize)", "DSSAT")
    G.set_indicator("UN/events/human/agriculture/food_production",
            "Historical Production (Maize)", "DSSAT")
    G.set_indicator("UN/entities/human/food/food_security", "IPC Phase Classification", "FEWSNET")
    G.set_indicator("UN/entities/food_availability", "Production, Meat indigenous, total", "FAO")
    G.set_indicator("UN/entities/human/financial/economic/market", "Inflation Rate", "ieconomics.com")
    G.set_indicator("UN/events/human/death", "Battle-related deaths", "WDI")
    with open(output, "wb") as f:
        pickle.dump(G, f)

if __name__ == "__main__":
    create_CAG_with_indicators(sys.argv[1], sys.argv[2])
