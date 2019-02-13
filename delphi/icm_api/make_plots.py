import json
from math import exp
from matplotlib import pyplot as plt

if __name__ == "__main__":
    with open("primitives.json", "r") as f:
        primitives = {x['label']:x for x in json.load(f)}

    with open("results.json", "r") as f:
        results = json.load(f)["results"]

    ys_precip = [r['intervened']['value']['value'] for r in results if r['id'] == primitives['UN/events/weather/precipitation']['id']]
    ys_fp = [r['intervened']['value']['value'] for r in results if r['id'] == primitives['UN/events/human/agriculture/food_production']['id']]
    tau=1.0
    ys_derivs = [-0.6*exp(-tau*i) for i in range(len(ys_precip))]
    xs = range(len(ys_precip))
    plt.style.use('ggplot')
    plt.plot(xs, ys_precip, label="precipitation")
    plt.plot(xs, ys_fp, label="food_production")
    plt.plot(xs, ys_derivs, label="∂(precipitation)/∂t")
    plt.legend()
    plt.savefig('experiment_timeseries.png')
