import sys
import json
import pickle

with open(sys.argv[1], 'rb') as f:
    G = pickle.load(f)


G.create_bmi_config_file("build/bmi_config.txt")
with open(sys.argv[2], 'w') as f:
    f.write(json.dumps(G.to_dict(), indent=2))
