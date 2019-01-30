import sys
import pickle

with open(sys.argv[1], 'rb') as f:
    G = pickle.load(f)
    G.to_sql()
