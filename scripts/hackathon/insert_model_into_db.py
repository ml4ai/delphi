import sys
import pickle
from datetime import date

with open(sys.argv[1], 'rb') as f:
    G = pickle.load(f)
    G.to_sql(last_known_value_date = date(2017,4,1))
