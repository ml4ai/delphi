import pickle
import sys
import os

from grfn_walker import to_agraph


filename = sys.argv[1]
lambda_trees = pickle.load(open(filename, "rb"))
header = filename.split("--")[0]
os.makedirs(header, exist_ok=True)
for (name, tree) in lambda_trees:
    A = to_agraph(tree)
    A.draw(f'{header}/{name}.pdf', prog='dot')
