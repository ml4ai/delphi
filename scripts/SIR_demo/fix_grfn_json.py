import sys
import json

filename = sys.argv[1]
iir = json.load(open(filename, "r"))
for i, container in enumerate(iir["containers"]):
    if container["return_value"] is None:
        iir["containers"][i]["return_value"] = []
    else:
        iir["containers"][i]["return_value"] = [container["return_value"]]
    for j, stmt in enumerate(container["body"]):
        if stmt["output"] is None:
            iir["containers"][i]["body"][j]["output"] = []
        else:
            iir["containers"][i]["body"][j]["output"] = [stmt["output"]]
json.dump(iir, open(f"fixed--{filename}", "w"))
