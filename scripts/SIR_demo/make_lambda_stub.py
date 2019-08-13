import sys
import json


iir = json.load(open("SIR-Gillespie-SD_GrFN.json", "r"))
with open("SIR-Gillespie-SD_lambdas.py", "w") as outfile:
    for container in iir["containers"]:
        for stmt in container["body"]:
            if stmt["function"]["type"] == "lambda_source":
                fn_name = stmt["function"]["name"]
                if "__decision__" in fn_name:
                    clean_inputs = list()
                    for i, inp in enumerate(stmt["input"]):
                        strs = inp.split("::")
                        if i < 2:
                            clean_inputs.append(f"{strs[1]}_{i}")
                        else:
                            clean_inputs.append(strs[1])
                else:
                    clean_inputs = [i.split("::")[1] for i in stmt["input"]]
                inputs = ", ".join(clean_inputs)
                outfile.write(f"def {fn_name}({inputs}):\n    return None\n\n\n")
