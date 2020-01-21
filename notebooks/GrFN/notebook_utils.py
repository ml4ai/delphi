import os
import sys
import json
import subprocess as sp


SOURCE_FILES = "files/"


def run_TR_pipeline(basename):
    cur_dir = os.getcwd()
    os.chdir(os.path.join(os.environ["AUTOMATES_LOC"], "text_reading/"))
    sp.run([
        "sbt",
        "-Dconfig.file=" + os.path.join(cur_dir, "files/SIR-simple.conf"),
        'runMain org.clulab.aske.automates.apps.ExtractAndAlign'
    ])
    sys.exit()
    os.chdir(cur_dir)
    tr_json_path = os.path.join(SOURCE_FILES, f"{basename}_with_groundings.json")
    norm_json_path = os.path.join(SOURCE_FILES, f"{basename}.json")
    if os.path.isfile(norm_json_path):
        os.remove(norm_json_path)
    if os.path.isfile(tr_json_path):
        os.rename(tr_json_path, norm_json_path)
    grfn = json.load(open(norm_json_path, "r"))
    return grfn


run_TR_pipeline("SIR-simple_GrFN")
