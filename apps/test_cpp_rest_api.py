import os 
import time
import json

SQLITE3_DB_PATH="/Users/aishwaya/Desktop/delphi/delphi_test.db"

def main():

    os.system("curl -X POST \"http://localhost:8123/delphi/create-model\" -d @../tests/data/delphi/causemos_create-model.json --header \"Content-Type: application/json\" ;")

    result_exp = os.popen("curl -X POST \"http://localhost:8123/delphi/models/XYZ/experiments\" -d @../tests/data/delphi/causemos_experiments_projection_input.json --header \"Content-Type: application/json\" ").read()

    experiment_id1 = json.loads(result_exp)["experimentId"]


    print("**************** result_exp res *********************** {}".format(experiment_id1))

    status = "in progress"
    while status == "in progress":
    	time.sleep(1)
    	rv = os.popen("curl \"http://localhost:8123/delphi/models/XYZ/experiments/"+experiment_id1+"\" ").read()
    	status = json.loads(rv)["status"]
    	print(status)


if __name__ == "__main__":
    main()