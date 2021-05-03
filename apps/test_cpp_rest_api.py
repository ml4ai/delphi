import os 
import time
import json


#SQLITE3_DB_PATH="${DELPHI_DB}"
#SELECT * FROM delphimodel where id='XYZ';

SQLITE3_DB_PATH="/Users/aishwaya/Desktop/delphi/delphi_test.db"




def main():
	#os.system("sqlite3 "+SQLITE3_DB_PATH+" '.mode json'  \"SELECT * FROM delphimodel where id='XYZ'\"")

    os.system("curl -X POST \"http://localhost:8123/delphi/create-model\" -d @../tests/data/delphi/causemos_create-model.json --header \"Content-Type: application/json\" ;")

    result_exp = os.popen("curl -X POST \"http://localhost:8123/delphi/models/XYZ/experiments\" -d @../tests/data/delphi/causemos_experiments_projection_input.json --header \"Content-Type: application/json\" ").read()

    experiment_id1 = json.loads(result_exp)["experimentId"]


    print("**************** result_exp res *********************** {}".format(experiment_id1))

    status = "in progress"
    while status == "in progress":
    	time.sleep(1)
    	#result_get = os.system("curl \"http://localhost:8123/delphi/models/XYZ/experiments/d93b18a7-e2a3-4023-9f2f-06652b4bba66\" ")
    	rv = os.popen("curl \"http://localhost:8123/delphi/models/XYZ/experiments/"+experiment_id1+"\" ").read()
    	status = json.loads(rv)["status"]
    	print(status)


if __name__ == "__main__":
    main()