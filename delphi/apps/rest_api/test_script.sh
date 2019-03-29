# Testing script for ICM API

port=5000

echo "Getting ICM ID"
icm_id=`curl -s localhost:${port}/icm | sed -n 2p | cut -d '"' -f 2`

curl -s localhost:$port/icm/$icm_id/primitive -o primitives.json

echo "Getting id of precipitation node"
precipitation_id=`cat primitives.json | python3 -c "import sys, json;
primitives = json.load(sys.stdin);
precipitation, = [p for p in primitives if p['label'] == 'UN/events/weather/precipitation'];
print(precipitation['id'])"`

post_request='{
    "baseType": "ForwardProjection",
    "interventions": [ {
        "id": '\"${precipitation_id}\"',
        "values": { 
            "active": "ACTIVE",
            "time": "2019-01-31",
            "value": {
                "baseType": "FloatValue",
                "value": 0.4
                }
            }
        }],
    "projection": {
        "startTime": "2019-01-31",
        "numSteps": 6,
        "stepSize": "MONTH"
    },
    "options": {
        "timeout": 60000
    }
}'

echo $post_request > post_request.json

echo "Sending forward projection experiment"
rm bmi_config.txt
experiment_id=`curl -s -d @post_request.json\
               -H "Content-Type:application/json"\
               http://localhost:$port/icm/$icm_id/experiment\
               | sed -n 2p | cut -d '"' -f 4`


echo "Getting experiment results"
curl -s http://localhost:$port/icm/$icm_id/experiment/$experiment_id -o results.json
python make_plots.py
