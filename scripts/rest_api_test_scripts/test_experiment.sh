# Testing script for running experiments with Delphi ICM API (production)

# Set the ICM API endpoint (if running the Flask app locally, you can set it to
# localhost:5000 (or whatever port you've set it to)

icm_endpoint=http://vanga.sista.arizona.edu/delphi/icm

# Get the ICM ID
icm_id=`curl -s $icm_endpoint | cut -d '"' -f 2`
curl -s $icm_endpoint/$icm_id/primitive -o primitives.json

# Construct the post request
post_request='{
  "baseType": "ForwardProjection",
  "interventions": [],
  "projection": {
    "startTime": "2017-05-21",
    "numSteps": 6,
    "stepSize": "MONTH"
  },
  "options": {
    "timeout": 60000
  }
}'

echo $post_request > post_request.json

# Get experiment results
experiment_id=`curl -s -d @post_request.json\
               -H "Content-Type:application/json"\
               $icm_endpoint/$icm_id/experiment\
               | cut -d '"' -f 4`
rm post_request.json
curl -s $icm_endpoint/$icm_id/experiment/$experiment_id
