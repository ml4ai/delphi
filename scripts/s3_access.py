""" This script allows interaction with World Modelers data from the Amazon S3 bucket.

You will need to install boto3 if you haven't already, with `pip install boto3`.

Additionally, ensure that your credentials secret stored in the file:
~/.aws/credentials. The contents of the file should look like the following:


```
[wmuser]
aws_access_key_id = WMUSER_ACCESS_KEY
aws_secret_access_key = WMUSER_SECRET_KEY
```

"""

import boto3
import requests

# Replace 'wmuser' with whatever you have in the ~/.aws/credentials config file
# if it is different.

profile = "wmuser"
bucket_name = "world-modelers"

session = boto3.Session(profile_name=profile)
s3 = session.resource("s3")
s3_client = session.client("s3")
bucket = s3.Bucket(bucket_name)
objects = bucket.objects.filter(Prefix="indra_models")

# S3 key for the file you want to download
s3_key = "indra_models/dart-20190905-stmts-location_and_time/statements.json"

bucket.download_file(s3_key, "statements.json")
