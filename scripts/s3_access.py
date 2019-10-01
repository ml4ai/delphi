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

import argparse
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



parser = argparse.ArgumentParser(
    description="Script to interact with the World Modelers AWS S3 bucket"
)

group = parser.add_mutually_exclusive_group()
group.add_argument("--list", help="Display list of INDRA statement corpuses",
        action="store_true")

group.add_argument("--download", help="key of item you want to download")

parser.add_argument("-o","--output", help="Name of file to write to", default="data/statements.json")

args = parser.parse_args()

if args.list:
    objects = bucket.objects.filter(Prefix="indra_models")
    for x in objects:
        print(x.key)

if args.download:
    bucket.download_file(args.download, args.output)

