import boto3
import requests

profile = "wmuser"
bucket_name = "world-modelers"
session = boto3.Session(profile_name=profile)

s3 = session.resource("s3")
s3_client = session.client("s3")
bucket = s3.Bucket(bucket_name)
s3_key = "indra_models/dart-20190905-stmts-location_and_time/statements.json"
bucket.download_file(s3_key, "statements.json")
