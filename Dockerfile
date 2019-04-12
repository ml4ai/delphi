# Use an official Python runtime as a parent image
from python:3.6
run apt-get update; apt-get install -y openjdk-8-jre

workdir app
copy setup.py /app
copy delphi /app/delphi
run pip install -e .

# Make port 80 available to the world outside this container
expose 80

# Run app.py when the container launches
cmd ["codex"]
