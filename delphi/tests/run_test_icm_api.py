import subprocess
import argparse
import os

# config used only for test
TEST_TEMPLATE = '''\
from pathlib import Path

# Statement for enabling the development environment
DEBUG = True

# Define the application directory
import os

BASE_DIR = Path(__file__).parent
SQLALCHEMY_DATABASE_URI = "sqlite:////tmp/test.db"
SQLALCHEMY_TRACK_MODIFICATIONS = False
DATABASE_CONNECT_OPTIONS = {}
CELERY_BROKER_URL = 'pyamqp://localhost//'
CELERY_RESULT_BACKEND = 'db+sqlite:////tmp/test.sqlite'
CELERY_TASK_SERIALIZER = 'pickle'
CELERY_ACCEPT_CONTENT = ['pickle']
# Application threads. A common general assumption is
# using 2 per available processor cores - to handle
# incoming requests using one and performing background
# operations using the other.
THREADS_PER_PAGE = 2
'''

# config used for normal running
TEMPLATE = '''\
from pathlib import Path

# Statement for enabling the development environment
DEBUG = True

# Define the application directory
import os

BASE_DIR = Path(__file__).parent

# Define the database - we are working with
# SQLite for this example
SQLALCHEMY_DATABASE_URI = f"sqlite:///{BASE_DIR}/delphi.db"
SQLALCHEMY_TRACK_MODIFICATIONS = False
DATABASE_CONNECT_OPTIONS = {}
CELERY_BROKER_URL = 'pyamqp://localhost//'
CELERY_RESULT_BACKEND = f"db+sqlite:///{BASE_DIR}/delphi.sqlite"
CELERY_TASK_SERIALIZER = 'pickle'
CELERY_ACCEPT_CONTENT = ['pickle']
# Application threads. A common general assumption is
# using 2 per available processor cores - to handle
# incoming requests using one and performing background
# operations using the other.
THREADS_PER_PAGE = 2
'''
if __name__ == '__main__':
    
    # write the test config to ../icm_api/config.py
    with open("../icm_api/config.py","w") as f:
        f.write(TEST_TEMPLATE)
    
    print ("Modified ../icm_api/config.py to be used for testing.")

    # activate the worker in a subprocess
    print ("Running worker in background.")
    p = subprocess.Popen(["pipenv run celery -A  delphi.icm_api.api.celery worker"], shell=True)
 
    # run the tests, also test for timeout
    try:
        subprocess.call("pipenv run pytest -s test_icm_api.py", shell=True, timeout=20)
    except subprocess.TimeoutExpired:
        print("Tests time out.")
    
    # terminate the worker subprocess
    print ("Terminate the worker in the background")
    p.terminate()
    
    # rewrite the config for normal running to config.py, overwrite the old one for test
    with open("../icm_api/config.py","w") as s:
        s.write(TEMPLATE)
    
    print ("Restored ../icm_api/config.py")
    
    # kill those celery process stilling running in the background by their pids
    subprocess.call("for pid in $(ps -ef | grep celery | awk '{print $2}'); do kill -9 $pid; done", shell=True)
