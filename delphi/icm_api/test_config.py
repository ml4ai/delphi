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
