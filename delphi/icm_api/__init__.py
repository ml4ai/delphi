from flask import Flask

# Import SQLAlchemy
from flask_sqlalchemy import SQLAlchemy

# Define the application object
app = Flask(__name__)

# Define the database objects which is imported
# by modules and controllers
db = SQLAlchemy(app)
