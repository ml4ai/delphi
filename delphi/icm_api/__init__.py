from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# Import SQLAlchemy

# Define the application object
# app = Flask(__name__)

# Define the database objects which is imported
# by modules and controllers
db = SQLAlchemy(app)
from delphi.icm_api.api import bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(bp)
    return app
