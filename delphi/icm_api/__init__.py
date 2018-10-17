from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from delphi.icm_api.api import bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(bp)
    return app
