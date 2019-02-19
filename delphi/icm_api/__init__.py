from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from delphi.db import db_session

db = SQLAlchemy()

def create_app(debug=False):
    from delphi.icm_api.api import bp

    app = Flask(__name__)
    app.config.from_object("delphi.icm_api.config")
    app.debug=debug
    db.init_app(app)
    app.register_blueprint(bp)
    return app
