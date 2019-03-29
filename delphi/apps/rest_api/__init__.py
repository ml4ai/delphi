from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app(debug=False):
    from delphi.apps.rest_api.api import bp

    app = Flask(__name__)
    app.config.from_object("delphi.apps.rest_api.config")
    app.debug=debug
    db.init_app(app)
    app.register_blueprint(bp)
    return app
