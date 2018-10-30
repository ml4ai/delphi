from flask import Flask
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


def create_app():
    from delphi.icm_api.api import bp

    app = Flask(__name__)
    app.config.from_object("delphi.icm_api.config")
    db.init_app(app)
    app.register_blueprint(bp)
    return app

