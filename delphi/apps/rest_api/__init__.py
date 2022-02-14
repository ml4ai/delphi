from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_executor import Executor

db = SQLAlchemy()
executor = Executor()

def create_app(debug=False):
    from delphi.apps.rest_api.api import bp

    app = Flask(__name__)
    app.config.from_object("delphi.apps.rest_api.config")
    app.debug=debug
    executor.init_app(app)
    app.config["EXECUTOR_TYPE"] = "thread"
    db.init_app(app)
    app.register_blueprint(bp)
    return app
