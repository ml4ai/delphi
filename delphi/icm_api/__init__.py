from flask import Flask


def create_app():
    from delphi.icm_api.api import bp

    app = Flask(__name__)
    db.init_app(app)
    app.register_blueprint(bp)
    return app
