from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from celery import Celery

db = SQLAlchemy()
 

# Integrate Celery with Flask
def make_celery(app):
    celery = Celery(
            app.import_name,
            backend=app.config['CELERY_RESULT_BACKEND'],
            broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)
    
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    
    celery.Task = ContextTask
    return celery

def create_app():
    from delphi.icm_api.api import bp

    app = Flask(__name__)
    app.config.from_object("delphi.icm_api.config")
    db.init_app(app)
    app.register_blueprint(bp)
    return app

