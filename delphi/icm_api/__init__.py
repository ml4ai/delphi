from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from celery import Celery

db = SQLAlchemy()
 
def make_celery(app):
    """Create an celery instance and set its configuration"""
    celery = Celery(
            app.import_name,
            backend=app.config['CELERY_RESULT_BACKEND'],
            broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)
    
    class ContextTask(celery.Task):
        """Integrate Celery with Flask"""
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    
    celery.Task = ContextTask
    return celery

def create_app():
    from delphi.icm_api.api import bp
    """Create an Flask app"""
    app = Flask(__name__)
    app.config.from_object("delphi.icm_api.config")
    db.init_app(app)
    app.register_blueprint(bp)
    return app

