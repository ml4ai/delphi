# Run a test server.
from delphi.icm_api import create_app
from celery import Celery

app = create_app()

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

celery = make_celery(app)
app.run(host="127.0.0.1", port=5000)
