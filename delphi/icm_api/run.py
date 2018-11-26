# Run a test server.
from delphi.icm_api import create_app
import subprocess

if __name__ == "__main__":
    
    with open('./api.py', 'r') as file:
        data = file.readlines()
    
    data[155] = 'celery = make_celery(create_app())\n'
    
    with open('./api.py', 'w') as file:
        file.writelines(data)
    
    print ("Restored ./api.py")
    
    subprocess.call("for pid in $(ps -ef | grep celery | awk '{print $2}'); do kill -9 $pid; done", shell=True)
    
    print ("Rerun worker in background.")
    p = subprocess.Popen(["pipenv run celery -A  delphi.icm_api.api.celery worker"], shell=True)
    app = create_app()
    app.run(host="127.0.0.1", port=5000)

