# Run a test server.
from delphi.icm_api import create_app
import subprocess

def switchConfig(whetherUsedForTest):
   '''Switch the app config from test use to normal use, or vice versa'''
    
   with open('./api.py', 'r') as file:
        data = file.readlines()
   if whetherUsedForTest:
        for i in range(len(data)):
            if data[i] == 'celery = make_celery(create_app())\n':
                data[i] = 'celery = make_celery(create_test_app())\n'
                print ("Changed for test.")
   else:
        for i in range(len(data)):
            if data[i] == 'celery = make_celery(create_test_app())\n':
                data[i] = 'celery = make_celery(create_app())\n'
                print ("Changed for normal.")
    
   with open('./api.py', 'w') as file:
        file.writelines(data)

	
		
if __name__ == "__main__":
    
    #with open('./api.py', 'r') as file:
        #data = file.readlines()
    
    #data[163] = 'celery = make_celery(create_app())\n'
    
    #with open('./api.py', 'w') as file:
        #file.writelines(data)
    
    #print ("Restored ./api.py")
    switchConfig(False)
    subprocess.call("for pid in $(ps -ef | grep celery | awk '{print $2}'); do kill -9 $pid; done", shell=True)
    
    #print ("Rerun worker in background.")
    p = subprocess.Popen(["pipenv run celery -A  delphi.icm_api.api.celery worker"], shell=True)
    app = create_app()
    app.run(host="127.0.0.1", port=5000)

    switchConfig(True)

    

