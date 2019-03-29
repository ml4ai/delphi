Running the webapp locally
--------------------------

To run the webapp locally, you need to have [Delphi](https://github.com/ml4ai/delphi) on
your system and in your `PYTHONPATH`. Additional requirements are listed in
`requirements.txt` and can be installed by invoking

```
pip install -r requirements.txt
```

To run the webapp, invoke `source run.sh` and navigate to `localhost:5000` in
your web browser.


Setting up on a SISTA VM
-----------------------------------------

```bash
sudo apt-get update
sudo apt-get install git
git clone https://github.com/ml4ai/automates
source automates/demo/setup_sista_vm.sh
```

Then do:

```bash
sudo vi /etc/apache2/sites-enabled/000-default.conf
```

Add the following lines at the top of the file (replace `adarsh` by your username if necessary):

```
LoadModule wsgi_module "/home/adarsh/anaconda3/lib/python3.7/site-packages/mod_wsgi/server/mod_wsgi-py37.cpython-37m-x86_64-linux-gnu.so"
WSGIPythonHome "/home/adarsh/anaconda3"
```

then, add the following lines under line `DocumentRoot /var/www/html`

```
    WSGIDaemonProcess automates_demo threads=5 python-path=/home/adarsh/anaconda3/lib/python3.7/site-packages user=www-data group=www-data
    WSGIScriptAlias / /var/www/html/automates_demo/app.wsgi

    <Directory automates_demo>
        WSGIProcessGroup automates_demo
        WSGIApplicationGroup %{GLOBAL}
        Order deny,allow
        Allow from all
    </Directory>

```

Then restart the server.

```
sudo apachectl restart
```


