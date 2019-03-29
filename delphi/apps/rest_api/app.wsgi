import os
import sys
import logging

logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/html/delphi/delphi/rest_api")
sys.path.insert(0,"/var/www/html/delphi")

from delphi.rest_api import create_app

application = create_app()
