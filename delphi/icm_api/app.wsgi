import os
import sys
import logging

logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/html/delphi/delphi/icm_api")
sys.path.insert(0,"/var/www/html/delphi")

from delphi.icm_api import create_app

application = create_app()
