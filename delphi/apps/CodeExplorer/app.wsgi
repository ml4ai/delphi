import os
import sys
import logging

logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/html/automates_demo")
sys.path.insert(0,"/var/www/html/delphi")

from app import app as application
