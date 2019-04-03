import os
import sys
import logging

logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/html/delphi/delphi/apps/CodeExplorer")
sys.path.insert(0,"/var/www/html/delphi")

from app import app as application
