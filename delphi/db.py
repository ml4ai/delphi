from .paths import db_path
from sqlalchemy import create_engine

engine = create_engine(f"postgresql://{str(db_path)}", echo=False)
