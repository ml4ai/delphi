from .paths import db_path
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine(f"sqlite:///{str(db_path)}", echo=False)
db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False,
    bind=engine))

Base = declarative_base()
Base.query = db_session.query_property() 

def init_db():
    import delphi.icm_api.models
    Base.metadata.create_all(bind=engine)
