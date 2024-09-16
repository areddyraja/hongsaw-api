from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


engine = create_engine('sqlite:///localdata.db')

SessionLocal = sessionmaker(autocommit= False, autoflush= False, bind=engine)
session = SessionLocal()

Base = declarative_base()