from sqlalchemy import Column, Integer, String, DateTime
from .database import Base
from datetime import datetime


class Devices(Base):
    __tablename__ = 'devices'

    id = Column(Integer, primary_key=True, index=True)
    device_ip = Column(String(125), unique=True)
    location = Column(String(125))
    created_on = Column(DateTime,default=datetime.utcnow)
    user_id = Column(Integer)
  

    
class DwellTime(Base):
    __tablename__ = 'dwelltime'

    id = Column(Integer, primary_key=True, index=True)
    left_time = Column(DateTime)
    return_time = Column(DateTime)
    dwell_time = Column(String(125))
    device_start_time = Column(DateTime)
    device_end_time = Column(DateTime)
    device_id = Column(Integer)
    
class HelmetDetection(Base):
    __tablename__ = 'helmet-detection'

    id = Column(Integer, primary_key=True, index= True)
    obj_id = Column(Integer)
    helmet_status = Column(String(125))
    device_start_time = Column(DateTime)
    device_end_time = Column(DateTime)
    device_id = Column(Integer)
    