from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime


class ModelRuntime(Base):
    __tablename__ = 'model_runtime'

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(Integer, ForeignKey('devices.id'))
    workflow_id = Column(Integer, ForeignKey('workflow.id'))
    start_time = Column(DateTime, default=datetime.utcnow)
    last_stop_time = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50))
    
    # Relationships
    device = relationship("Devices", back_populates="model_runtimes")
    workflow = relationship("Workflow", back_populates="model_runtimes")

class Devices(Base):
    __tablename__ = 'devices'

    id = Column(Integer, primary_key=True, index=True)
    device_ip = Column(String(125), unique=True)
    location = Column(String(125))
    created_on = Column(DateTime,default=datetime.utcnow)
    user_id = Column(Integer)

    model_runtimes = relationship("ModelRuntime", back_populates="device")
  

    
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

class Workflow(Base):
    __tablename__ = 'workflow'
    id = Column(Integer, primary_key=True, index= True)
    name = Column(String(125))
    description = Column(String(1000))
    version = Column(String(125))
    status = Column(Boolean)
    created_on = Column(DateTime)
    last_updated_on = Column(DateTime)

    model_runtimes = relationship("ModelRuntime", back_populates="workflow")