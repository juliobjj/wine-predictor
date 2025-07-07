from sqlalchemy import Column, Integer, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class WineSample(Base):
    __tablename__ = 'wine_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    fixed_acidity = Column(Float)
    volatile_acidity = Column(Float)
    citric_acid = Column(Float)
    residual_sugar = Column(Float)
    chlorides = Column(Float)
    free_sulfur_dioxide = Column(Float)
    total_sulfur_dioxide = Column(Float)
    density = Column(Float)
    ph = Column(Float)
    sulphates = Column(Float)
    alcohol = Column(Float)
    predicted_quality = Column(Integer)

# Conexão com o SQLite
engine = create_engine('sqlite:///wine_data.db')
Base.metadata.create_all(engine)

SessionLocal = sessionmaker(bind=engine)
