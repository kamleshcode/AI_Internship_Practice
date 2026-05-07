"""
SQLAlchemy : is a python Database toolkit or high level python library that help you work with DB using python instead
             of writing raw SQL Queries.

it has two main parts:
1. SQLAlchemy Core
2. SQLAlchemy ORM

SQLAlchemy Architecture
        python code -> SQLAlchemy ORM/Core -> Engine -> Connection Pooling -> DBAPI Drivers (pyodbc) ->Database

Transaction flow
        Session -> Ask Engine -> need connection
        Engine  -> Ask Pool -> give connection
        Session -> Use Connection -> run queries
        session -> commit() and rollback()
        session -> close

"""
from sqlalchemy.orm import sessionmaker,declarative_base
from sqlalchemy import create_engine, Column, Integer, String, Float

engine = create_engine('mssql+pyodbc://localhost/kamlesh?driver=ODBC+Driver+17+for+SQL+Server',
                       pool_size=2,    # Keep 2
                       max_overflow=1, # Allow 1 extra
                       pool_timeout=30,# wait for 30sec if pool is full
                       echo=True)

Base = declarative_base()

class Employee(Base):
    __tablename__ = 'practice'
    id = Column(Integer, primary_key=True)
    name = Column(String(10))
    salary = Column(Float)

# Create Table
Base.metadata.create_all(engine)

# Session
Session = sessionmaker(bind=engine)
session = Session()

# Insert Data
emp = Employee( id=1, name='Kamlesh', salary=50000)
session.add(emp)
session.commit()

session.close()
