from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

# Define base class
Base = declarative_base()

# Connection string using ODBC
engine = create_engine(
    "mssql+pyodbc://localhost/kamlesh?driver=ODBC+Driver+17+for+SQL+Server",
    echo=True
)
print("Connected to SQLAlchemy.")


Session = sessionmaker(bind=engine)
session = Session()
print("Session created")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    email = Column(String(100))

    def __repr__(self):
        return f"<User(id={self.id}, name='{self.name}', email='{self.email}')>"

#Create Table
Base.metadata.create_all(engine)

try:
    users_to_add = [
        User(name="Kamlesh", email="kamlesh@example.com"),
        User(name="Ravi", email="ravi@example.com"),
        User(name="Sneha", email="sneha@example.com"),
        User(name="Amit", email="amit@example.com")
    ]

    session.add_all(users_to_add)
    session.commit()
    print("Multiple users inserted successfully")

except Exception as e:
    session.rollback()
    print("Error inserting multiple users:", e)


#Query Data
users = session.query(User).all()
for u in users:
    print(u)

# Update
user = session.query(User).filter_by(name="Kamlesh").first()
user.email = "new@example.com"
session.commit()

# Delete
session.delete(user)
session.commit()




