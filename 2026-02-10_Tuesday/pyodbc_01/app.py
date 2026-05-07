"""
Pyodbc : Is a low level python library/ module that allows python program to connect directly to database using ODBC Drivers and
         execute SQL Query manually

(Direct Connection + Cursor)

Architecture:
            python code -> pyodbc -> ODBC Driver -> Database

Transaction Flow:
   connect() -> cursor() -> execute() -> commit() -> rollback() -> close()

"""

import pyodbc
# print(pyodbc.drivers())

# Create a connection
conn = pyodbc.connect("DRIVER={ODBC Driver 17 for SQL Server};"
                      "SERVER={1R-79};"
                      "DATABASE={kamlesh};"
                      "trusted_Connection=yes;")

# Create Cursor
cursor = conn.cursor()

# Insert data to users table in db
cursor.execute("INSERT INTO users(name, email) VALUES('Om', 'om@example.com')")
# Delete user name Om
cursor.execute("DELETE FROM users where name = 'Om'")

#Commit
conn.commit()

#close Everything
cursor.close()
conn.close()