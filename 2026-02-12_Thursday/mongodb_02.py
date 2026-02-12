import pymongo

def add_records(collection):
    payroll_records = [
        {
            'pay_id': 'P001',
            'emp_id': 'I001',
            'salary': 50000.00,
            'bonus': 2500.00,
            'health_insurance_plan': 'Standard'
        },
        {
            'pay_id': 'P002',
            'emp_id': 'I002',
            'salary': 55000.00,
            'bonus': 3000.00,
            'health_insurance_plan': 'Premium'
        },
        {
            'pay_id': 'P003',
            'emp_id': 'I003',
            'salary': 48000.00,
            'bonus': 2000.00,
            'health_insurance_plan': 'Basic'
        },
        {
            'pay_id': 'P004',
            'emp_id': 'I004',
            'salary': 60000.00,
            'bonus': 5000.00,
            'health_insurance_plan': 'Premium'
        },
        {
            'pay_id': 'P005',
            'emp_id': 'I005',
            'salary': 45000.00,
            'bonus': 1500.00,
            'health_insurance_plan': 'Standard'
        }
    ]

    payroll_records=collection.insert_many(payroll_records)
    print(f"{len(payroll_records.inserted_ids)}Records Inserted Successfully....")

def update_records(collection):
    print("records of employee where bonus < 3000 the set performance weak")
    for update in collection.find({'bonus': {'$gte': 3000}}):
        collection.update_one({'emp_id':update["emp_id"]},{'$set':{"performance":"Bad"}})
        print(f"{update['emp_id']} Updated Successfully....")

def fetch_salary_by_dept(employees,collection):
    print("I want total salary group by department")
    pipeline=[
        {
            '$lookup':{'from':'Payroll','localField': 'id','foreignField': 'emp_id','as': 'pay_info'}
        },
        {
            '$unwind':'$pay_info'
        },
        {
            '$group' :{'_id': '$department', 'total_dept_salary':{'$sum':{ '$add' : ['$pay_info.salary', '$pay_info.bonus'] }}}
        },
        {
            '$sort':{'total_dept_salary':-1}
        }
    ]

    results = employees.aggregate(pipeline)
    print("--- Total Salary Grouped by Department ---")

    for doc in results:
        print(f"Department: {doc['_id']} | Total Salary: {doc['total_dept_salary']}")

if __name__ =='__main__':
    client=pymongo.MongoClient("mongodb://localhost:27017")
    print(f'Database :{client.list_database_names()}')
    db=client['Employees']
    print(f'Collections :{db.list_collection_names()}')
    collections = db['Payroll']
    employee=db['employeeinfo']

    # add_records(collections)
    # update_records(collections)
    fetch_salary_by_dept(employee,collections)