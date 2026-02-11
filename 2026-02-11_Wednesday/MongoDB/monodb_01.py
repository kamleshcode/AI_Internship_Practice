import pymongo

def add_records(collection):
    records = [
        {
            '_id': 'I001', 'firstname': 'Kamlesh', 'lastname': 'Patel', 'department': 'AI', 'qualification': 'BTech', 'age':'21','Marks': {'Statistics': 89,'Python': 50,'SQLAlchemy': 68},'Status':'Pass'
        },
        {
            '_id': 'I002', 'firstname': 'Harsh', 'lastname': 'Mistry', 'department': 'HR', 'qualification': 'BTech', 'age':'21','Marks': {'Statistics': 99,'Python': 80,'SQLAlchemy': 69},'Status':'Pass'
        },
        {
            '_id': 'I003', 'firstname': 'Om', 'lastname': 'Mishra', 'department': 'ML', 'qualification': 'BTech', 'age':'20','Marks': {'Statistics': 80,'Python': 70,'SQLAlchemy': 69},'Status':'Pass'
        },
        {
            '_id': 'I004', 'firstname': 'Samarth', 'lastname': 'Prajapati', 'department': 'AI', 'qualification': 'BE', 'age':'21','Marks': {'Statistics': 90,'Python': 100,'SQLAlchemy': 99},'Status':'Pass'
        },
        {
            '_id': 'I005', 'firstname': 'Alok', 'lastname': 'Yadav', 'department': 'BA', 'qualification': 'Bsc', 'age':'18','Marks': {'Statistics': 80,'Python': 10,'SQLAlchemy': 9},'Status':'Pass'
        }
    ]
    result=collection.insert_many(records)
    print(f'Successfully Inserted {len(result.inserted_ids)} records')
    # # Query on basis o equality condition
    # for record in information.find({'id':'I005'}):
    #     print(record)

def view_records(collection):
    for record in (collection.find()): # or information.find({})
        print(record)

def query_records(collection):
    print("Query Document using Query operators($in,$lt,$gt)")

    print("Records of employee with qualification BTech and BE")
    for record in (collection.find({'qualification':{'$in':['BTech','BE']}})): print(record)

    print("Records of employee whose age > 18 and qualification BTech")
    for record in (collection.find({'qualification':'BTech','age': {'&gt' :18} })): print(record)

    # $or and $and operator
    print("Records of employee with BTech degree or AI department")
    for record in (collection.find({'$or':[{'department':'AI'},{'qualification':'BTech'}]})): print(record)
    for record in (collection.find({'$and': [{'department': 'AI'}, {'qualification': 'BTech'}]})): print(record)

    print("------------")
    for record in (collection.find({'Marks': {'Statistics': 80,'Python': 70,'SQLAlchemy': 69}})): print(record)

def update_records(collection):
    print("Performing update operation.....................")
    # update_one()
    print("Updating python subject marks of employee name 'Alok' to 99 and also add lastModified date to records ")
    updates=collection.update_one({'firstname':'Alok'},{'$set':{'Marks.Python':99},'$currentDate':{'lastModified':True}})
    print(updates)
    # update_many()
    print("Update employee Status to Fail if marks total less than 200 ")
    result = collection.update_many({},
        [{'$set': {'Status': {'$cond': {'if': {'$lt': [{'$add': ['$Marks.Statistics', '$Marks.Python', '$Marks.SQLAlchemy']},200]},'then': 'Fail','else': '$Status' }}}}])
    print(f"Documents updated: {result.modified_count}")

    print("Replacing document for employee I001 with a new structure...")

    # replace_one()
    new_data = {
        'firstname': 'Kamlesh',
        'lastname': 'Patel',
        'department': 'Data Science',
        'qualification': 'MTech',
        'age': 25,
        'Status': 'Senior'
    }
    replace_result = collection.replace_one({'_id': 'I001'}, new_data)
    print(f'Replaced data for employee I001: {replace_result.modified_count}')

def delete_records(collection):
    print("Deleting records..........................")
    # delete_one()
    del_one = collection.delete_one({'_id': 'I001'})
    print(f"Removing document of I001: {del_one.deleted_count}")

    # delete_many()
    del_many = collection.delete_many({'Status': 'Fail'})
    print(f"Removing document where status is fail: {del_many.deleted_count}")

    # Clear all records
    # collection.delete_many({})


if __name__ == '__main__':
    print("MongoDB Basics")
    client = pymongo.MongoClient("mongodb://localhost:27017")  # protocol://ipaddress:port
    mydb = client["Employees"] # DB creation
    collections = mydb["employeeinfo"] # Collection


    add_records(collections)
    # view_records(collections)
    # query_records(collections)
    # update_records(collections)
    # delete_records(collections)