from fastapi import FastAPI, Path, HTTPException, Query
import json

app=FastAPI()

def load_data():
    with open("patient.json","r",encoding="utf-8") as f:
        data=json.load(f)
    return data

@app.get("/",tags=["Patient"])
def home():
    return {"message":"Patient Management APIS"}

@app.get("/about")
def about():
    return {
        "message1": "To view all data: /view",
        "message2": "To view data of specific patient: /view/{patient_id}",
        "message3": "To sort data by height, weight or bmi & optional order asc or desc: /view?sort_by=height_cm&order=desc"
    }


@app.get("/view")
def view():
    data = load_data()
    return data

@app.get("/view/{patient_id}")
def view(patient_id: str = Path(...,description="Patient ID",example="P001")):
    data = load_data()
    for patient in data["patients"]:
        if patient["patient_id"] == patient_id:
            return patient
    raise HTTPException(status_code=404, detail="Patient not found")

@app.get("/sort")
def sort(sort_by :str = Query(...,description="Sort on basis of height, weight or bmi "),order: str = Query('asc',description="Sort in asc or desc order")):
    data = load_data()

    valid_fields = ["height_cm","weight_kg","bmi"]

    if sort_by not in valid_fields:
        raise HTTPException(status_code=400, detail="Sort by field not valid")
    if order not in ["asc","desc"]:
        raise HTTPException(status_code=400, detail="Sort by field not valid")

    sort_order = True if order == "desc" else False

    sorted_data = sorted(data["patients"],key=lambda patient:patient[sort_by],reverse=sort_order)
    return sorted_data
