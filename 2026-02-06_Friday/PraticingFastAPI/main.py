from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

bmi_store={}

#Data Structure Class : defines the expected input structure
class BMIInput(BaseModel):
    name:str
    height:float
    weight:int

def calculate_bmi(height,weight):
    return weight/(height*height) #height-meter,weight-kg

@app.get("/")
def bmi():
    return bmi_store

@app.post("/bmi")
def create_bmi(user:BMIInput):
    #Here, user is just the name of the function parameter.
    #It’s the variable that will hold the incoming request data.
    # :BMIInput part is a type annotation.It tells FastAPI:“This parameter must be an object of type BMIInput.”
    bmi_value=calculate_bmi(user.height,user.weight)
    bmi_store[user.name]={
        "weight":user.weight,
        "height":user.height,
        "bmi":bmi_value
    }
    return {
        "message":"BMI calculated successfully",
        "data":bmi_store[user.name]
    }

# example a client sends a POST request with JSON: :
# request body:{
#     "height":int,
#     "weight":int,
#     "bmi":int
# }
# FastAPI automatically:Reads the request body & Validates it against the BMIInput class.
# Creates a BMIInput object and passes it into your function as user
# 1. FastAPI doesn’t care about the order of keys in JSON.JSON is a key–value structure, not a sequence.
# 2. As long as the keys match the fields in your BMIInput class (name, height, weight), FastAPI will parse it correctly.
# 3. Missing fields → FastAPI returns a 422 error.
# 4. Wrong types → FastAPI returns a 422 error.
