import pandas as pd

country=["US","Australia","Japan","India","Russia"]
drive_right=[False,True,False,True,True]
cpu=[800,727,590,200,260]

my_dict={'country':country,'drive_right':drive_right,'cpu':cpu}

cars=pd.DataFrame(my_dict)
# print(cars)
# print("Describe:",cars.describe())
# print("Shape:",cars.shape)
# print("Info:",cars.info())

#============Column Selection===============
# list_countries=cars[["country","cpu"]]
# print(list_countries)

#============Row Selection (iloc,loc,ix)===============
# iloc - positional indexing
# loc  - label based indexing

print(cars.iloc[0])
print(cars.loc[0])






