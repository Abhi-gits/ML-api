from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.model_selection import train_test_split

# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel
from sklearn.metrics import accuracy_score

# creating FastAPI instance
app = FastAPI()

# Creating a simple endpoint for get request
@app.get("/")
def first_example():

    
    return {"My ML API ": "FastAPI"}


 
# Creating class to define the request body
# and the type hints of each attribute

class request_body(BaseModel):
        rainfall : float
        fertilizer : float
        nitrogen : float
        potassium : float
        phosphorus : float
        temperature : float

def predict_yield(test_data):

    # Loading  Dataset
    df1 = pd.read_csv('crop_yield_datas.csv')

    # Getting our Features and Targets
    X = df1.drop('Yeild (Q/acre)', axis=1)
    y = df1['Yeild (Q/acre)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Creating and Fitting our Model
    # model = DecisionTreeRegressor()
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(test_data)[0]
    # # convert y_pred into array
    # y_pred = y_pred.tolist()
    # accuracy = model.score(test_data, y_pred)

    return y_pred


def irrigation_model(test_data):
    # Load dataset
    data = pd.read_csv('cpdata.csv')

    # Split features (X) and target variable (y)
    X = data[['rainfall', 'temperature', 'humidity']]
    y = data['label']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(test_data)[0]

    # accuracy = accuracy_score(y_test, y_pred)

    return y_pred

 
# Creating an Endpoint to receive the data
# to make prediction on.
@app.post('/predict-yield')
def predict(data : request_body):
    # Making the data in a form suitable for prediction
    test_data = [[
        data.rainfall,
        data.fertilizer,
        data.nitrogen,
        data.potassium,
        data.phosphorus,
        data.temperature
    ]]
     
    # Predicting the Class
    try: 
        y_pred = predict_yield(test_data)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid User Input " + str(e))
    
    # Return the Result
    return {"Predicted yield (Q/acre)": y_pred}

class irrigation_data(BaseModel):
    rainfall : float
    humidity : float
    temperature : float


@app.post('/predict-irrigation')
def predict(data : irrigation_data):
    # Making the data in a form suitable for prediction
    test_data = [[
        data.rainfall,
        data.humidity,
        data.temperature
    ]]


    try:
        y_pred = irrigation_model(test_data)

    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid User Input " + str(e))
    
    # Return the Result
    return {"Plant to irrigate": y_pred}








# {
#   "rainfall": 50.0,
#   "fertilizer": 40.0,
#   "nitrogen": 30.0,
#   "potassium": 35.0,
#   "phosphorus": 26.0,
#   "temperature": 28.0
# }