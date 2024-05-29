import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# Loading  Dataset
df1 = pd.read_csv('crop_yield_datas.csv')

# Getting our Features and Targets
X = df1.drop('Yeild (Q/acre)', axis=1)
y = df1['Yeild (Q/acre)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("loading model")
# Creating and Fitting our Model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

print("model loaded")
new_data = pd.DataFrame({'Rain Fall (mm)': [70.0],
       'Fertilizer': [60.0],
       'Nitrogen (N)': [70.0],
       'Phosphorus (P)': [20.0],
       'Potassium (K)': [18.0],
       'Temperature': [36.0]})


y_pred = model.predict(new_data)
print(y_pred[0])
print("Prediction completed")