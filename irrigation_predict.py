import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

"""#Connecting to the drive"""

# !pip install pydrive
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials

# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)
# data_dir = '/content/drive/MyDrive/data_of_irigation'

# from google.colab import drive
# drive.mount('/content/drive')

"""#data type of the dataset"""

# dataset = pd.read_csv('/content/drive/MyDrive/data_of_irigation/cpdata.csv')

# Print the data types of the dataset
# dataset.dtypes

"""#Load and Preprocess Data"""

# Load dataset
data = pd.read_csv('cpdata.csv')

# Split features (X) and target variable (y)
X = data[['rainfall', 'temperature', 'humidity']]
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""#names of the crops dataset ontains"""

#name of all crops

# Get the unique values from the 'label' column
# unique_crops = data['label'].unique()

# Print the unique crops
# print("Unique Crops:", unique_crops)

# total unique data and column and names in csv files

# Get the number of unique values in each column
# unique_values = data.nunique()

# # Get the column names
# column_names = data.columns

# # Print the number of unique values and column names
# print("Unique Values:")
# print(unique_values)
# print("\nColumn Names:")
# print(column_names)

"""#Train the Model"""

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

"""#Evaluate the model

"""

# Make predictions on the test set
y_pred = model.predict(X_test)
print(y_pred, y_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

"""#Predict Irrigation Decision"""

new_data = pd.DataFrame({'rainfall': [210.5], 'temperature': [23.0], 'humidity': [56]})
prediction = model.predict(new_data)
print("Predicted Irrigation Decision:", prediction)
