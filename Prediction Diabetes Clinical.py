# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 23:43:05 2024

@author: Mary
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('D:\Learning\ML\Datasets\diabetes_dataset.csv')
print(data)
print("Head:",data.head(10))
#df.replace(0,nan,inplace=True)
missing_values = data.isnull().sum()
print(missing_values)
# Display data types of all columns
print("Data Types of All Columns:")
print(data.dtypes)

#Identify numerical columns
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
print("\nNumerical Columns:")
print(numerical_cols)
object_cols=data.select_dtypes(include=['object']).columns
print(object_cols)
unique_locations = data['location'].nunique()
print(unique_locations)

# Count unique values in the 'location' column to determine the encoding process, in this case it is better to use label encoder.
unique_locations = data['location'].nunique()

# Print the number of unique locations
print(f"Number of unique locations: {unique_locations}")


duplicated=data[data[['year','gender','age','location','race:AfricanAmerican','race:Asian','race:Caucasian','race:Hispanic','race:Other','hypertension','heart_disease','smoking_history','bmi','hbA1c_level','blood_glucose_level']].duplicated(keep=False)]
print("duplicated1:",len(duplicated))

duplicated=data[data.duplicated(['year','gender','age','location','race:AfricanAmerican','race:Asian','race:Caucasian','race:Hispanic','race:Other','hypertension','heart_disease','smoking_history','bmi','hbA1c_level','blood_glucose_level'],keep='first')]
duplicated=data.duplicated().sum()
print("duplicated:",duplicated)

# for index,row in duplicated.iterrows():
 
#        print(index,row['year'],row['gender'],row['age'],row['location'],
#              row['race:AfricanAmerican'],row['race:Asian'],row['race:Caucasian']
#              ,row['race:Hispanic'],row['race:Other'],row['hypertension'],
#              row['heart_disease'],row['smoking_history'],row['bmi'],row['hbA1c_level'],row['blood_glucose_level'])

missing_values = data.isnull().sum()

# Print the missing values count for each column
print(missing_values)


# Display data types of all columns
print("Data Types of All Columns:")
print(data.dtypes)

# Identify numerical columns
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
print("\nNumerical Columns:")
print(numerical_cols)

# Identify object type columns
object_cols = data.select_dtypes(include=['object']).columns
print("\nObject (Categorical) Columns:")
print(object_cols)
# Count unique values in the 'location' column to determine the encoding process, in this case it is better to use label encoder.
unique_locations = data['location'].nunique()

# Print the number of unique locations
print(f"Number of unique locations: {unique_locations}")
# Apply preprocessing


def preprocess(data):
    # Apply one-hot encoding to 'gender' and 'smoking_history'
    data = pd.get_dummies(data, columns=['gender', 'smoking_history'])
    
    # Apply label encoding to 'location'
    label_encoder = LabelEncoder()
    data['location'] = label_encoder.fit_transform(data['location'])
    
    return data
data_encoded = preprocess(data)

# Display all columns of the processed data
columns_list = data_encoded.columns
print("Columns after preprocessing:")
print(columns_list)

# Display the first few rows of the processed data
print("Head Tuples:")
print(data_encoded.head(10))


# Display all columns of the processed data
columns_list = data_encoded.columns
print("Columns after preprocessing:")
print(columns_list)
print("corelation:")
corr=data_encoded.corr()['diabetes']
print(corr)

# Display the first few rows of the processed data
data_encoded.head()
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import xgboost as xgb
# Features selection
features = ['age', 'location', 'race:AfricanAmerican', 'race:Asian',
            'race:Caucasian', 'race:Hispanic', 'race:Other', 
            'bmi', 'gender_Female', 'gender_Male', 'gender_Other',
            'smoking_history_No Info', 'smoking_history_current',
            'smoking_history_ever', 'smoking_history_former',
            'smoking_history_never', 'smoking_history_not current']

X = data_encoded[features]
y = data_encoded['diabetes']
# Convert the dataset into a DMatrix object (XGBoost's optimized data structure)
dtrain = xgb.DMatrix(X, label=y)

# Define parameters for the XGBoost model, adjusting to combat overfitting
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 4,  # Reduced depth to prevent complex models
    'eta': 0.1,  # Lower learning rate to make the learning process more conservative
    'lambda': 1,  # L2 regularization term on weights (increased to reduce overfitting)
    'alpha': 0.3,  # L1 regularization term on weights
    'seed': 42
}

# Perform cross-validation
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=500,  # Number of boosting rounds (trees to build)
    nfold=10,  # Number of folds in CV
    metrics={'error'},  # Primary metric to evaluate during CV
    early_stopping_rounds=50,  # Stop if 50 rounds without improvement
    as_pandas=True,
    seed=42
)

# Compute mean accuracy from the CV results
mean_train_accuracy = (1 - cv_results['train-error-mean']).iloc[-1]
mean_test_accuracy = (1 - cv_results['test-error-mean']).iloc[-1]

print(f"Mean Train Accuracy: {mean_train_accuracy:.3f}")
print(f"Mean Test Accuracy: {mean_test_accuracy:.3f}")

# Plotting the training and testing error metrics
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(1 - cv_results['train-error-mean'], label='Train Accuracy')
ax.plot(1 - cv_results['test-error-mean'], label='Test Accuracy')
ax.set_xlabel("Number of Boosting Rounds")
ax.set_ylabel("Accuracy")
ax.set_title("Training vs Validation Accuracy Across Boosting Rounds")
ax.legend()
plt.show()

# Train the model to get the feature importance
model = xgb.train(params, dtrain, num_boost_round=cv_results.shape[0])

# Get feature importance scores and plot them
feature_importance = model.get_score(importance_type='weight')
sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh([x[0] for x in sorted_importance], [x[1] for x in sorted_importance])
ax.set_xlabel("Feature Importance Score")
ax.set_ylabel("Features")
ax.set_title("Feature Importance")
plt.show()

# Train the model to get the feature importance
model = xgb.train(params, dtrain, num_boost_round=cv_results.shape[0])

# Get feature importance scores and plot them
feature_importance = model.get_score(importance_type='weight')
sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh([x[0] for x in sorted_importance], [x[1] for x in sorted_importance])
ax.set_xlabel("Feature Importance Score")
ax.set_ylabel("Features")
ax.set_title("Feature Importance")
plt.show()
# Example Function to take user input and make a prediction if patient has diabetes or not.
def predict_diabetes(age, location, race_africanamerican, race_asian, race_caucasian, race_hispanic, race_other, bmi, 
                     gender_female, gender_male, gender_other, smoking_no_info, smoking_current, smoking_ever, 
                     smoking_former, smoking_never, smoking_not_current):
 # Create a DataFrame for the input values
   input_data = pd.DataFrame({
        'age': [age],
        'location': [location],
        'race:AfricanAmerican': [race_africanamerican],
        'race:Asian': [race_asian],
        'race:Caucasian': [race_caucasian],
        'race:Hispanic': [race_hispanic],
        'race:Other': [race_other],
        'bmi': [bmi],
        'gender_Female': [gender_female],
        'gender_Male': [gender_male],
        'gender_Other': [gender_other],
        'smoking_history_No Info': [smoking_no_info],
        'smoking_history_current': [smoking_current],
        'smoking_history_ever': [smoking_ever],
        'smoking_history_former': [smoking_former],
    'smoking_history_never': [smoking_never],
    'smoking_history_not current': [smoking_not_current]
    })
    # Convert the input data into a DMatrix
   dtest_input = xgb.DMatrix(input_data)

# Make prediction
   predicted_value = model.predict(dtest_input)

# Convert probability to binary class
   predicted_class = (predicted_value > 0.5).astype(int)

   return predicted_class[0], predicted_value[0]

# Example od data input:
age = 45
location = 1  # Replace with actual value for location, the location is transformed by LabelEncoder() in preprocess function.
race_africanamerican = 0
race_asian = 0
race_caucasian = 1
race_hispanic = 0
race_other = 0
bmi = 28.5
gender_female = 0
gender_male = 1
gender_other = 0
smoking_no_info = 0

