# import neccessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load Titanic dataset
data = sns.load_dataset("titanic")

# check for missing values
data.isnull().sum()

# Preprocess Data
data.drop(columns=['deck', 'alive', 'class', 'who', 'embark_town'], inplace=True)
data['age'].fillna(data['age'].median(), inplace=True)
data['embarked'].fillna(data['embarked'].mode()[0], inplace=True)

# Convert categorical columns to numeric
data['sex'] = data['sex'].map({'female': 0, 'male': 1})
data['embarked'] = data['embarked'].map({'C': 0, 'Q': 1, 'S': 2})
data['alone'] = data['alone'].astype(int)
data['adult_male'] = data['adult_male'].astype(int)

# define dependent and independent variable
X = data.drop(columns='survived') 
y = data['survived']               

# train Logistic Regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival")

# Takin User Input
pclass = st.selectbox("Passenger Class (1 = First, 2 = Second, 3 = Third)", [1, 2, 3])
sex = st.radio("Gender", ["Male", "Female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.slider("Parents/Children Aboard", 0, 6, 0)
fare = st.number_input("Fare Paid", min_value=0.0, value=10.0, step=1.0)
embarked = st.selectbox("Embarked from", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])
adult_male = st.radio("Is an Adult Male?", ["Yes", "No"])
alone = st.radio("Is Traveling Alone?", ["Yes", "No"])

# Convert inputs to numerical values
sex = 1 if sex == "Male" else 0
embarked = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}[embarked]
adult_male = 1 if adult_male == "Yes" else 0
alone = 1 if alone == "Yes" else 0

# Predict survival
if st.button("Predict Survival"):
    user_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked, adult_male, alone]])
    prediction = model.predict(user_data)[0]
    result = "🎉 Survived!" if prediction == 1 else "💀 Did Not Survive"
    st.subheader(result)

# display model accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f"Model Accuracy: **{accuracy:.2f}**")

# Visualize Feature Importance
feature_importance = pd.DataFrame(model.coef_[0], index=X.columns, columns=['Coefficient']).sort_values(by='Coefficient', ascending=False)
st.bar_chart(feature_importance)
