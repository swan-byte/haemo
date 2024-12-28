import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example data for hemophilia (Clotting factor levels and diagnosis)
data = {
    'Clotting Factor Level 1': [40, 30, 60, 25, 80, 10],
    'Clotting Factor Level 2': [35, 25, 55, 20, 70, 5],
    'Diagnosis': ['Normal', 'Hemophilia', 'Normal', 'Hemophilia', 'Normal', 'Hemophilia']
}

df = pd.DataFrame(data)

# Convert categorical diagnosis into numeric for model
df['Diagnosis'] = df['Diagnosis'].map({'Normal': 0, 'Hemophilia': 1})

# Train a simple Random Forest classifier on the data
X = df[['Clotting Factor Level 1', 'Clotting Factor Level 2']]
y = df['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app
st.title("Hemophilia Detection App")

st.write("This app predicts if a person has hemophilia based on their clotting factor levels.")

# Get user inputs for clotting factor levels
clotting_factor_1 = st.number_input("Enter Clotting Factor Level 1 (in %)", min_value=0, max_value=100)
clotting_factor_2 = st.number_input("Enter Clotting Factor Level 2 (in %)", min_value=0, max_value=100)

# Make prediction based on user input
if st.button("Predict"):
    input_data = pd.DataFrame([[clotting_factor_1, clotting_factor_2]], columns=['Clotting Factor Level 1', 'Clotting Factor Level 2'])
    prediction = model.predict(input_data)

    if prediction == 1:
        st.write("The person is likely to have Hemophilia.")
    else:
        st.write("The person is likely to be Normal.")

# Display the accuracy of the model
st.write(f"Model Accuracy on test data: {accuracy * 100:.2f}%")
