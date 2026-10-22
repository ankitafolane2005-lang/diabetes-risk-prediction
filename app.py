import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Page settings
st.set_page_config(page_title="SVM Prediction App")

st.title("SVM Machine Learning Project")
st.write("Iris Flower Prediction using SVM")

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"{accuracy * 100:.2f}%")

# User input
st.subheader("Enter Flower Measurements")

sepal_length = st.number_input(
    "Sepal Length (cm)", 0.0, 10.0, 5.1
)
sepal_width = st.number_input(
    "Sepal Width (cm)", 0.0, 10.0, 3.5
)
petal_length = st.number_input(
    "Petal Length (cm)", 0.0, 10.0, 1.4
)
petal_width = st.number_input(
    "Petal Width (cm)", 0.0, 10.0, 0.2
)

if st.button("Predict"):
    prediction = model.predict([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    flower_name = iris.target_names[prediction[0]]
    st.success(f"Predicted Flower: {flower_name}")
