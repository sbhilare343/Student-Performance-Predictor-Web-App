# Student-Performance-Predictor-Web-App

### 1. **Importing Libraries**
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
```
- **NumPy & Pandas**: Useful for data manipulation and numerical operations.
- **Scikit-learn**: Contains tools for model selection (`train_test_split`) and creating a machine learning model (`RandomForestRegressor`).
- **Matplotlib & Seaborn**: Visualization libraries, though not explicitly used in this code snippet.

### 2. **Loading and Inspecting the Dataset**
```python
df = pd.read_csv('/content/student_data.csv')
df.head()

df.describe()
```
- Loads the dataset from a CSV file.
- `df.head()` shows the first few rows of the data.
- `df.describe()` provides statistical insights (mean, min, max, etc.) for numerical columns in the dataset.

### 3. **One-Hot Encoding of Categorical Variables**
```python
df = pd.get_dummies(df, columns=['SES', 'Parent Involvement', 'Tutoring', 'Health Conditions'])
df.head()
```
- **One-hot encoding**: Converts categorical variables into a form that machine learning models can understand. For example, if 'SES' has three categories (Low, Middle, High), it will create three new binary columns (`SES_Low`, `SES_Middle`, `SES_High`), and the value 1 indicates the presence of that category.
- Applies one-hot encoding to `SES`, `Parent Involvement`, `Tutoring`, and `Health Conditions`.

### 4. **Defining Features and Target Variable**
```python
X = df.drop('Final Grade', axis=1)
y = df['Final Grade']
```
- **X**: Contains all feature columns except the target column `Final Grade`.
- **y**: Stores the `Final Grade` column, which is what the model will predict.

### 5. **Splitting Data into Training and Testing Sets**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- **train_test_split**: Splits the data into training (80%) and testing sets (20%).
- `random_state=42`: Ensures the split is reproducible by setting a seed.

### 6. **Model Creation and Training**
```python
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```
- **RandomForestRegressor**: A machine learning model that uses an ensemble of decision trees to predict a continuous value (regression).
- `n_estimators=100`: Specifies the number of decision trees in the forest.
- `model.fit(X_train, y_train)`: Trains the model using the training data.

### 7. **Making Predictions**
```python
y_pred = model.predict(X_test)
```
- **model.predict(X_test)**: Uses the trained model to predict the target values (`Final Grade`) for the test dataset.

### 8. **Evaluating Model Performance**
```python
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)
```
- **mean_squared_error**: Measures the average squared difference between actual and predicted values.
- **rmse**: Root Mean Squared Error, which is the square root of MSE.
- **r2_score**: R-squared value, which measures how well the model's predictions fit the data (closer to 1 indicates a better fit).

### 9. **Saving the Model**
```python
import pickle
pickle.dump(model, open('model.pkl', 'wb'))
```
- **pickle**: Used to save the trained model to a file (`model.pkl`).
- `pickle.dump`: Serializes the model object to the file `model.pkl` in write-binary mode (`'wb'`).

### 10. **Gradio Setup for Web Interface**
```python
!pip install gradio

import gradio as gr
```
- **Gradio**: A Python library for creating interactive web interfaces for machine learning models. The `!pip install gradio` command installs it.

### 11. **Function for Predicting Final Grade**
```python
def predict_final_grade(attendance, study_hours, homework_completion, test_scores, ses, extracurricular_activities, parent_involvement, tutoring, sleep_hours, health_conditions):
    try:
        input_data = pd.DataFrame({
            'Attendance': [attendance],
            'Study Hours': [study_hours],
            'Homework Completion': [homework_completion],
            'Test Scores': [test_scores],
            'SES': [ses],
            'Extracurricular Activities': [extracurricular_activities],
            'Parent Involvement': [parent_involvement],
            'Tutoring': [tutoring],
            'Sleep Hours': [sleep_hours],
            'Health Conditions': [health_conditions]
        })

        input_data = pd.get_dummies(input_data, columns=['SES', 'Parent Involvement', 'Tutoring', 'Health Conditions'])
        input_data = input_data.reindex(columns=X.columns, fill_value=0)

        predicted_grade = model.predict(input_data)

        if predicted_grade[0] >= 90:
            emoji = "🎉"
        elif predicted_grade[0] >= 80:
            emoji = "👍"
        elif predicted_grade[0] >= 70:
            emoji = "🙂"
        else:
            emoji = "😕"

        return f"Predicted Final Grade: {predicted_grade[0]:.2f} {emoji}"
    except Exception as e:
        return f"Error: {e}"
```
- **predict_final_grade**: This function takes several input parameters (such as attendance, study hours, test scores, etc.) and predicts the student's final grade using the trained model.
- It also performs **one-hot encoding** on categorical features and ensures the input data matches the original training data.
- The function returns the predicted grade along with an emoji indicating performance based on the predicted value.

### 12. **Loading the Trained Model**
```python
try:
    with open('/content/model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: Model file not found.")
except Exception as e:
    print(f"Error loading model: {e}")
```
- Attempts to load the saved model from `model.pkl` using `pickle`. If the file is not found, it prints an error message.

### 13. **Creating the Gradio Interface**
```python
iface = gr.Interface(
    fn=predict_final_grade,
    inputs=[
        gr.Number(label="Attendance (70-100)"),
        gr.Number(label="Study Hours (10-30)"),
        gr.Number(label="Homework Completion (75-100)"),
        gr.Number(label="Test Scores (50-100)"),
        gr.Dropdown(label="SES", choices=["Low", "Middle", "High"]),
        gr.Number(label="Extracurricular Activities (0-4)"),
        gr.Dropdown(label="Parent Involvement", choices=["Low", "Medium", "High"]),
        gr.Dropdown(label="Tutoring", choices=["Yes", "No"]),
        gr.Number(label="Sleep Hours (6-9)"),
        gr.Dropdown(label="Health Conditions", choices=["None", "Mild", "Severe"])
    ],
    outputs="text",
    title="Student Academic Performance Predictor"
)
```
- **gr.Interface**: Creates a web interface using Gradio.
- `inputs`: Specifies the user input types (e.g., `gr.Number` for numeric inputs, `gr.Dropdown` for categorical choices).
- `fn`: The function that is called when the user submits input (`predict_final_grade`).
- `outputs`: Defines the output as a text field, which will display the predicted final grade.
- `title`: The title of the web interface.

### 14. **Launching the Interface**
```python
iface.launch(debug=True)
```
- **launch**: Starts the Gradio web interface. `debug=True` enables debug mode, which prints helpful information for troubleshooting.
