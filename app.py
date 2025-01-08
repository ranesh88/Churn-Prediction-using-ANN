from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Route for the home page (index)
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the prediction logic
@app.route('/predict', methods=['POST'])
def predict():
    # User input from the form
    geography = request.form.get('geography')
    gender = request.form.get('gender')
    age = int(request.form.get('age'))
    balance = float(request.form.get('balance'))
    credit_score = float(request.form.get('credit_score'))
    estimated_salary = float(request.form.get('estimated_salary'))
    tenure = int(request.form.get('tenure'))
    num_of_products = int(request.form.get('num_of_products'))
    has_cr_card = int(request.form.get('has_cr_card'))
    is_active_member = int(request.form.get('is_active_member'))

    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict churn
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    # Return the result to the user
    churn_message = 'The customer is likely to churn.' if prediction_proba > 0.5 else 'The customer is not likely to churn.'
    return render_template('index.html', churn_prob=f'{prediction_proba:.2f}', churn_message=churn_message)

if __name__ == '__main__':
    app.run(debug=True)