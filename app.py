from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and columns
model = pickle.load(open('model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

@app.route('/')
def home():
    # Only location names from columns (skip first 3 which are sqft, bath, bhk)
    locations = [col for col in columns[3:]]
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    sqft = float(request.form['sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])
    location = request.form['location']

    # Prepare input array
    input_array = np.zeros(len(columns))
    input_array[0] = sqft
    input_array[1] = bath
    input_array[2] = bhk
    if location in columns:
        loc_index = list(columns).index(location)
        input_array[loc_index] = 1

    predicted_price = model.predict([input_array])[0]
    return render_template('index.html', prediction_text=f"Estimated Price: ₹ {round(predicted_price, 2)} Lakhs", locations=columns[3:])

if __name__ == '__main__':
    app.run(debug=True)
