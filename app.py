from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and preprocessor (which includes both encoder and scaler)
model = joblib.load('final_random_forest_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Load cleaned car data for dropdowns
car = pd.read_csv('Cleaned_Car_data.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    # Load dropdown options for the form
    manufacturers = sorted(car['manufacturer'].unique())
    fuel_types = sorted(car['fuel'].unique())
    conditions = ['excellent', 'good', 'fair', 'poor']
    transmissions = ['automatic', 'manual']
    drive_types = ['fwd', 'rwd', 'awd']

    return render_template('index.html', 
                           manufacturers=manufacturers, 
                           fuel_types=fuel_types, 
                           conditions=conditions, 
                           transmissions=transmissions, 
                           drive_types=drive_types)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data (user inputs)
        manufacturer = request.form.get('manufacturer')
        condition = request.form.get('condition')
        cylinders = request.form.get('cylinders')
        fuel = request.form.get('fuel')
        type_ = request.form.get('type')
        paint_color = request.form.get('paint_color')
        size = request.form.get('size')
        title_status = request.form.get('title_status')
        transmission = request.form.get('transmission')
        drive = request.form.get('drive')
        car_age = request.form.get('car_age')
        odometer = request.form.get('odometer')

        # Create a DataFrame for the input data
        input_data = pd.DataFrame([[manufacturer, condition, cylinders, fuel, type_, paint_color, size, title_status,
                                    transmission, drive, car_age, odometer]], 
                                  columns=['manufacturer', 'condition', 'cylinders', 'fuel', 'type', 
                                           'paint_color', 'size', 'title_status', 'transmission', 
                                           'drive', 'car_age', 'odometer'])

        # Transform the input data using the preprocessor
        input_transformed = preprocessor.transform(input_data)

        # Make the prediction
        prediction = model.predict(input_transformed)

        # Convert the predicted log price back to the original price
        predicted_price = np.round(np.exp(prediction[0]), 2)  # Convert from log scale to normal scale

        # Load dropdown options for the form
        manufacturers = sorted(car['manufacturer'].unique())
        fuel_types = sorted(car['fuel'].unique())
        conditions = ['excellent', 'good', 'fair', 'poor']
        transmissions = ['automatic', 'manual']
        drive_types = ['fwd', 'rwd', 'awd']

        # Render the template with the prediction and user selections
        return render_template('index.html', 
                               manufacturers=manufacturers, 
                               fuel_types=fuel_types, 
                               conditions=conditions, 
                               transmissions=transmissions, 
                               drive_types=drive_types,
                               prediction_text=f'Predicted Car Price: ${predicted_price}',
                               selected_manufacturer=manufacturer,
                               selected_condition=condition,
                               selected_cylinders=cylinders,
                               selected_fuel=fuel,
                               selected_type=type_,
                               selected_paint_color=paint_color,
                               selected_size=size,
                               selected_title_status=title_status,
                               selected_transmission=transmission,
                               selected_drive=drive,
                               selected_car_age=car_age,
                               selected_odometer=odometer)
    
    except Exception as e:
        # Render the template with an error message if something goes wrong
        return render_template('index.html', prediction_text=f'Error: {str(e)}')




if __name__ == '__main__':
    app.run(debug=True)

