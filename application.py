from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
app=Flask(__name__)

model=pickle.load(open('12.pkl','rb'))

car=pd.read_csv("Cleaned_Car_data.csv")
@app.route('/')
def home():

    return render_template('home1.html')


@app.route('/index4.html')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique())
    fuel_type = sorted(car['fuel_type'].unique())
    return render_template('index4.html', companies=companies, car_models =car_models , years=year,fuel_type=fuel_type)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))
    print(company, car_model, year, fuel_type, kms_driven)

    data = pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    prediction = model.predict(data)
    prediction_value = np.round(prediction[0],2)
    return str(prediction_value)






if __name__=="__main__":
    app.run(debug=True)