import numpy as np
from flask import Flask, render_template, request
import pickle as pkl
import jsonify
import requests
import pickle
import sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

# model =pkl.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predicts', methods=['POST']) # post is not a function. it is a html file render method to use input
def predicts():
    '''
    for rendering on html gui
    '''

    # input
    if request.method == 'POST':
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        sqft_lot = float(request.form['sqft_lot'])
        view = float(request.form['view'])
        waterfront = float(request.form['waterfront'])
        grade = float(request.form['grade'])

        condition = float(request.form['condition'])
        sqft_living15 = float(request.form['sqft_living15'])
        sqft_above = float(request.form['sqft_above'])
        sqft_basement = float(request.form['sqft_basement'])
        yr_built = float(request.form['yr_built'])

        yr_renovated = float(request.form['yr_renovated'])
        zipcode = float(request.form['zipcode'])

        lat = float(request.form['lat'])
        long = float(request.form['long'])
        floors = float(request.form['floors'])

        Date = request.form['Date']
        day = int(pd.to_datetime(Date, format="%Y-%m-%d").day)
        month = int(pd.to_datetime(Date, format="%Y-%m-%d").month)
        year = int(pd.to_datetime(Date, format="%Y-%m-%d").year)

        print(day,floors)



    #     prediction = model.predict([[Present_Price, Kms_Driven2, Owner, Year, Fuel_Type_Diesel, Fuel_Type_Petrol,
    #                                  Seller_Type_Individual, Transmission_Mannual]])
    #     output = round(prediction[0], 2)
    #     if output < 0:
        output = 100

        return render_template('index.html', prediction_text='House Price should be $ {}'.format(output))

    #     else:
    #         return render_template('index.html', prediction_text="You Can Sell The Car at {}".format(output))
    # else:
    #     return render_template('index.html')

    # input_features = np.array(in_features)


    # predicts
    # output = mode.predict(input_features)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)

# https://www.youtube.com/watch?v=mrExsjcvF4o
