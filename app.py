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

model =pkl.load(open('model.pkl','rb'))

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

        dummy=1
        X=[dummy,
         bedrooms,
         bathrooms,
         sqft_lot,
         floors,
         waterfront,
         view,
         condition,
         grade,
         sqft_above,
         sqft_basement,
         yr_built,
         yr_renovated,
         zipcode,
         lat,
         long,
         sqft_living15,
         month,
         day,
         year]

        X=np.array(X)
        prediction = X.dot(np.array(model['weights']))
        output = round(prediction, 2)
        return render_template('index.html', prediction_text='House Price should be $ {}'.format(output))

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
# https://www.youtube.com/watch?v=mrExsjcvF4o
