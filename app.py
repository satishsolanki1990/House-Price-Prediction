import numpy as np
from flask import Flask, render_template, request
import pickle as pkl

app = Flask(__name__)

# model =pkl.load(open('model.pkl','rb'))

@app.route('./')
def home():
    return render_template('./templates/index.html')

@app.route('./predict', method=['POST']) # post is not a function. it is a html file render method to use input
def predicts():
    '''
    for rendering on html gui
    '''

    # input
    in_features=[int(x) for x in request.form.values()]
    input_features = np.array(in_features)

    # predicts
    output = mode.predict(input_features)

    return render_template('index.html',prediction_text='House Price should be $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

# https://www.youtube.com/watch?v=mrExsjcvF4o
