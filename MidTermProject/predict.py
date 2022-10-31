#!/usr/bin/env python
# coding: utf-8


import pickle
from flask import Flask
from flask import request
from flask import jsonify
import warnings
warnings.filterwarnings("ignore")

# paramiters 
d = 25
n = 20
model_file = f"model_{d}_{n}"


with open(model_file, 'rb') as f_in:
    dv, rf = pickle.load(f_in)  

app = Flask('fraud')
@app.route('/predict', methods=['POST'] )
def predict():
    card = request.get_json()
    X = dv.fit_transform([card])
    fraude = rf.predict(X)[0] == 0
    result = {
        "prediction": float(rf.predict(X)[0]),
        "card type": bool(fraude)
    }
    
    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

