#flask application
from flask import Flask, request, render_template
import numpy as np 
import pandas as pd
import os
import sys 

from sklearn.preprocessing import LabelEncoder

application = Flask(__name__)
app = application

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from src.pipeline.predict_pipeline import PredictPipeline
except Exception as e:
    print(f"Warning: Could not import PredictPipeline: {e}")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            data = request.form.to_dict()
            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict(data)
            return render_template('index.html', results=prediction)
        except Exception as e:
            return render_template('index.html', error=str(e))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
