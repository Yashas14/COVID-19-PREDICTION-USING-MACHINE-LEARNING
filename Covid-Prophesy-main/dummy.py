import numpy as np
from flask import Flask, render_template, request,jsonify
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    if(prediction==0.0):
        output="Does NOT have COVID"
    else:
        output="HAS COVID"
    return render_template("index.html", prediction_text = "The patient {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)
