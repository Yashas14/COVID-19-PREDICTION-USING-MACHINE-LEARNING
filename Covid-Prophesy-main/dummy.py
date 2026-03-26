import numpy as np
from flask import Flask, render_template, request
import pickle
import os


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = list(request.form.values())
        if len(values) != 10 or any(v == '' for v in values):
            return render_template("index.html",
                prediction_text="Please answer all questions before predicting.")
        float_features = [float(x) for x in values]
        features = [np.array(float_features)]
        prediction = model.predict(features)
        if prediction[0] == 0.0:
            output = "Does NOT have COVID"
        else:
            output = "HAS COVID"
        return render_template("index.html", prediction_text="The patient {}".format(output))
    except Exception:
        return render_template("index.html",
            prediction_text="Error processing input. Please try again.")


if __name__ == "__main__":
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode)
