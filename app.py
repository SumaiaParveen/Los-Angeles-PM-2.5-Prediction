from flask import Flask, render_template, url_for, request
import pandas as pd 
import pickle


# load the model from disk
loaded_model = pickle.load(open('UTXGB_regression_LA.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = loaded_model.predict(final_features)
    print(prediction[0])

    return render_template('home.html', prediction_text = "PM 2.5 Concentration: {} micro-gram/cubic-meter".format(prediction[0]))


if __name__ == '__main__':
    app.run(debug=True)