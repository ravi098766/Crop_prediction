

from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('crop_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    prediction = model.predict([data])
    return render_template('result.html', crop=prediction[0])

if __name__ == '__main__':
    print("✅ Flask app is running at http://127.0.0.1:5000/")
    app.run(debug=True)


  
