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
    result = model.predict([data])
    return render_template('result.html', crop=result)

if __name__ == "__main__":
    app.run(debug=True)
