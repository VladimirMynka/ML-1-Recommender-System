from flask import Flask
from model import My_Rec_Model

app = Flask(__name__)

@app.route('/api/predict')
def predict():
    pass


@app.route('/api/log')
def log():
    pass


@app.route('/api/info')
def info():
    pass


@app.route('/api/reload')
def predict():
    pass


@app.route('/api/similar')
def similar():
    pass