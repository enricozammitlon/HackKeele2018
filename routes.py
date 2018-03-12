#from app import app
from flask import render_template
from flask import Flask, request,flash,redirect
import random

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/upload')
def upload():
	a=random.randint(1,1000)
	print(a)
	return render_template('index2.html')


app.run(host='127.0.0.1', port=8080, debug=True)
