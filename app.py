from flask import Flask, render_template
import tensorflow as tf

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello world!'


@app.route('/ml')
def hello():
    return 'Hello, Machine learning'


@app.route('/add/<int:id1>/<int:id2>')
def add(id1, id2):
    value = id1+id2
    return 'Calculated:' + str(value)


@app.route('/overview/')
def overview():
    par = "This is some random text that doesn't have HTML formatting"
    return render_template('overview.html', title="OxyKodit overview", body=par)


@app.route('/tufa/')
def tufa():
    return render_template('tufa_img.html', title="Tufa image", border_color="green")
