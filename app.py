from flask import Flask, render_template, request

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


@app.route('/tufa/', defaults={'count': 5})
@app.route('/tufa/<int:count>')
def tufa(count):
    return render_template('tufa_img.html', title="Tufa image", border_color="green", count=count)


@app.route('/grid/', methods = ['GET', 'POST'])
def grid():
    if request.method == 'GET':
        return render_template('tufa_grid.html', title="Tufa image grid", border_color="green")
    if request.method == 'POST':
        color = request.form.get('color')
        return render_template('tufa_grid.html', title="Tufa image grid", border_color=color)


