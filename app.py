from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello from OxyKodit  :)'


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
        border_colors="0,0,0"
        index_clicked = -1

    if request.method == 'POST':
        border_colors = request.form.get('border_colors')
        index_clicked = int(request.form.get('index_clicked', -1))

    color_list = border_colors.split(",")
    if index_clicked > -1:
        color_list[index_clicked] = str((int(color_list[index_clicked]) + 1) % 3)
        border_colors = ",".join(color_list)

    return render_template('tufa_grid.html', title="Tufa image grid", border_colors=border_colors, border_color_0=get_color(color_list[0]), border_color_1=get_color(color_list[1]), border_color_2=get_color(color_list[2]))

def get_color(color_index):
    if color_index == "0":
        return "blue"
    if color_index == "1":
        return "green"
    if color_index == "2":
        return "red"
    return "black"
