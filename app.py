from flask import Flask, render_template, request

import subprocess

app = Flask(__name__)


@app.route('/')
def index():
    return 'Website under construction !'


@app.route('/overview/')
def overview():
    par = "This is some random text that doesn't have HTML formatting"
    return render_template('basic.html', title="OxyKodit overview", body=par)


@app.route('/grid/', methods=['GET', 'POST'])
def grid():
    if request.method == 'GET':
        border_colors = "0,0,0"
        index_clicked = -1

    if request.method == 'POST':
        border_colors = request.form.get('border_colors')
        index_clicked = int(request.form.get('index_clicked', -1))

    color_list = border_colors.split(",")
    if index_clicked > -1:
        color_list[index_clicked] = str((int(color_list[index_clicked]) + 1) % 3)
        border_colors = ",".join(color_list)

    # TODO: pass arguments as dict for n images
    return render_template('tufa_grid.html', title="Tufa image grid", border_colors=border_colors, border_color_0=_get_color(color_list[0]), border_color_1=_get_color(color_list[1]), border_color_2=_get_color(color_list[2]))


def _get_color(color_index):
    if color_index == "0":
        return "blue"
    if color_index == "1":
        return "green"
    if color_index == "2":
        return "red"
    return "black"

