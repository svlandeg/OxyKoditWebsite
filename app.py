from flask import Flask, render_template, request
import ml_model

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
    img_nr = 3  # TODO: 42

    if request.method == 'POST':
        border_colors = request.form.get('border_colors')
        preds = [0.5, 0.5, 0.5]
        index_clicked = int(request.form.get('index_clicked', -1))
        to_retrain = request.form.get('retrain', 'No')
    else:
        border_colors = "0,0,0"
        preds = [0.5, 0.5, 0.5]
        index_clicked = -1
        to_retrain = "No"

    color_list = border_colors.split(",")
    if index_clicked > -1:
        color_list[index_clicked] = str((int(color_list[index_clicked]) + 1) % 3)
        border_colors = ",".join(color_list)

    if to_retrain == "Yes":
        tufa_image_list = ["img" + str(i) + ".jpg" for i in range(img_nr) if color_list[i] == "1"]
        nontufa_image_list = ["img" + str(i) + ".jpg" for i in range(img_nr) if color_list[i] == "2"]
        all_image_list = ["img" + str(i) + ".jpg" for i in range(img_nr)]
        if (len(tufa_image_list)) > 0 and (len(nontufa_image_list) > 0):
            preds = ml_model.main(tufa_image_list, nontufa_image_list, all_image_list)

    # TODO: pass arguments as dict for n images
    return render_template('tufa_grid.html',
                           title="Tufa image grid",
                           border_colors=border_colors,
                           border_color_0=_get_border_color(color_list[0]),
                           border_color_1=_get_border_color(color_list[1]),
                           border_color_2=_get_border_color(color_list[2]),
                           bg_color_0=_get_bg_color(preds[0]),
                           bg_color_1=_get_bg_color(preds[1]),
                           bg_color_2=_get_bg_color(preds[2]))


def get_pred():
    return[0, 0.3, 0.9]


def _get_border_color(color_index):
    if color_index == "0":
        return "blue"
    if color_index == "1":
        return "green"
    if color_index == "2":
        return "red"
    return "black"


def _get_bg_color(color_float):
    if color_float < 0 or color_float > 1:
        return "(0, 255, 255)"
    r = color_float * 255
    return "(" + str(r) + ",0,0)"

