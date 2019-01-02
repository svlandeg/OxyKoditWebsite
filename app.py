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
    img_nr = 42

    if request.method == 'POST':
        border_indices = request.form.get('border_indices')
        preds = [0.5] * img_nr
        index_clicked = int(request.form.get('index_clicked', -1))
        to_retrain = request.form.get('retrain', 'No')
    else:
        color_list = ["0"] * img_nr
        border_indices = ",".join(color_list)
        preds = [0.5] * img_nr
        index_clicked = -1
        to_retrain = "No"

    color_list = border_indices.split(",")
    if index_clicked > -1:
        color_list[index_clicked] = str((int(color_list[index_clicked]) + 1) % 3)
        border_indices = ",".join(color_list)

    if to_retrain == "Yes":
        tufa_image_list = ["img" + str(i) + ".jpg" for i in range(img_nr) if color_list[i] == "1"]
        nontufa_image_list = ["img" + str(i) + ".jpg" for i in range(img_nr) if color_list[i] == "2"]
        all_image_list = ["img" + str(i) + ".jpg" for i in range(img_nr)]
        if (len(tufa_image_list)) > 0 and (len(nontufa_image_list) > 0):
            preds = ml_model.main(tufa_image_list, nontufa_image_list, all_image_list)

    return render_template('tufa_grid.html',
                           title="Tufa image grid",
                           border_indices=border_indices,
                           image_indices=range(img_nr),
                           border_colors=[_get_border_color(color_list[i]) for i in range(img_nr)],
                           bg_colors=[_get_bg_color(preds[i]) for i in range(img_nr)])


def _get_border_color(color_index):
    if color_index == "0":
        return "blue"
    if color_index == "1":
        return "green"
    if color_index == "2":
        return "black"
    return "red"


def _get_bg_color(tufa_pred_float):
    transparant = 0.5

    tufa_pred_float = max(tufa_pred_float, 0)
    tufa_pred_float = min(tufa_pred_float, 1)

    # more red if its not a tufa, more green if it's a tufa
    r = (1-tufa_pred_float)*255
    g = tufa_pred_float * 255

    return "(" + str(r) + "," + str(g) + ",127.5, " + str(transparant) + ")"

