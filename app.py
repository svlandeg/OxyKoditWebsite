from flask import Flask, render_template, request
import ml_model
import os

# ==============================================================================
# By Sofie Van Landeghem @ OxyKodit
#
# Flask framework to run the Deep Learning image recognition demo online
# (http://www.oxykodit.com/blog/tufa)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================


app = Flask(__name__)


def get_ml_model():
    my_dir = os.path.dirname(__file__)

    # model cached from 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2'
    # return 224, os.path.join(my_dir, 'ml_model/mobilenet_v2_100_224/')

    # model cached from 'https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/2'
    return 128, os.path.join(my_dir, 'ml_model/mobilenet_v2_035_128/')


@app.route('/')
def index():
    return render_template('index_redirect.html')


@app.route('/grid/', methods=['GET', 'POST'])
def grid():
    img_nr = 25
    grid_width = 5

    # retrieve the POST parameters or set default ones
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

    # if an image was clicked, change its border color and index
    color_list = border_indices.split(",")
    if index_clicked > -1:
        color_list[index_clicked] = str((int(color_list[index_clicked]) + 1) % 3)
        border_indices = ",".join(color_list)

    # inspect the current selection of tufas and non-tufas
    tufa_image_list = ["img" + str(i) + ".jpg" for i in range(img_nr) if color_list[i] == "1"]
    nontufa_image_list = ["img" + str(i) + ".jpg" for i in range(img_nr) if color_list[i] == "2"]
    all_image_list = ["img" + str(i) + ".jpg" for i in range(img_nr)]
    if (len(tufa_image_list)) > 0 and (len(nontufa_image_list) > 0):
        allow_training = True
        train_disabled = ""
    else:
        allow_training = False
        train_disabled = "disabled"

    # train the model and make preditions
    if allow_training and to_retrain == "Yes":
        img_size, model = get_ml_model()
        preds = ml_model.main(tufa_image_list, nontufa_image_list, all_image_list, model, img_size)

    # return the HTML page
    return render_template('tufa_grid.html',
                           grid_width=grid_width,
                           title="Tufa image grid",
                           border_indices=border_indices,
                           image_indices=range(img_nr),
                           border_colors=[_get_border_color(color_list[i]) for i in range(img_nr)],
                           bg_colors=[_get_bg_color(preds[i]) for i in range(img_nr)],
                           train_disabled=train_disabled)


def _get_border_color(color_index):
    if color_index == "0":
        return "white"
    if color_index == "1":
        return "green"
    if color_index == "2":
        return "red"
    return "blue"


def _get_bg_color(tufa_pred_float):
    transparant = 0.5

    tufa_pred_float = max(tufa_pred_float, 0)
    tufa_pred_float = min(tufa_pred_float, 1)

    # more red if its not a tufa, more green if it's a tufa
    r = (1-tufa_pred_float)*255
    g = tufa_pred_float * 255

    return "(" + str(int(r)) + "," + str(int(g)) + ",127, " + str(transparant) + ")"

