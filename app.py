from flask import Flask, render_template, request

import subprocess

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
    return render_template('basic.html', title="OxyKodit overview", body=par)


@app.route('/tufa/', defaults={'count': 5})
@app.route('/tufa/<int:count>')
def tufa(count):
    return render_template('tufa_img.html', title="Tufa image", border_color="green", count=count)


@app.route('/grid/', methods=['GET', 'POST'])
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


tmp_dir = "tmp-train/"
graph = "output_graph.pb"
labels = "output_labels.txt"
size = 224


@app.route('/train/')
def train():
    par = "Currently training the model"
    subprocess.call(['python', 'retrain_single.py', '--image_dir=data/tufa_limited_training',
                     '--testing_percentage=0', '--validation_percentage=0',
                     '--validation_batch_size=0', '--how_many_training_steps=25',
                     '--tfhub_module=https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2',
                     '--output_graph=' + tmp_dir + graph,
                     '--intermediate_output_graphs_dir=' + tmp_dir + 'intermediate_graph/',
                     '--output_labels=' + tmp_dir + labels,
                     '--summaries_dir=' + tmp_dir + 'retrain_tufa',
                     '--bottleneck_dir=' + tmp_dir + 'bottleneck'])

    return render_template('basic.html', title="Tufa training !", body=par)


@app.route('/predict/')
def predict():
    par = "Currently making predictions with the trained model"
    subprocess.call(['python', 'label_image_dir.py', '--graph=' + tmp_dir + graph, '--labels=' + tmp_dir + labels,
                     '--input_layer=Placeholder', '--output_layer=final_result',
                     '--input_height=' + str(size), '--input_width=' + str(size),
                     '--dir=static/', '--output_file=static/predictions.csv'])

    return render_template('basic.html', title="Tufa prediction !", body=par)
