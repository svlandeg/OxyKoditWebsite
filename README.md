##  Python & Flask code to run [Tufa example @ OxyKodit website](http://www.oxykodit.com/blog/tufa)

**app.py** contains the [Flask framework](http://flask.pocoo.org/) that parses GET/POST information, keeps track of the image annotations, trains the Machine Learning model and runs the predictions.

**ml_model.py** contains the Python Machine Learning code that takes the pretrained neural net and retrains the last layers with the annotated Tufa examples. This file is based on code from [the Tensorflow github](https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py).

**templates** contains the HTML files, including javascript, for rendering the interactive grid page.

**static** contains the image JPG files.
