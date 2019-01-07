from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# This work was derived from tensorflow/examples/image_retraining (TensorFlow Authors)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# It was further edited by Sofie Van Landeghem @ OxyKodit to enable online Tufa learning
#
# ==============================================================================

from collections import OrderedDict
from datetime import datetime
import os.path
import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# hard-coded FLAGS settings
FLAGS = dict()

FLAGS['final_tensor_name'] = 'final_result'
FLAGS['how_many_training_steps'] = 12
FLAGS['learning_rate'] = 0.01
FLAGS['eval_step_interval'] = 10
FLAGS['train_batch_size'] = 100

# distortions: lets not do them for now
FLAGS['flip_left_right'] = False
FLAGS['random_crop'] = False
FLAGS['random_scale'] = False
FLAGS['random_brightness'] = False

# A module is understood as instrumented for quantization with TF-Lite
# if it contains any of these ops.
FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars',
                  'FakeQuantWithMinMaxVarsPerChannel')

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def get_image_path(image_lists, label_name, index, category):
    """Returns a path to an image for a label at the given index.

    Args:
      image_lists: OrderedDict of training images for each label.
      label_name: Label string we want to get an image for.
      index: Int offset of the image we want. This will be moduloed by the
      available number of images for the label, so it can be arbitrarily large.
      category: Name string of set to pull images from - training, testing, or
      validation.

    Returns:
      File system path string to an image that meets the requested parameters.

    """
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.',
                         label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    my_dir = os.path.dirname(__file__)
    sub_dir = os.path.join(my_dir, label_lists['dir'])
    full_path = os.path.join(sub_dir, base_name)
    return full_path


def run_bottleneck_on_image(sess, image_path, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer.
    Args:
      sess: Current active TensorFlow Session.
      image_path: location of raw JPEG data.
      image_data_tensor: Input data layer in the graph.
      decoded_image_tensor: Output of initial image resizing and preprocessing.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: Layer before the final softmax.
    Returns:
      Numpy array of bottleneck values.
    """
    # First decode the JPEG image, resize it, and rescale the pixel values.
    image_data = tf.gfile.GFile(image_path, 'rb').read()
    resized_input_values = sess.run(decoded_image_tensor,
                                    {image_data_tensor: image_data})
    # Then run it through the recognition network.
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def create_module_graph(module_spec):
    """Creates a graph and loads Hub Module into it.

    Args:
      module_spec: the hub.ModuleSpec for the image module being used.

    Returns:
      graph: the tf.Graph that was created.
      bottleneck_tensor: the bottleneck values output by the module.
      resized_input_tensor: the input images, resized as expected by the module.
      wants_quantization: a boolean, whether the module has been instrumented
        with fake quantization ops.
    """
    height, width = hub.get_expected_image_size(module_spec)
    with tf.Graph().as_default() as graph:
        resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
        m = hub.Module(module_spec)
        bottleneck_tensor = m(resized_input_tensor)
        wants_quantization = any(node.op in FAKE_QUANT_OPS for node in graph.as_graph_def().node)
    return graph, bottleneck_tensor, resized_input_tensor, wants_quantization


def add_jpeg_decoding(module_spec):
    """Adds operations that perform JPEG decoding and resizing to the graph..

    Args:
      module_spec: The hub.ModuleSpec for the image module being used.

    Returns:
      Tensors for the node to feed JPEG data into, and the output of the
        preprocessing steps.
    """
    input_height, input_width = hub.get_expected_image_size(module_spec)
    input_depth = hub.get_num_image_channels(module_spec)
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    # Convert from full range of uint8 to range [0,1] of float32.
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                          tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    return jpeg_data, resized_image


def add_final_retrain_ops(class_count, final_tensor_name, bottleneck_tensor, quantize_layer, is_training):
    """Adds a new softmax and fully-connected layer for training and eval.

    We need to retrain the top layer to identify our new classes, so this function
    adds the right operations to the graph, along with some variables to hold the
    weights, and then sets up all the gradients for the backward pass.

    The set up for the softmax and fully-connected layers is based on:
    https://www.tensorflow.org/tutorials/mnist/beginners/index.html

    Args:
      class_count: Integer of how many categories of things we're trying to
          recognize.
      final_tensor_name: Name string for the new final node that produces results.
      bottleneck_tensor: The output of the main CNN graph.
      quantize_layer: Boolean, specifying whether the newly added layer should be
          instrumented for quantization with TF-Lite.
      is_training: Boolean, specifying whether the newly add layer is for training
          or eval.

    Returns:
      The tensors for the training and cross entropy results, and tensors for the
      bottleneck input and ground truth input.
    """
    batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
    assert batch_size is None, 'We want to work with arbitrary batch size.'
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor,
            shape=[batch_size, bottleneck_tensor_size],
            name='BottleneckInputPlaceholder')

        ground_truth_input = tf.placeholder(tf.int64, [batch_size], name='GroundTruthInput')

    # Organizing the following ops so they are easier to see in TensorBoard.
    layer_name = 'final_retrain_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal(
                [bottleneck_tensor_size, class_count], stddev=0.001)
            layer_weights = tf.Variable(initial_value, name='final_weights')

        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')

        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

    # The tf.contrib.quantize functions rewrite the graph in place for
    # quantization. The imported model graph has already been rewritten, so upon
    # calling these rewrites, only the newly added final layer will be transformed.
    if quantize_layer:
        if is_training:
            tf.contrib.quantize.create_training_graph()
        else:
            tf.contrib.quantize.create_eval_graph()

    tf.summary.histogram('activations', final_tensor)

    # If this is an eval graph, we don't need to add loss ops or an optimizer.
    if not is_training:
        return None, None, bottleneck_input, ground_truth_input, final_tensor

    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
            labels=ground_truth_input, logits=logits)

    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(FLAGS['learning_rate'])
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
            final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results.

    Args:
      result_tensor: The new final node that produces results.
      ground_truth_tensor: The node we feed ground truth data
      into.

    Returns:
      Tuple of (evaluation step, prediction).
    """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(prediction, ground_truth_tensor)
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def get_bottlenecks(image_lists, category, sess, image_data_tensor,
                                  decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor):
    """
    Retrieves bottleneck values for cached images.
    If no distortions are being applied, this function can retrieve the cached
    bottleneck values directly from disk for images. It picks a random set of
    images from the specified category.
    Args:
      image_lists: OrderedDict of training images for each label.
      category: Name string of which set to pull from - training, testing, or
      validation.
    Returns:
      List of bottleneck arrays, their corresponding ground truths, and the
      relevant filenames.
    """
    bottlenecks = []
    ground_truths = []
    filenames = []

    # Retrieve all bottlenecks.
    for label_index, label_name in enumerate(image_lists.keys()):
        for image_index, image_name in enumerate(
                image_lists[label_name][category]):
            image_path = get_image_path(image_lists, label_name, image_index, category)

            bottleneck = run_bottleneck_on_image(sess, image_path, image_data_tensor,
                                                 decoded_image_tensor, resized_input_tensor,
                                                 bottleneck_tensor)

            bottlenecks.append(bottleneck)
            ground_truths.append(label_index)
            filenames.append(image_name)

    return bottlenecks, ground_truths, filenames


def build_eval_graph(sess, module_spec, class_count):
    """Builds an restored eval session without train operations for exporting.

    Args:
      sess: Session object
      module_spec: The hub.ModuleSpec for the image module being used.
      class_count: Number of classes

    Returns:
      The eval graph.
      The bottleneck input, ground truth, eval step, and prediction tensors.
    """
    eval_graph = sess.graph

    # If quantized, we need to create the correct eval graph for exporting.
    height, width = hub.get_expected_image_size(module_spec)

    resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
    m = hub.Module(module_spec)
    bottleneck_tensor = m(resized_input_tensor)
    wants_quantization = any(node.op in FAKE_QUANT_OPS for node in eval_graph.as_graph_def().node)

    with eval_graph.as_default():
        # Add the new layer
        (_, _, bottleneck_input,
         ground_truth_input, final_tensor) = add_final_retrain_ops(
            class_count, FLAGS['final_tensor_name'], bottleneck_tensor,
            wants_quantization, is_training=False)

        add_evaluation_step(final_tensor, ground_truth_input)

    return eval_graph


def main(tufa_image_list, nontufa_image_list, all_image_list, model_url, img_size):
    tf.logging.set_verbosity(tf.logging.INFO)

    # create lists of all the images - not working through the actual folder system
    image_lists = OrderedDict([('nontufa', {'dir': 'static/img', 'training': nontufa_image_list}),
                               ('tufa', {'dir': 'static/img', 'training': tufa_image_list})])

    # order needs to be the same as in the dict!
    labels = ['nontufa', 'tufa']

    class_count = len(image_lists.keys())

    # Set up the pre-trained graph.
    module_spec = hub.load_module_spec(model_url)
    graph, bottleneck_tensor, resized_image_tensor, wants_quantization = (create_module_graph(module_spec))

    # Add the new layer that we'll be training.
    with graph.as_default():
        (train_step, cross_entropy, bottleneck_input,
         ground_truth_input, final_tensor) = add_final_retrain_ops(
            class_count, FLAGS['final_tensor_name'], bottleneck_tensor,
            wants_quantization, is_training=True)

    with tf.Session(graph=graph) as sess:
        # Initialize all weights: for the module to their pretrained values,
        # and for the newly added retraining layer to random initial values.
        init = tf.global_variables_initializer()
        sess.run(init)

        # Set up the image decoding sub-graph.
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(module_spec)

        # Create the operations we need to evaluate the accuracy of our new layer.
        evaluation_step, _ = add_evaluation_step(final_tensor, ground_truth_input)

        merged = tf.summary.merge_all()

        # Create a train saver that is used to restore values into an eval graph
        # when exporting models.
        train_saver = tf.train.Saver()

        # Get a batch of input bottleneck values
        (train_bottlenecks,
         train_ground_truth, _) = get_bottlenecks(image_lists, 'training', sess, jpeg_data_tensor,
                                                                decoded_image_tensor, resized_image_tensor,
                                                                bottleneck_tensor)

        # Run the training for as many cycles as requested on the command line.
        for i in range(FLAGS['how_many_training_steps']):
            # Feed the bottlenecks and ground truth into the graph, and run a training
            # step. Capture training summaries for TensorBoard with the `merged` op.
            train_summary, _ = sess.run(
                [merged, train_step],
                feed_dict={bottleneck_input: train_bottlenecks,
                           ground_truth_input: train_ground_truth})

            # Every so often, print out how well the graph is training.
            is_last_step = (i + 1 == FLAGS['how_many_training_steps'])
            if (i % FLAGS['eval_step_interval']) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run(
                    [evaluation_step, cross_entropy],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth})
                tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                                (datetime.now(), i, train_accuracy * 100))
                tf.logging.info('%s: Step %d: Cross entropy = %f' %
                                (datetime.now(), i, cross_entropy_value))

        eval_graph = build_eval_graph(sess, module_spec, class_count)
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                        eval_graph.as_graph_def(),
                                                                        [FLAGS['final_tensor_name']])

        output_graph = tf.Graph()
        with output_graph.as_default():
            tf.import_graph_def(output_graph_def)

        # obtain the final graph
        input_layer = "Placeholder"
        output_layer = "final_result"
        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = output_graph.get_operation_by_name(input_name)
        output_operation = output_graph.get_operation_by_name(output_name)

    # predict
    input_height = img_size
    input_width = img_size
    input_mean = 0
    input_std = img_size

    pred_list = []
    for file_name in all_image_list:
        if file_name.lower().endswith(".jpg"):
            t = read_tensor_from_image_file(
                os.path.join('static/img', file_name),
                input_height=input_height,
                input_width=input_width,
                input_mean=input_mean,
                input_std=input_std)

            with tf.Session(graph=output_graph) as sess:
                results = sess.run(output_operation.outputs[0], {
                    input_operation.outputs[0]: t
                })
            results = np.squeeze(results)

            top_k = results.argsort()[-5:][::-1]

            # we only care about the tufa % prediction
            for i in top_k:
                if labels[i] == "tufa":
                    pred_list.append(results[i])

    return pred_list


if __name__ == '__main__':
    start = time.time()
    my_tufa_image_list = ['img2.jpg']
    my_nontufa_image_list = ['img0.jpg', 'img1.jpg']
    my_all_image_list = []
    for i in range(25):
        my_all_image_list.append("img" + str(i) + ".jpg")
    my_pred_list = main(my_tufa_image_list, my_nontufa_image_list, my_all_image_list)

    for img, pred in zip(my_all_image_list, my_pred_list):
        print(img, pred)
    end = time.time()

    print()
    print("timing", end - start)
