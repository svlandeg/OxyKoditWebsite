import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

import ml_model


def get_bottleneck(bottleneck_path):
    """
    Retrieves cached calculates bottleneck values for an image.
    If a cached version of the bottleneck data exists on-disk, return that,
    otherwise calculate the data and save it to disk for future use.
    Returns:
      Numpy array of values produced by the bottleneck layer for the image.
    """

    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values


def create_bottleneck_file(bottleneck_path, image_path, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
    """Create a single bottleneck file."""
    tf.logging.info('Creating bottleneck at ' + bottleneck_path)
    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = tf.gfile.GFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                     str(e)))
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer.
    Args:
      sess: Current active TensorFlow Session.
      image_data: String of raw JPEG data.
      image_data_tensor: Input data layer in the graph.
      decoded_image_tensor: Output of initial image resizing and preprocessing.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: Layer before the final softmax.
    Returns:
      Numpy array of bottleneck values.
    """
    # First decode the JPEG image, resize it, and rescale the pixel values.
    resized_input_values = sess.run(decoded_image_tensor,
                                    {image_data_tensor: image_data})
    # Then run it through the recognition network.
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


if __name__ == '__main__':
    my_image_list = []
    my_bottleneck_list = []
    module_spec = hub.load_module_spec(ml_model.FLAGS['tfhub_module'])
    graph, my_bottleneck_tensor, my_resized_input_tensor, wants_quantization = (ml_model.create_module_graph(module_spec))

    # Add the new layer that we'll be training.
    with graph.as_default():
        (train_step, cross_entropy, bottleneck_input,
         ground_truth_input, final_tensor) = ml_model.add_final_retrain_ops(
            2, ml_model.FLAGS['final_tensor_name'], my_bottleneck_tensor,
            wants_quantization, is_training=True)

    with tf.Session(graph=graph) as my_sess:
        # Initialize all weights: for the module to their pretrained values,
        # and for the newly added retraining layer to random initial values.
        init = tf.global_variables_initializer()
        my_sess.run(init)

        # Set up the image decoding sub-graph.
        my_jpeg_data_tensor, my_decoded_image_tensor = ml_model.add_jpeg_decoding(module_spec)

        for i in range(42):
            my_image = "static/img/img" + str(i) + ".jpg"
            my_bottleneck = "static/bottlenecks/img" + str(i) + ".txt"

            create_bottleneck_file(my_bottleneck, my_image, my_sess, my_jpeg_data_tensor,
                                   my_decoded_image_tensor, my_resized_input_tensor,
                                   my_bottleneck_tensor)
