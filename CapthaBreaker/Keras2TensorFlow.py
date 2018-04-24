import os
import os.path as osp
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from keras.models import load_model
from keras import backend as K

class Keras2TensorFlow:

    """
    Convert Keras model to Tensorflow proto buffer file format.
    Based on:

    https://github.com/amir-abdi/keras_to_tensorflow/blob/master/keras_to_tensorflow.ipynb
    http://www.bitbionic.com/2017/08/18/run-your-keras-models-in-c-tensorflow/
    """
    def convert(self, model_path, output_dir, prefix = 'conversor_output', name = 'output_graph.pb', num_outputs=1):

        """
        Convert Keras model to Tensorflow proto buffer file format.

        :param model_path: Path of your model in format .h5
        :param output_dir: Directory to output the proto buffer file and the graph.
        :param prefix:
        :param name:
        :param num_outputs:
        :return:
        """

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        K.set_learning_phase(0)

        net_model = load_model(model_path)

        pred = [None] * num_outputs
        pred_node_names = [None] * num_outputs

        for i in range(num_outputs):
            pred_node_names[i] = prefix + '_' + str(i)
            pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])

        print('Use this names as input and output nodes to use native Tensorflow:')
        print('Output nodes names are: ', pred_node_names)
        print('Input nodes names are: ', net_model.input_names)
        sess = K.get_session()

        # Write the graph in human readable
        f = 'graph_def_for_reference.pb.ascii'
        tf.train.write_graph(sess.graph.as_graph_def(), output_dir, f, as_text=True)
        print('Saved the graph definition at: ', osp.join(output_dir, f))

        # Write the graph in binary .pb file
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
        graph_io.write_graph(constant_graph, output_dir, name, as_text=False)
        print('Saved the constant graph at: ', osp.join(output_dir, name))


if __name__ == '__main__':

    conversor = Keras2TensorFlow()

    model_path = 'U:\\Dataset\\CNN_Model\\CNN_Model.h5'
    output_dir =  'U:\\Dataset\\CNN_Model\\'


    conversor.convert(model_path = model_path, output_dir = output_dir)