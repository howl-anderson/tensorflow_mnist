from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.training.monitored_session import MonitoredTrainingSession
import tensorflow as tf

FLAGS = None

LOG_DIR = os.path.abspath('./log')
MODEL_DIR = os.path.abspath('./model')
MNIST_DIR = os.path.abspath('./mnist/input_data')
PATH_TO_SPRITE_IMAGE = os.path.abspath('./image/mnist_10k_sprite.png')


class MnistSoftmax(object):
    def __init__(self):
        self.__graph = None
        self.mnist = input_data.read_data_sets(MNIST_DIR, one_hot=True)

        self.serialized_tf_example = None
        self.train_step = None
        self.test_data = None
        self.x = None
        self.y = None
        self.y_ = None
        self.accuracy = None
        self.values = None

    def build_graph(self):
        self.__graph = tf.Graph()
        with self.__graph.as_default():

            # Create the model
            self.serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
            feature_configs = {
                'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32), }
            tf_example = tf.parse_example(self.serialized_tf_example, feature_configs)
            self.x = tf.identity(tf_example['x'],
                            name='x')  # use tf.identity() to assign name
            # x = tf.placeholder(tf.float32, [None, 784], name='x')
            W = tf.Variable(tf.zeros([784, 10]))
            b = tf.Variable(tf.zeros([10]))
            self.y = tf.matmul(self.x, W) + b

            self.test_data = tf.get_variable(
                initializer=(
                    tf.matmul(self.mnist.test.images,
                              W.initialized_value()) + b.initialized_value()
                ),
                name='test_data'
            )

            self.values, indices = tf.nn.top_k(self.y, 10)
            table = tf.contrib.lookup.index_to_string_table_from_tensor(
                tf.constant([str(i) for i in range(10)]))
            self.prediction_classes = table.lookup(tf.to_int64(indices))

            # Define loss and optimizer
            self.y_ = tf.placeholder(tf.float32, [None, 10], name='y')

            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
            tf.summary.scalar('loss', cross_entropy)

            global_step = tf.Variable(0, name='global_step', trainable=False)

            self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy, global_step=global_step)

            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # Train
            summary_writer = tf.summary.FileWriter(LOG_DIR)

            meta_data_file = os.path.join(LOG_DIR, 'metadata.tsv')

            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = self.test_data.name
            embedding.sprite.image_path = PATH_TO_SPRITE_IMAGE
            embedding.sprite.single_image_dim.extend([28, 28])
            embedding.metadata_path = meta_data_file
            projector.visualize_embeddings(summary_writer, config)

            with open(meta_data_file, 'wt') as fd:
                fd.write("\n".join(map(lambda x: str(x),
                                       np.argmax(self.mnist.test.labels, 1))))

    def train(self):
        with self.__graph.as_default():
            with MonitoredTrainingSession(checkpoint_dir=LOG_DIR,
                                          save_checkpoint_secs=1) as sess:
                for step in range(1000):
                    if sess.should_stop():
                        break

                    batch_xs, batch_ys = self.mnist.train.next_batch(100)
                    sess.run([self.train_step, self.test_data],
                             feed_dict={self.x: batch_xs, self.y_: batch_ys})

                print(sess.run(self.accuracy, feed_dict={self.x: self.mnist.test.images,
                                                    self.y_: self.mnist.test.labels}))

    def export(self):
        with self.__graph.as_default():
            sess = tf.Session()

            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR))

            builder = tf.saved_model.builder.SavedModelBuilder(MODEL_DIR)

            # Build the signature_def_map.
            classification_inputs = tf.saved_model.utils.build_tensor_info(
                self.serialized_tf_example)
            classification_outputs_classes = tf.saved_model.utils.build_tensor_info(
                self.prediction_classes)
            classification_outputs_scores = tf.saved_model.utils.build_tensor_info(
                self.values)

            classification_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                            classification_inputs
                    },
                    outputs={
                        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                            classification_outputs_classes,
                        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                            classification_outputs_scores
                    },
                    method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

            tensor_info_x = tf.saved_model.utils.build_tensor_info(self.x)
            tensor_info_y = tf.saved_model.utils.build_tensor_info(self.y)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'images': tensor_info_x},
                    outputs={'scores': tensor_info_y},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

            builder.add_meta_graph_and_variables(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict_images':
                        prediction_signature,
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        classification_signature,
                },
                legacy_init_op=legacy_init_op
            )

            builder.save()


if __name__ == '__main__':
    ms = MnistSoftmax()
    ms.build_graph()
    ms.train()
    ms.export()
