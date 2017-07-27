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
PATH_TO_SPRITE_IMAGE = os.path.abspath('./image/mnist_10k_sprite.png')


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    test_data = tf.get_variable(
        initializer=(
            tf.matmul(mnist.test.images,
                      W.initialized_value()) + b.initialized_value()
        ),
        name='test_data'
    )

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    tf.summary.scalar('loss', cross_entropy)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy, global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    meta_data_file = os.path.join(LOG_DIR, 'metadata.tsv')

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = test_data.name
    embedding.sprite.image_path = PATH_TO_SPRITE_IMAGE
    embedding.sprite.single_image_dim.extend([28, 28])
    embedding.metadata_path = meta_data_file
    projector.visualize_embeddings(summary_writer, config)

    with open(meta_data_file, 'wt') as fd:
        fd.write("\n".join(map(lambda x: str(x),
                               np.argmax(mnist.test.labels, 1))))

    with MonitoredTrainingSession(checkpoint_dir=LOG_DIR,
                                  save_checkpoint_secs=1) as sess:
        for step in range(1000):
            if sess.should_stop():
                break

            batch_xs, batch_ys = mnist.train.next_batch(100)
            _, _, summary = sess.run([train_step, test_data, summary_op],
                                  feed_dict={x: batch_xs, y_: batch_ys})

        print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                            y_: mnist.test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='./mnist/input_data',
                        help='Directory for storing input data')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
