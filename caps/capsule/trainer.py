from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # mute low-priority warnings from below line
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import datetime
import time

#import model as m
from . import model as m
from .config import CapsuleConfig


def define_flags():
    # input parameters
    tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size.")
    tf.app.flags.DEFINE_integer("input_size", 784, "Input dimension, viz. flattened MNIST.")
    tf.app.flags.DEFINE_integer("image_size", 28, "Height (or width) of MNIST image, square root of above.")
    tf.app.flags.DEFINE_integer("n_classes", 10, "Number of classes for MNIST, the digits 0 - 9.")
    # layer 1 parameters
    tf.app.flags.DEFINE_integer("conv1_k", 9, "Kernel size for first conv layer.")
    tf.app.flags.DEFINE_integer("conv1_s", 1, "Stride length for first conv layer.")
    tf.app.flags.DEFINE_integer("conv1_output_channels", 256, "Output channels (features to extract) for first conv layer.")
    # layer 2 parameters
    tf.app.flags.DEFINE_integer("capsule_conv_k", 9, "Kernel width for capsule conv layer.")
    tf.app.flags.DEFINE_integer("capsule_conv_s", 2, "Stride length for capsule conv layer.")
    tf.app.flags.DEFINE_integer("capsule_conv_output_channels", 32, "Output channels (i.e. number of capsules) in capsule conv layer.")
    tf.app.flags.DEFINE_integer("capsule_conv_dim", 8, "Dimension of capsule vector output of capsule conv layer.")
    # layer 3 parameters
    tf.app.flags.DEFINE_integer("capsule_dim", 16, "Dimension of capsule vector output of capsule layer.")
    tf.app.flags.DEFINE_integer("capsule_width", 10, "Number of capsules in output layer (one for each class, i.e. MNIST digit.")
    # decoder parameters
    tf.app.flags.DEFINE_integer("decoder_dim1", 512, "Dimension of first fc decoder layer.")
    tf.app.flags.DEFINE_integer("decoder_dim2", 1024, "Dimension of second fc decoder layer.")
    tf.app.flags.DEFINE_integer("decoder_dim3", 784, "Dimension of third fc decoder layer.")
    # hyperparameters
    tf.app.flags.DEFINE_float("lr", 0.0005, "Learning rate.")
    tf.app.flags.DEFINE_float("mp", 0.9, "Margin parameter for correct classifications in loss function.")
    tf.app.flags.DEFINE_float("mm", 0.1, "Margin parameter for incorrect classifications in loss function.")
    tf.app.flags.DEFINE_float("lam", 0.5, "Relative weight of incorrect compared to correct classification errors in loss function.")
    tf.app.flags.DEFINE_float("reconst_weight", 0.0005, "Relative weight of reconstruction loss compared to classification loss in total loss function.")
    # run parameters
    tf.app.flags.DEFINE_string("data_dir", './mnist/data', "Directory where data will be stored.")
    tf.app.flags.DEFINE_string("ckpt_dir", './mnist/ckpt', "Directory where model will be saved.")
    tf.app.flags.DEFINE_integer("print_every", 100, "Number of batches to skip before evaluating.")
    tf.app.flags.DEFINE_integer("save_every", int(1e3), "Number of batches to skip before saving.")
    tf.app.flags.DEFINE_integer("steps", int(1e4), "Number of batches to train on.")

FLAGS = tf.app.flags.FLAGS
define_flags()


def initialize(config):
    return m.Capsule(config)

def start_and_train():
    """Train the capsule network."""
    config = CapsuleConfig(FLAGS)
  
    data_dir = config.data_dir
    ckpt_dir = config.ckpt_dir
    
    print_every = config.print_every
    save_every = config.save_every
    
    mnist = input_data.read_data_sets(data_dir, one_hot=True)
    model = initialize(config)
    
    batch_size = config.batch_size
    n_batches = int(mnist.train.num_examples / batch_size)
    steps = config.steps
    
    adam = tf.train.AdamOptimizer(learning_rate=config.lr)
    train_step = adam.minimize(model.loss)

  
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
    
        for i in range (steps):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={model.X: X_batch, model.Y: Y_batch})
            # evaluate, print, and log accuracy
            if i % print_every == 0:
                X_test_batch, Y_test_batch = mnist.test.next_batch(batch_size)
                acc = sess.run(model.accuracy, feed_dict={model.X: X_test_batch, model.Y: Y_test_batch})
                print("Batch: ", i, "; Accuracy: ", acc)
            # save model checkpoint
            # TODO
            if i % save_every == 0:
                pass
