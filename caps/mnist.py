from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _mypath

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import json
import datetime
import getopt
import time 

import capsules as caps


def mnist_train(model, mnist, cp_path, graph_path,
                batch_size = 128, dropout = 0.75, skip_step = 10, save_step = 100, n_epochs = 1,
                restore_dir = None, extra_verbose = False):
    """
    Function that trains a capsule network on the mnist dataset,
    stores intermediate/final states and prints out intermediate/final results
    Args:
        model : Network object
        mnist : DataSets object, assumed to be constructed from the MNIST dataset
        cp_path : checkpoint output path
        graph_path : graph output path
        batch_size : batch size
        dropout : dropout rate
        skip_step : steps to skip before showing average loss and average accuracy
        save_step : steps to skip before saving state
        n_epochs : number of epochs to run
        restore_dir : location of state to restore
        extra_verbose : print extra information
    """
    n_classes = model.n_classes

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # Restore session if instructed
        if restore_dir != None:
            ckpt = tf.train.get_checkpoint_state(restore_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Session restored!")

        # Set up parameters, variables, etc
        writer = tf.summary.FileWriter(graph_path, sess.graph)
        initial_step = model.get_global_step().eval()
        start_time = time.time()
        n_batches = int(mnist.train.num_examples / batch_size)
        n_validation_batches = int(mnist.validation.num_examples / batch_size)
        n_test_batches = int(mnist.test.num_examples / batch_size)
        total_loss = 0.0
        total_accuracy = 0.0

        # Mini-validation batch whose accuracy will be tested every (skip_step) steps
        X_val, Y_val = mnist.validation.next_batch(batch_size)


        # For rotationally invariant networks, we might want to
        # set the number of classes to 9, 6 and 9 being related to each other
        # by a rotation
        if n_classes == 9: 
            Y_val = caps.rot_objective(Y_val)

        # Train away!
        for index in range(initial_step, n_batches * n_epochs):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            if n_classes == 9:
                Y_batch = caps.rot_objective(Y_batch)
            _, loss_batch, summary, l2, acc_batch  = \
                sess.run([model.get_optimizer(), model.get_loss(), model.get_summary_op(), model.L2, model.get_accuracy()], 
                         feed_dict={model.get_X(): X_batch, model.get_Y():Y_batch,
                                    model.get_X_target(): X_batch, model.dropout: dropout}) 
            if extra_verbose:
                print(loss_batch)
                print(l2)
            writer.add_summary(summary, global_step=index)
            total_loss += loss_batch
            total_accuracy += acc_batch

            # Print intermediate results on screen
            if (index + 1) % skip_step == 0:
                print('Step {}'.format(index + 1))
                print('Average loss: {:6.2f}'.format(total_loss/skip_step))
                total_loss = 0.0
                print('Average accuracy: {0}'.format(total_accuracy/skip_step))
                total_accuracy = 0.0
                val_accuracy, = sess.run([model.get_accuracy()], 
                                          feed_dict={model.X: X_val, model.Y: Y_val, model.dropout: 1.0}) 
                print('Validation batch accuracy: {0}'.format(val_accuracy))

            # Save state
            if (index + 1) % save_step == 0:
                saver.save(sess, cp_path, index)

            # End of epoch
            if (index + 1) % n_batches == 0:
                print("End of epoch {}".format( (index + 1) / n_batches ))

                # Compute accuracy of entire validation set
                total_correct_preds = 0

                for i in range(n_validation_batches):
                    X_batch, Y_batch = mnist.validation.next_batch( batch_size )
                    if n_classes == 9:
                        Y_batch = caps.rot_objective(Y_batch)
                    accuracy, = sess.run([model.get_accuracy()], 
                                         feed_dict={model.X: X_batch, model.Y: Y_batch, model.dropout: 1.0}) 
                    total_correct_preds += accuracy
    
                print("Validation accuracy : {0}".format(total_correct_preds/n_validation_batches))
        
        saver.save(sess, cp_path, index)
        print("Optimization Finished!")
        print("Total time: {0} seconds".format(time.time() - start_time))
        
        # Compute accuracy of entire test set
        print("Testing on test set ...")
        total_correct_preds = 0
        for i in range(n_test_batches):
            X_batch, Y_batch = mnist.test.next_batch( batch_size )
            if n_classes == 9:
                Y_batch = caps.rot_objective(Y_batch)
            accuracy, = sess.run([model.get_accuracy()], 
                                 feed_dict={model.X: X_batch, model.Y: Y_batch, model.dropout: 1.0}) 
            total_correct_preds += accuracy
        print("Test accuracy : {0}".format(total_correct_preds/n_test_batches))



def main():

    # get config file
    opts, args = getopt.getopt(sys.argv[1:], "c:h")
    cfile = None
    for opt, arg in opts:
        if opt == "-c":
            cfile = arg
        if opt == "-h":
            print("python mnist_run.py -c <json config file>")
            return
    if cfile == None:
        raise ValueError("No input json file given!")
    configs = json.load(open(cfile, "rb"))

    # Unpack configs
    input_dict = configs["input_dict"]
    dir_configs = configs["dir_configs"]
    conv_params = configs["conv_params"]
    train_params = configs["train_params"]

    # Set up relevant directories for input and output
    check_dir = dir_configs["check_dir"]
    graph_dir = dir_configs["graph_dir"]
    data_dir = dir_configs["data_dir"]
    if dir_configs["date_time"]:
        ct = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
        check_dir += ct
        graph_dir += ct

    # Make directories if necessary
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
    checkpoint = os.path.join(check_dir, "./checkpoint")

    # Set up capsule network
    model = caps.CapsNet(**input_dict)
    model.initialize(conv_params = conv_params)
    mnist = input_data.read_data_sets(data_dir, one_hot=True)

    # Train
    mnist_train(model, mnist, checkpoint, graph_dir, **train_params)



if __name__ == "__main__":
main()
