# Copyright 2018 Samuel B. Johnson

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ===========================================================================================

""" An implementation of the Capsule Network Model
    described in 'Dynamic Routing Between Capsules' (Sabour, Frost, and Hinton),
    available at arXiv:1710.09829
"""

import tensorflow as tf
import numpy as np

import random
import time
from math import sqrt

from .utils import conv, fc, capsule_conv, capsule, loss_total, accuracy

FLAGS = tf.app.flags.FLAGS
   
def caps_cnn(flat_images, nf, rf, s):
    """
    Modified CNN architecture for first layer of capsule netowrk.
    """
    n = int(sqrt(flat_images.get_shape().as_list()[1]))
    images = tf.reshape(flat_images, [-1, n, n, 1]) # transform input to standard NHWC format
    return tf.nn.relu(conv(images, 'c1', nf=nf, rf=rf, stride=s, init_scale=np.sqrt(2)))
    
def decoder(output, input, target, **decoder_kwargs):
    """
    Decoder network that attempts to reconstruct original image from encoded version
    (output of capsule network)
    """
    d = decoder_kwargs
    with tf.name_scope('decoder'):
        shape = output.get_shape().as_list()
        masked_output = tf.einsum('bij,bi->bij', output, target) # use one-hot target as mask
        new_shape = [-1, d['din']*d['nclasses']]
        masked_output = tf.reshape(masked_output, new_shape)
        reconst = fc(masked_output, 'fc1', d['d1'])
        reconst = fc(reconst, 'fc2', d['d2'])
        return fc(reconst, 'fc3', d['d3'], act=tf.nn.sigmoid)
        
def network(x, scope='network', reuse=False, **layer_kwargs):
    """
    Main, 3-layer capsule network.
    """
    cv = layer_kwargs['layer1']
    cc = layer_kwargs['layer2']
    c = layer_kwargs['layer3']
    with tf.variable_scope(scope, reuse=reuse):
            h = caps_cnn(x, nf=cv['c'], rf=cv['k'], s=cv['s'])
            h = capsule_conv(h, 'capsconv', cc['k'], cc['s'], cc['c'], cc['d'])
            return capsule(h, 'caps', c['w'], c['d'], from_conv=True)

class Capsule(object):
    """
    Class to instantiate the Capsule Network architecture,
    including data placeholders.
    """
    def __init__(self, config):
    
        # set parameters, hyperparameters from config object
        self.nx = config.input_size
        self.nclasses = config.n_classes
        
        self.X = tf.placeholder(tf.float32, [None, self.nx])
        self.Y = tf.placeholder(tf.float32, [None, self.nclasses])
        
        layer_kwargs = {}
        layer_kwargs['layer1'] = config.layer1_kwargs
        layer_kwargs['layer2'] = config.layer2_kwargs
        layer_kwargs['layer3'] = config.layer3_kwargs
        decoder_kwargs = config.decoder_kwargs
    
        with tf.variable_scope('capsule'):
            self.h = network(self.X, **layer_kwargs)     
            self.reconst = decoder(self.h, self.X, self.Y, **decoder_kwargs)
            
            self.loss, self.cl_loss, self.rc_loss = loss_total(output=self.h, target=self.Y, \
                input=self.X, reconst=self.reconst, mp=config.mp, mm=config.mm, lam=config.lam)
            self.accuracy = accuracy(self.h, self.Y)

        
