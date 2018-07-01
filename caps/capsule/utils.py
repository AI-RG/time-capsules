import numpy as np
import tensorflow as tf

"""
    Utility functions for capsule networks.
"""

def conv(x, scope, *, nf, rf, stride, pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, nf]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, nf, 1, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [nf] if one_dim_bias else [1, nf, 1, 1]
    nin = x.get_shape()[channel_ax].value
    wshape = [rf, rf, nin, nf]
    with tf.variable_scope(scope):
        w = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        b = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            b = tf.reshape(b, bshape)
        return b + tf.nn.conv2d(x, w, strides=strides, padding=pad, data_format=data_format)

def fc(x, scope, nh, act=tf.nn.relu, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return act(tf.matmul(x, w)+b)

    
def capsule_conv(x, scope, rf, stride, ncaps, capsdim):
    shape = x.get_shape().as_list()
    with tf.variable_scope(scope):
        out = conv(x, 'c', nf=ncaps*capsdim, rf=rf, stride=stride)
        newshape = [-1] + out.get_shape().as_list()[1:-1] + [ncaps, capsdim]
        out = tf.reshape(out, newshape)
        return _squash(out, axis=4) #activation is squashing along capsule-vector dimension
            
def capsule(x, scope, ncaps, capsdim, init_scale=1.0, from_conv=True):
    if from_conv:
        x = conv_to_caps(x)
    with tf.variable_scope(scope):
        assert len(x.get_shape()) == 3
        _, i, k = x.get_shape().as_list()
        w = tf.get_variable('w', [ncaps, i, capsdim, k], initializer=ortho_init(init_scale))
        u = tf.einsum('jilk,bik->bjil', w, x)
        return _route(u, 'route')
        
def _route(x, scope, r=5, reset=True):
    """
    Dynamic routing function.
    Input of shape [B, J, I, K]: batch, ncaps, nincaps, capsdim.
    """
    b = x[:, :, :, 0] * 0.0
    with tf.variable_scope(scope):
        for _ in range(r):
            c = tf.nn.softmax(b, axis=1)
            s = tf.einsum('bjik,bji->bjk', x, c)
            v = _squash(s, axis=2, inroute=True)
            b += tf.einsum('bjk,bjik->bji', v, x) 
        return v
    
def _squash(x, axis=3, eps=1e-6, inroute=False):
    n2 = tf.reduce_sum(tf.square(x), axis=axis)
    # workaround: broadcasting does not allow 'None' dimensions in multiple arguments.
    # this fix assumes (as is generally necessary) that both None's will be equal when fed.
    d = x.get_shape().as_list()[-1]
    if not inroute:
        n2pad = tf.einsum('bijk,l->bijkl', n2, tf.ones([d], tf.float32))
    else:
        n2pad = tf.einsum('bi,j->bij', n2, tf.ones([d], tf.float32))
    return (x / tf.maximum(tf.sqrt(n2pad), eps)) * n2pad / (1.0 + n2pad)
        
def conv_to_fc(x):
    nh = np.prod([v.value for v in x.get_shape()[1:]])
    return tf.reshape(x, [-1, nh])
    
def conv_to_caps(x):
    shape = x.get_shape().as_list()
    nh = np.prod([v for v in shape[1:-1]])
    return tf.reshape(x, [-1, nh, shape[-1]])
    
def loss_classification(output, target, mp=0.9, mm=0.1, lam=0.5):
    # shape [b, j, k], [b, j]
    output_norm = tf.sqrt(tf.reduce_sum(tf.square(output), axis=2))
    losses = target * tf.square(tf.maximum(0.0, mp - output_norm)) + lam * (1.0 - target) * tf.square(tf.maximum(0.0, output_norm - mm))
    return tf.reduce_sum(losses, axis=[0,1])

def loss_reconstruction(output, input):
    """
    Reconstruction loss.
    Args:
        output: output of decoder network
        input: original images (in flat format) 
    """
    return tf.reduce_sum(tf.square(output - input))

def loss_total(output, target, input, reconst, rc_weight=0.0005, mp=0.9, mm=0.1, lam=0.5):
    # returns triple: total loss, cl_loss, rc_loss
    # (note that the returned rc_loss is not multiplied by the rc_loss coefficient)
    cl_loss = loss_classification(output, target, mp=mp, mm=mm, lam=lam)
    rc_loss = loss_reconstruction(reconst, input)
    return cl_loss + rc_weight * rc_loss, cl_loss, rc_loss
    
def accuracy(output, target):
    output_norm2 = tf.reduce_sum(tf.square(output), axis=2)
    correct = tf.equal(tf.argmax(output_norm2, axis=1), tf.argmax(target, axis=1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))
    
def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init
