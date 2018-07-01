"""Config class.
   This class holds configuration parameters for
   the capsule network, taking them from tf.app.flags.
"""

class CapsuleConfig(object):
  """Initial configuration settings for model"""

  config_keys = '''batch_size input_size image_size n_classes conv1_k
  conv1_s conv1_output_channels capsule_conv_k
  capsule_conv_s capsule_conv_output_channels capsule_conv_dim
  capsule_dim capsule_width decoder_dim1 decoder_dim2 decoder_dim3
  lr mp mm lam reconst_weight data_dir ckpt_dir print_every save_every steps
  '''.split()

  def __init__(self, FLAGS, **kws):
    for key in self.config_keys:
      val = kws.get(key, getattr(FLAGS, key, None))
      setattr(self, key, val)
    
    # set kwarg dictionaries for layers of network
    self.layer1_kwargs = {} # layer 1
    self.layer1_kwargs['k'] = self.conv1_k
    self.layer1_kwargs['s'] = self.conv1_s
    self.layer1_kwargs['c'] = self.conv1_output_channels
    self.layer2_kwargs = {} # layer 2
    self.layer2_kwargs['k'] = self.capsule_conv_k
    self.layer2_kwargs['s'] = self.capsule_conv_s
    self.layer2_kwargs['c'] = self.capsule_conv_output_channels
    self.layer2_kwargs['d'] = self.capsule_conv_dim
    self.layer3_kwargs = {} # layer 3
    self.layer3_kwargs['d'] = self.capsule_dim
    self.layer3_kwargs['w'] = self.capsule_width
    self.decoder_kwargs = {} # layer 4 (decoder)
    self.decoder_kwargs['d1'] = self.decoder_dim1
    self.decoder_kwargs['d2'] = self.decoder_dim2
    self.decoder_kwargs['d3'] = self.decoder_dim3
    self.decoder_kwargs['nclasses'] = self.n_classes
    self.decoder_kwargs['din'] = self.capsule_dim

    # assertions for consistency of model dimensions and parameters
    assert self.decoder_dim3 == self.input_size
    assert self.image_size ** 2 == self.input_size
    assert self.mp <= 1.0 and self.mp >= 0.0 and self.mm <= 1.0 and self.mm >= 0.0
    assert self.lam >= 0.0
    assert self.reconst_weight >= 0.0
    assert self.lr >= 0.0

  def __str__(self):
    msg1 = ("l1 k %d l1 s %d l1 out %d l2 k %d l2 s %d l2 out %d l2 c %d \
    l3 dim %d dec0 %d dec1 %d dec3 %d" \
            % (self.conv1_k, self.conv1_s, self.conv1_output_channels, \
               self.capsule_conv_k, self.capsule_conv_s, self.capsule_conv_output_channels, \
               self.capsule_conv_dim, self.capsule_dim, self.decoder_dim0, self.decoder_dim1, \
               self.decoder_dim2))
    msg2 = ("mp %.2f mm %.3f lr %.2f lam %.2f rw %.2f %s" %
            (self.mp, self.mm, self.lr, self.lam,
             self.reconst_weight, msg1))
    return msg2
