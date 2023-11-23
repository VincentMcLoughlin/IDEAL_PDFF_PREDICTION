import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, ZeroPadding2D
from models.Augment import Augment
from models.model_components.ResnetBlocks import ConvBlock, IdentityBlock

class SmallResnet(tf.keras.Model):
  
  def __init__(self, f1, max_pixel_value, num_classes=10, add_augment=False, aug_config = None, add_rescale = True, **kwargs):
    super(SmallResnet, self).__init__()
    
    self.add_augmentation = add_augment
    self.aug_config = aug_config
    
    if self.add_augmentation:      
      self.augment = Augment(self.aug_config)

    half_max = max_pixel_value / 2.0
    self.add_rescaling = add_rescale
    self.rescale = layers.Rescaling(1./half_max, offset = -1)

    self.zero_padding = ZeroPadding2D(padding=(3,3))
    self.initial_conv = Conv2D(f1, kernel_size=(7,7), strides=(2,2))
    scaling = 2
    self.convblock_1 = ConvBlock(f1, f1*scaling, s=1, block_name='conv1')
    self.identity_block_1a = IdentityBlock(f1, f1*scaling, block_name='id1a')
    self.identity_block_1b = IdentityBlock(f1, f1*scaling, block_name='id1b')

    f2 = f1*2
    self.convblock_2 = ConvBlock(f2, f2*scaling, s=2, block_name='conv2')
    self.identity_block_2a = IdentityBlock(f2, f2*scaling, block_name='id2a')
    self.identity_block_2b = IdentityBlock(f2, f2*scaling, block_name='id2b')        

    self.avg_pooling = layers.GlobalAveragePooling2D()
    self.flatten = layers.Flatten()

    if num_classes > 1:
      self.output_layer = layers.Dense(num_classes, name='output_layer', activation="softmax")
    else:
      self.output_layer = layers.Dense(num_classes, name='output_layer')

  def call(self, inputs):

    x = inputs

    if self.add_augmentation:
      x = self.augment(x)

    if self.add_rescaling:
      x = self.rescale(x)

    x = self.zero_padding(inputs)
    x = self.initial_conv(x)

    x = self.convblock_1(x)
    x = self.identity_block_1a(x)
    x = self.identity_block_1b(x)

    x = self.convblock_2(x)
    x = self.identity_block_2a(x)
    x = self.identity_block_2b(x)

    x = self.avg_pooling(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    #x = self.flatten(x)
    #x = self.dense1(x)
    #x = self.dense2(x)
    x = self.output_layer(x)

    return x