import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, ZeroPadding2D     
from tensorflow.keras import activations
from models.Augment import Augment
from models.model_components.ResnetBlocks import ConvBlock, IdentityBlock

class Resnet50(tf.keras.Model):

  def __init__(self, f1, max_pixel_value, num_classes=10, add_augment=False, aug_config = None, add_rescale = True, **kwargs):
    super(Resnet50, self).__init__()

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
    self.identity_block_1c = IdentityBlock(f1, f1*scaling, block_name='id1c')

    f2 = f1*2
    self.convblock_2 = ConvBlock(f2, f2*scaling, s=2, block_name='conv2')
    self.identity_block_2a = IdentityBlock(f2, f2*scaling, block_name='id2a')
    self.identity_block_2b = IdentityBlock(f2, f2*scaling, block_name='id2b')
    self.identity_block_2c = IdentityBlock(f2, f2*scaling, block_name='id2c')

    f3 = f2*2
    self.convblock_3 = ConvBlock(f3, f3*scaling, s=2, block_name='conv3')
    self.identity_block_3a = IdentityBlock(f3, f3*scaling, block_name='id3a')
    self.identity_block_3b = IdentityBlock(f3, f3*scaling, block_name='id3b')
    self.identity_block_3c = IdentityBlock(f3, f3*scaling, block_name='id3c')

    f4 = f3*2
    self.convblock_4 = ConvBlock(f4, f4*scaling, s=2, block_name='conv4')
    self.identity_block_4a = IdentityBlock(f4, f4*scaling, block_name='id4a')
    self.identity_block_4b = IdentityBlock(f4, f4*scaling, block_name='id4b')
    self.identity_block_4c = IdentityBlock(f4, f4*scaling, block_name='id4c')

    self.avg_pooling = layers.GlobalAveragePooling2D()
    

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

    x = self.convblock_3(x)
    x = self.identity_block_3a(x)
    x = self.identity_block_3b(x)

    x = self.convblock_4(x)
    x = self.identity_block_4a(x)
    x = self.identity_block_4b(x)
    x = self.avg_pooling(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = self.output_layer(x)

    return x