import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D,\
     Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add, Dropout, Input
from tensorflow.keras import activations

class IdentityBlock(tf.keras.layers.Layer):

  def __init__(self, units=32, units2 = 64, block_name=None):
    super().__init__()
    self.units = units
    self.units2 = units2
    self.block_name = block_name

  def build(self, input_shape):
    self.conv1 = Conv2D(self.units, kernel_size=(1,1),strides=(1,1), padding='valid', name=f'{self.block_name}_conv1')
    self.conv2 = Conv2D(self.units, kernel_size=(3,3), strides=(1,1), padding='same', name=f'{self.block_name}_conv2')
    self.conv3 = Conv2D(self.units, kernel_size=(1,1), strides=(1,1), padding='valid', name=f'{self.block_name}_conv3')
    self.relu = Activation(activations.relu)
    self.add = Add()
    self.batch_norm1 = BatchNormalization(name = "ID_bn")
    self.batch_norm2 = BatchNormalization(name = "ID_bn2")
    self.batch_norm3 = BatchNormalization(name = "ID_bn3")

  def call(self, inputs):
    x_skip = inputs
    x = self.conv1(inputs)
    x = self.batch_norm1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.batch_norm2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.batch_norm3(x)    
    x = self.add(([x, x_skip]))
    x = self.relu(x)

    return x

class ConvBlock(tf.keras.layers.Layer):

  def __init__(self, units, units2, s, block_name):
    super().__init__()
    self.s = s
    self.units = units
    self.units2 = units2
    self.block_name = block_name

  def build(self, input_shape):
    self.conv1 = Conv2D(self.units, kernel_size=(1,1), strides=(self.s, self.s), padding='valid', name=f'{self.block_name}_conv1')
    self.conv2 = Conv2D(self.units, kernel_size=(3,3), strides=(1,1), padding='same', name=f'{self.block_name}_conv2')
    self.conv3 = Conv2D(self.units, kernel_size = (1,1), strides=(1,1), padding='valid', name=f'{self.block_name}_conv3')
    self.conv_shortcut = Conv2D(self.units, kernel_size=(1,1), strides=(self.s, self.s), padding='valid', name=f'{self.block_name}_conv_sc')
    self.relu = Activation(activations.relu, name="CONV_relu")
    self.add = Add()
    self.batch_norm1 = BatchNormalization(name="CONV_bn")
    self.batch_norm2 = BatchNormalization(name="CONV_bn2")
    self.batch_norm3 = BatchNormalization(name="CONV_bn3")
    self.batch_norm4 = BatchNormalization(name="CONV_bn4")

  def call(self, inputs):
    x_skip = inputs
    x = self.conv1(inputs)
    x = self.batch_norm1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.batch_norm2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.batch_norm3(x)    

    x_skip = self.conv_shortcut(x_skip)    
    x = self.add([x, x_skip])
    x = self.relu(x)
    return x