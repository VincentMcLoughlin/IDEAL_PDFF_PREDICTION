import tensorflow as tf
from models.Augment import Augment

class PretrainedModel(tf.keras.Model):

  def __init__(self, base_model, max_value, crop_height, crop_width, augment_config): #Will need to crop to model size
    super().__init__()
    half_max = max_value / 2.0
    #self.crop = tf.keras.layers.CenterCrop(height = crop_height, width = crop_width)
    self.preprocess_augment = Augment(augment_config)
    self.base_model = base_model
    self.rescale = tf.keras.layers.Rescaling(1./half_max, offset = -1)
    self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()    
    #self.dense_1 = tf.keras.layers.Dense(100)
    #self.dense_2 = tf.keras.layers.Dense(100)
    self.prediction_layer = tf.keras.layers.Dense(1)


  def call(self, inputs):    
    x = inputs
    x = self.preprocess_augment(x)    
    x = self.rescale(x)
    x = self.base_model(x, training=False)
    x = self.global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    #x = self.dense_1(x)
    #x = self.dense_2(x)
    x = self.prediction_layer(x)
    return x

