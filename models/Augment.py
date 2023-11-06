import tensorflow as tf 

class Augment(tf.keras.layers.Layer):
  def __init__(self, augment_config):
    super().__init__()
    self.aug_config = augment_config

  def build(self, input_shape):

    self.augment_layers = list()
    if "flip" in self.aug_config:
        self.augment_layers.append(tf.keras.layers.RandomFlip(self.aug_config["flip"]))
    
    if "rotate" in self.aug_config:
        self.augment_layers.append(tf.keras.layers.RandomRotation(self.aug_config["rotate"]))
    
    if "zoom_height" in self.aug_config and "zoom_width" in self.aug_config:
       self.augment_layers.append(tf.keras.layers.RandomZoom(self.aug_config["zoom_height"], self.aug_config["zoom_width"]))

    if "translate_height" in self.aug_config and "translate_width" in self.aug_config:
        self.augment_layers.append(tf.keras.layers.RandomTranslation(self.aug_config["translate_height"], self.aug_config["translate_width"]))

  def call(self, inputs):
    x = inputs
    
    for aug_layer in self.augment_layers:
       x = aug_layer(x)

    return x