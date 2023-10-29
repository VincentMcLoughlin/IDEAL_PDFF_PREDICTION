from DataManager import DataManager
from models.Resnet50 import Resnet50
from models.SmallResnet import SmallResnet
from utils.RSquared import RSquared
import tensorflow as tf
import numpy as np
from pathlib import Path
#from tensorflow.keras.metrics import R2Score #Unfortunately not in our version of tf

#input shape is (232, 256, 36)

class diff(tf.keras.metrics.Metric):
    def __init__(self, name = 'diff', **kwargs):
        super(diff, self).__init__(**kwargs)
        self.tp = self.add_weight('tp', initializer = 'zeros')
        self.fp = self.add_weight('fp', initializer = 'zeros')

    def update_state(self, y_true, y_pred,sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.math.greater(y_pred, 0.5)      
        true_p = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        false_p = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        self.tp.assign_add(tf.reduce_sum(tf.cast(true_p, self.dtype)))
        self.fp.assign_add(tf.reduce_sum(tf.cast(false_p, self.dtype)))
    
    def reset_state(self):
        self.tp.assign(0)
        self.fp.assign(0)

    def result(self):
        return self.tp - self.fp

def setup_logger(model_name, dataset_name, batch_size, n, other_names):

    folder_name = f"logging/{model_name}/" 
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    filename = f"{folder_name}bs={batch_size}_dataset={dataset_name}_firstNeurons={n}_{other_names}.csv"
    csv_logger = tf.keras.callbacks.CSVLogger(filename, separator=',', append=False)

    return csv_logger

def main():

    config_path = "config.yaml"    
    data_manager = DataManager(config_path)    
    train_dataset, test_dataset = data_manager.build_datasets()    
    
    config = data_manager.get_config()
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    
    starting_neurons = 32
    #resnet_model = Resnet50(starting_neurons, num_classes=1)
    resnet_model = SmallResnet(starting_neurons, num_classes=1)
    
    model_name = "small_resnet"
    dataset_name = "full_ideal"
    other_names = ""
    csv_logger = setup_logger(model_name, dataset_name, batch_size, starting_neurons, other_names)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    num_train = data_manager.get_num_train()
    num_test = data_manager.get_num_test()

    steps_per_epoch = int(np.ceil(num_train / batch_size))
    val_steps_per_epoch = int(np.ceil(num_test / batch_size))

    resnet_model.compile(optimizer=optimizer,              
              loss = 'mean_absolute_error',
              metrics=['mean_absolute_error','mean_absolute_percentage_error', RSquared(name='r2_score')])
    
    resnet_model.build(input_shape = (batch_size, 232, 256, 36))
    
    history = resnet_model.fit(train_dataset.repeat(), validation_data = test_dataset.repeat(), batch_size=batch_size, epochs=epochs, 
                               steps_per_epoch=steps_per_epoch, validation_steps=val_steps_per_epoch, callbacks=(csv_logger))

if __name__ == "__main__":
    main()