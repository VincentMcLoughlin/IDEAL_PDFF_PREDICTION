from DataManager import DataManager
from models.Resnet50 import Resnet50
import tensorflow as tf
import numpy as np
#input shape is (232, 256, 36)
def main():

    config_path = "config.yaml"    
    data_manager = DataManager(config_path)    
    train_dataset, test_dataset = data_manager.build_datasets()    
    
    config = data_manager.get_config()
    batch_size = config["batch_size"]    
        
    resnet_model = Resnet50(64, num_classes=1)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    num_train = data_manager.get_num_train()

    steps_per_epoch = int(np.ceil(num_train / batch_size))
    resnet_model.compile(optimizer=optimizer,
              #loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              loss = 'mean_absolute_error',
              metrics=['mean_absolute_error','mean_absolute_percentage_error','accuracy'])
    
    resnet_model.build(input_shape = (batch_size, 232, 256, 36))
    
    history = resnet_model.fit(train_dataset, validation_data = test_dataset, batch_size=batch_size, epochs=2, steps_per_epoch=steps_per_epoch)

if __name__ == "__main__":
    main()