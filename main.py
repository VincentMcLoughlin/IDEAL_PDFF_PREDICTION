from DataManager import DataManager
from models.Resnet50 import Resnet50
from models.SmallResnet import SmallResnet
import tensorflow as tf
import numpy as np
#input shape is (232, 256, 36)
def main():

    config_path = "config.yaml"    
    data_manager = DataManager(config_path)    
    train_dataset, test_dataset = data_manager.build_datasets()    
    
    config = data_manager.get_config()
    batch_size = config["batch_size"]
    epochs = config["epochs"]
        
    #resnet_model = Resnet50(32, num_classes=1)
    resnet_model = SmallResnet(64, num_classes=1)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    num_train = data_manager.get_num_train()
    num_test = data_manager.get_num_test()

    steps_per_epoch = int(np.ceil(num_train / batch_size))
    val_steps_per_epoch = int(np.ceil(num_train / batch_size))

    resnet_model.compile(optimizer=optimizer,
              #loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              loss = 'mean_absolute_error',
              metrics=['mean_absolute_error','mean_absolute_percentage_error','accuracy'])
    
    resnet_model.build(input_shape = (batch_size, 232, 256, 36))
    
    history = resnet_model.fit(train_dataset.repeat(), validation_data = test_dataset.repeat(), batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=val_steps_per_epoch)

if __name__ == "__main__":
    main()