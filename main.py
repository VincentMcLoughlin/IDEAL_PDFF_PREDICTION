from DataManager import DataManager
from models.Resnet50 import Resnet50
import tensorflow as tf

#input shape is (232, 256, 36)
def main():

    config_path = "config.yaml"    
    data_manager = DataManager(config_path)    
    train_dataset, test_dataset = data_manager.build_datasets()    
    
    config = data_manager.get_config()
    batch_size = config["batch_size"]

    # for element in train_dataset:
    #     print(element)
        
    resnet_model = Resnet50(64, num_classes=1)

    resnet_batch_size = 128
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    resnet_model.compile(optimizer=optimizer,
              #loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              loss = 'mean_absolute_error',
              metrics=['mean_absolute_error','accuracy'])
    
    resnet_model.build(input_shape = (batch_size, 232, 256, 36))
    
    history = resnet_model.fit(train_dataset, validation_data = test_dataset, batch_size=batch_size, epochs=2)

if __name__ == "__main__":
    main()