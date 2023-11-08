from DataManager import DataManager
from models.Resnet50 import Resnet50
from models.PretrainedModel import PretrainedModel
from utils.RSquared import RSquared
import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
#from tensorflow.keras.metrics import R2Score Unfortunately not in our version of tf

#input shape is (232, 256, 36), which we crop to 224x224x36
MAX_PIXEL_VALUE = 375 # Max pixel value across entire dark dataset
IMG_SHAPE = (224, 224, 36)

def get_multichannel_weights(weights, num_channels):
  avg_weight = np.mean(weights, axis=-2)  
  avg_weights = avg_weight.reshape(weights[:,:,-1:,:].shape)

  multi_channel_weights = np.tile(avg_weights, (num_channels, 1))
  return multi_channel_weights

def adjust_pretrained_input(base_model, first_conv_idx, num_channels, new_input_shape):
    base_config = base_model.get_config()
    base_config["layers"][0]["config"]["batch_input_shape"] = new_input_shape
    new_model = Model.from_config(base_config)

    new_model_layer_names = []
    new_model_config = new_model.get_config()

    for i in range(len(new_model_config['layers'])):
        new_model_layer_names.append(new_model_config['layers'][i]['name'])

    first_conv_name = new_model_layer_names[first_conv_idx]
    print(first_conv_name)

    for layer in base_model.layers:

        if layer.name in new_model_layer_names:

            if layer.get_weights() != []:
                target_layer = new_model.get_layer(layer.name)

                if layer.name in first_conv_name:
                    weights = layer.get_weights()[0]
                    biases = layer.get_weights()[1]

                    weights_extra_channels = get_multichannel_weights(weights, num_channels)
                    print(weights_extra_channels.shape)
                    target_layer.set_weights([weights_extra_channels, biases])
                    target_layer.trainable = False

                else:
                    target_layer.set_weights(layer.get_weights())
                    target_layer.trainable = False

    return new_model

def setup_logger(model_name, dataset_name, batch_size, n, other_names):

    folder_name = f"logging/{model_name}/" 
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    filename = f"{folder_name}bs={batch_size}_dataset={dataset_name}_firstNeurons={n}_{other_names}.csv"
    csv_logger = tf.keras.callbacks.CSVLogger(filename, separator=',', append=False)

    return csv_logger

def res_scheduler(epoch): #Original
    if epoch < 40:
        return 0.0001
    if epoch < 65:
        return 0.00001
    if epoch < 100:
        return 0.000001
    return 0.0008

def main():
    
    config_path = "pretrained_config.yaml"    
    data_manager = DataManager(config_path)
    train_dataset, test_dataset = data_manager.build_datasets()    
    
    config = data_manager.get_config()
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
    
    pretrained_model = PretrainedModel(base_model, max_value = MAX_PIXEL_VALUE, crop_height=224, crop_width=224, augment_config=config["augmentation"])
    
    model_name = "mobilenet_pretrained"
    dataset_name = "full_ideal"
    other_names = "transfer_learn"
    csv_logger = setup_logger(model_name, dataset_name, batch_size, "pretrained", other_names)
    base_learning_rate = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate)
    lrate_res = LearningRateScheduler(res_scheduler)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

    num_train = data_manager.get_num_train()
    num_test = data_manager.get_num_test()

    steps_per_epoch = int(np.ceil(num_train / batch_size))
    val_steps_per_epoch = int(np.ceil(num_test / batch_size))
    
    #train_dataset, test_dataset = apply_augmentation(train_dataset, test_dataset)
    
    pretrained_model.compile(optimizer=optimizer,              
              loss = 'mean_absolute_error',
              metrics=['mean_absolute_error','mean_absolute_percentage_error', RSquared(name='r2_score')])
    
    pretrained_model.build(input_shape = (batch_size, 232, 256, 36))
    
    history = pretrained_model.fit(train_dataset.repeat(), validation_data = test_dataset.repeat(), batch_size=batch_size, epochs=epochs, 
                               steps_per_epoch=steps_per_epoch, validation_steps=val_steps_per_epoch, callbacks=(lrate_res, csv_logger, early_stop))        
    
    #### Fine tuning
    pretrained_model.base_model.trainable = True
    other_names = "fine_tune"
    csv_logger = setup_logger(model_name, dataset_name, batch_size, "pretrained", other_names)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

    #Switch layers to trainable
    num_tuneable_layers = 100

    for layer in pretrained_model.base_model.layers[:num_tuneable_layers]:
        layer.trainable = False    

    pretrained_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0, name='accuracy')])
    
    tuning_epochs = 10

    total_epochs = epochs + tuning_epochs

    history_fine = pretrained_model.fit(train_dataset.repeat(),
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=test_dataset.repeat(),
                         callbacks = [early_stop])    

if __name__ == "__main__":
    main()    