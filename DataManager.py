import yaml
import pandas as pd 
import tensorflow as tf
import numpy as np

MAX_PIXEL_VALUE = 4095

def read_numpy_file(data_path, label):
    img_tensor = np.load(data_path)/MAX_PIXEL_VALUE
    return img_tensor, label

class DataManager:

    def __init__(self, config_path):
        self.config_dict = self._read_yaml(config_path)
        self.ref_df = self._read_csv(self.config_dict["reference_csv_path"])                
        self.data_column_name = self.config_dict["data_column_name"]
        self.score_column_name = self.config_dict["score_column_name"]
        self.train_test_frac = self.config_dict["train_test_split_fraction"]
        self.validation_frac = self.config_dict["validate_fraction"]
        self.batch_size = self.config_dict["batch_size"]

    def _read_yaml(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _read_csv(self, path):
        return pd.read_csv(path)
        
    def _make_train_test_split(self):
        mask = np.random.rand(self.ref_df.shape[0]) < self.train_test_frac
        train_df = self.ref_df[mask]
        test_df = self.ref_df[~mask]        
                
        # val_fraction_of_test = self.validation_frac / (1 - self.train_test_frac)
        # val_mask = np.random.rand(test_df.shape[0]) < val_fraction_of_test
        # val_df = self.ref_df[val_mask]
        # test_df = self.ref_df[~val_mask]

        return train_df, test_df #, val_df

    def _build_dataset(self, data_info_df, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((data_info_df[self.data_column_name].tolist(),data_info_df[self.score_column_name].tolist()))
        dataset = dataset.map(lambda x, y: tf.numpy_function(read_numpy_file, [x,y], [tf.dtypes.float32, tf.dtypes.float32])).prefetch(tf.data.AUTOTUNE).batch(batch_size)
        it = dataset.__iter__() #See https://stackoverflow.com/questions/58933151/as-list-is-not-defined-on-an-unknown-tensorshape
        return it
        #return dataset
    
    def build_datasets(self):
                
        #self.train_df, self.test_df, self.val_df = self._make_train_test_split()        
        self.train_df, self.test_df = self._make_train_test_split()        
        
        train_dataset = self._build_dataset(self.train_df, self.batch_size)
        test_dataset = self._build_dataset(self.test_df, self.batch_size)
        #val_dataset = self._build_dataset(self.val_df, batch_size)

        return train_dataset, test_dataset #, val_dataset

    def get_config(self):
        return self.config_dict
