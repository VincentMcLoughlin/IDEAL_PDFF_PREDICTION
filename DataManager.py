import yaml
import pandas as pd 
import tensorflow as tf
import numpy as np

MAX_PIXEL_VALUE = 4095
def read_numpy_file(data_path, label):
    img_tensor = np.load(data_path)/MAX_PIXEL_VALUE
    return img_tensor, label

class DataManager:

    def __init__(self, config_path, pdff_scores = None):
        self.config_dict = self._read_yaml(config_path)
        self.ref_df = self._read_csv(self.config_dict["reference_csv_path"])
        self.data_paths = self.ref_df["data_path"]
        self.max_pixel_value = self.config_dict["max_pixel_value"]

        if not pdff_scores:
            self.pdff_scores = self.ref_df["PDFF"]
        else:
            self.pdff_scores = pdff_scores

    def _read_yaml(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _read_csv(self, path):
        return pd.read_csv(path)
        
    def build_dataset(self, data_paths, labels):
        dataset = tf.data.Dataset.from_tensor_slices((data_paths,labels))
        dataset = dataset.map(lambda x, y: tf.numpy_function(read_numpy_file, [x,y], [tf.dtypes.float32, tf.dtypes.float32])).prefetch(tf.data.AUTOTUNE).batch(batch_size)
        return dataset

