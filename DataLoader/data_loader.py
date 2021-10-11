import math
import random
from pathlib import Path

import numpy as np
import cv2 as cv
import tensorflow as tf

from Base import BaseDataLoader


AUTOTUNE = tf.data.experimental.AUTOTUNE

class CatDogDataLoader(BaseDataLoader):
    def __init__(self, data_root_dir, validation_split, dataset_cache_dir, cache_clean=True, batch_size=32, shuffle=True):

        self.train_data = None
        self.val_data = None
        self.mk_data(data_root_dir, validation_split, shuffle)
        super(CatDogDataLoader, self).__init__(self.train_data, self.val_data, dataset_cache_dir, cache_clean, batch_size, validation_split)


    def mk_data(self, data_root_dir, validation_split, shuffle=True):
        data_root = Path(data_root_dir)
        all_image_paths = list(data_root.glob('*/*.jpg'))
        all_image_paths = [str(path) for path in all_image_paths]
        label_names = [item.name for item in data_root.glob('*/') if item.is_dir()]
        label_to_index = dict((name, index) for index, name in enumerate(label_names))
        all_image_labels = [label_to_index[Path(path).parent.name] for path in all_image_paths]

        samples_count = len(all_image_paths)
        paths_with_labels = list(zip(all_image_paths, all_image_labels))

        if shuffle:
            random.shuffle(paths_with_labels)
            
        val_split_idx = math.ceil(samples_count * validation_split)
        train_paths_labels = paths_with_labels[val_split_idx:]
        train_paths, train_labels = zip(*train_paths_labels)
        self.train_data = tf.data.Dataset.from_tensor_slices((list(train_paths), list(train_labels)))
        self.train_data = self.train_data.map(CatDogDataLoader.parse_func, num_parallel_calls=AUTOTUNE)

        if validation_split:
            val_paths_labels = paths_with_labels[:val_split_idx]
            val_paths, val_labels = zip(*val_paths_labels)
            self.val_data = tf.data.Dataset.from_tensor_slices((list(val_paths), list(val_labels)))
            self.val_data = self.val_data.map(CatDogDataLoader.parse_func, num_parallel_calls=AUTOTUNE)

        print("Label and index:  ", label_to_index)


    @staticmethod    
    def preprocess(path):

        path = path.numpy().decode()
        image_t = cv.imread(path)
        image_t = cv.resize(image_t, (224, 224))
        image_t = image_t - np.mean(image_t)
        image_t = image_t / 255.0

        return image_t


    @staticmethod
    def parse_func(path, label):

        [image, ] = tf.py_function(CatDogDataLoader.preprocess, [path], [tf.float32])
        image.set_shape((224, 224, 3))
        return image, label
