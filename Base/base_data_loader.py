from pathlib import Path
import tensorflow as tf


AUTOTUNE = tf.data.experimental.AUTOTUNE

class BaseDataLoader:
    def __init__(self, train_data, val_data, dataset_cache_dir, cache_clean, batch_size, validation_split):
        self.validation_split = validation_split
        self.train_data = train_data
        self.val_data = val_data
        print("(Train / Test) Samples nums: ", len(train_data))
        self.build_data_pipeline(dataset_cache_dir, cache_clean, batch_size)
        

    def get_data(self):
        return self.train_data, self.val_data


    def build_data_pipeline(self, dataset_cache_dir, cache_clean, batch_size):
        dataset_cache_dir = Path(dataset_cache_dir)

        if not dataset_cache_dir.exists():
            dataset_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            if cache_clean:
                for file_path in dataset_cache_dir.glob("*"):
                    Path.unlink(file_path)

        self.train_data = self.train_data\
            .batch(batch_size)\
            .cache(str(dataset_cache_dir / "train"))\
            .prefetch(buffer_size=AUTOTUNE)

        if self.validation_split:
            self.val_data = self.val_data\
                .batch(batch_size)\
                .cache(str(dataset_cache_dir / "val"))\
                .prefetch(buffer_size=AUTOTUNE)
