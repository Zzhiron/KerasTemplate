import time
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import load_model

import Base.callback as callback_module


class BaseTrainer:
    def __init__(self, train_data, val_data, model, config):
        self.trainer_config = config['trainer']
        self.epochs = self.trainer_config['epochs']

        self.train_data = train_data
        self.val_data = val_data
        self.train_steps = len(self.train_data)
        self.val_steps = len(self.val_data)

        self.model = model
        if self.trainer_config['checkpoint']:
            model = load_model(self.trainer_config['checkpoint'])  
            print("Load checkpoint learning rate: ", self.model.optimizer.lr.numpy())  
        if self.trainer_config['pretrained_weights_path']:
            self.model.load_weights(self.trainer_config['pretrained_weights_path'])

        self.proj_name = config['name']
        self.saved_dir = self.trainer_config['saved_dir']
        self.callbacks_list = []
        self.callbacks_list_from_config = self.trainer_config['callbacks']
        self.build_callbacks_list()


    def train(self):
        raise NotImplementedError


    def build_callbacks_list(self):
        for callback in self.callbacks_list_from_config:
            if callback['used']:
                callback_obj = self.init_cbks(callback_module, callback)
                self.callbacks_list += [callback_obj]


    def init_cbks(self, callback_module, callback):
        _t = time.localtime(time.time())
        time_stamp = str(_t.tm_year) + str(_t.tm_mon) + str(_t.tm_mday) + "_" + str(_t.tm_hour) + str(_t.tm_min) + str(_t.tm_sec)

        saved_dir = Path(self.saved_dir) / self.proj_name / time_stamp

        if callback['used']:
            callback_cls = getattr(callback_module, callback['type'])()

            if (callback_cls is tf.keras.callbacks.ModelCheckpoint):
                model_saved_dir = saved_dir / "models"
                if not model_saved_dir.exists():
                    model_saved_dir.mkdir(parents=True, exist_ok=True)
                weights_path =   model_saved_dir / "weights_{epoch:04d}-{val_loss:.4f}.h5"
                callback['args']['filepath'] = str(weights_path)
  
            if (callback_cls is tf.keras.callbacks.TensorBoard):
                log_saved_dir = callback['args']['log_dir']
                log_saved_dir = saved_dir / "log"
                if not log_saved_dir.exists():
                    log_saved_dir.mkdir(parents=True, exist_ok=True)
                callback['args']['log_dir'] = str(log_saved_dir)

            return callback_cls(**callback['args'])
