import argparse

import tensorflow as tf
from tensorflow.keras import optimizers

import Model.loss as loss_module
import Model.metrics as metrics_module
import Model as model_module
import DataLoader as data_loader_module
from Trainer import Trainer 

from parse_config import ConfigParser
from utils import check_gpu


def main(config):
    print('Check GPU status-------------------------------------------------------')

    gpu_num = check_gpu(config['gpu']['devices'], config['gpu']['memory_limit'])

    print('Create the data loader-------------------------------------------------')

    data_loader_HANDLE = config.init_obj(data_loader_module, 'data_loader')
    train_data, val_data = data_loader_HANDLE.get_data()

    print('Initialize optimizer, loss, metrics------------------------------------')

    optimizer = config.init_obj(optimizers, 'optimizer')
    loss = config.init_ftn(loss_module, 'loss')
    metric_list = [getattr(metrics_module, met) for met in config['metrics']]

    print('Create the model-------------------------------------------------------')

    devices = ["/gpu:" + str(i) for i in range(gpu_num)]
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=devices)
    with mirrored_strategy.scope():
        model_HANDLE = config.init_obj(model_module, 'model')
        model = model_HANDLE.build_model(loss, metric_list, optimizer)
        model_HANDLE.print_model_info(model)

    print('Create the trainer and Start training the model------------------------')

    trainer = Trainer(
        train_data=train_data,
        val_data=val_data,
        model=model,
        config=config
        )
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='KerasTemplate')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')

    config = ConfigParser.from_args(args)
    main(config)
