import argparse

import tensorflow as tf
from tensorflow.keras import optimizers

import Model.loss as loss_module
import Model.metric as metrics_module
import Model as model_module
import DataLoader as data_loader_module

from parse_config import ConfigParser
from utils import check_gpu


def main(config):
    print('Check GPU status-------------------------------------------------------')

    gpu_num = check_gpu(config['gpu']['devices'], config['gpu']['memory_limit'])

    print('Create the data loader-------------------------------------------------')

    data_loader_HANDLE = config.init_obj(
        data_loader_module, 'data_loader',
        data_root_dir= "../Dataset/CatDog/test",
        validation_split=0.0,
        cache_clean=True,
        batch_size=1,
        shuffle=False 
        )
    test_data, _ = data_loader_HANDLE.get_data()

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
        model.load_weights(config.weights_path)
        model_HANDLE.print_model_info(model)

    print('Testing the model------------------------------------------------------')
    model.evaluate(test_data, verbose=2)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='KerasTemplate')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-w', '--weights_path', default=None, type=str,
                      help='config file path (default: None)')

    config = ConfigParser.from_args(args)
    main(config)
