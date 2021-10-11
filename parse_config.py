import argparse

import os
from pathlib import Path
from utils import read_json


class ConfigParser:
    def __init__(self, config, weights_path=None):
        self.config = config
        self._weights_path = weights_path

    @classmethod
    def from_args(cls, args):
        
        args = args.parse_args()
        cfg_fname = Path(args.config)
        config = read_json(cfg_fname)

        weights_path = None
        if hasattr(args, 'weights_path') and args.weights_path:
            print(args.weights_path)
            weights_path = args.weights_path

        return cls(config, weights_path)

    def init_obj(self, module, name, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        module_args.update(kwargs)

        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, module, name):
        module_name = self[name]['type']
        return getattr(module, module_name)


    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]


    @property
    def weights_path(self):
        return self._weights_path


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    args = argparse.ArgumentParser(description='KerasTemplate')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')

    config = ConfigParser.from_args(args)
