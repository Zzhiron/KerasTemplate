# KerasTemplate
Combining some features easy to use of Keras-Tensorflow, Pytorch, Caffe for training and testing workflow.  

# Personal views as follows:  
## Kera-Tensorflow
1. optimized well and can make training process very fast.  
2. easy to start for beginner.  
3. supported with many tools in engineering.  
## Keras
1. easy to customize and debug.  
2. ops supported flexible with numpy.   
## Caffe
1. Config.json is realy easy to use when to modified training configurations but don't want to change code.   

# KerasTemplate Project

* [KerasTemplate Project](#Kerastemplate-project)
	* [Environment recommended](#environment-recommended)
	* [Folder Structure](#folder-structure)
	* [Usage](#usage)
		* [Config file format](#config-file-format)
        * [Training](#training)
        * [Testing](#testing)
	* [Customization in two Steps](#customization-in-two-steps)
		* [Implement in code](#Implement-in-code)
		* [Modify config json file](#modify-config-json-file)
		* [Others](#Others)
            add args
            change code
	* [Reference](#contribution)

## Environment recommended
* python == 3.8.11  
* tensorflow == 2.3.0 (auto with Keras)  
* tensorboard == 2.6.0  
* pydot == 1.4.2 (for save model_info.png)  
* graphviz == 0.17 (for save model_info.png)  

## Folder Structure
  ```
  Workfolder/
    ├── Dataset/CatDog/ - default dataset dir
    |    |
    |    ├── train/
    |    |     ├── Cat/ - pictures of cat
    |    |     └── Dog/ - pictures of dog 
    |    |
    |    └── test/
    |          ├── Cat/ - pictures of cat
    |          └── Dog/ - pictures of dog 
    |    
    ├── Saved/CatDog/2021111_111111/ - default saved dir for log and model
    |    |
    |    ├── log/ - tensorboard log dir
    |    └── models/ - model will be saved here
    |    
    ├── KerasTemplate/
    |    │
    |    ├── train.py - main script to start training
    |    ├── test.py - evaluation of trained model
    |    │
    |    ├── config.json - holds configuration for training
    |    ├── parse_config.py - class to handle config file
    |    │
    |    ├── Base/ - base classes and common / customized callback functions
    |    │      ├── base_data_loader.py
    |    │      ├── base_model.py
    |    │      ├── base_trainer.py
    |    │      └── callback.py
    |    │
    |    ├── DataLoader/ - anything about data loading goes here
    |    │      └── data_loader.py
    |    │
    |    ├── Model/ - models, losses, and metrics
    |    │      ├── model.py
    |    │      ├── metric.py
    |    │      └── loss.py
    |    │
    |    ├── Trainer/ - trainers
    |    │      └── trainer.py
    |    │  
    |    └── utils/ - small utility functions
    |           ├── util.py
  ```

## Usage
### Config file format
```javascript
{
    "name": "CatDog",                   // training session name
    "gpu":{                             // GPU settings
        "devices" : "0, 1",             // GPU devices choose if you have multi gpus
        "memory_limit" : true           // GPU memory limit for run more programs  
    },
    "data_loader": {                    
        "type": "CatDogDataLoader",                                     // selecting data loader
        "args":{
            "data_root_dir": "../Dataset/CatDog/train",                 // where the dataset is 
            "validation_split": 0.2,                                    // size of validation dataset. float(portion)
            "dataset_cache_dir" : "../Dataset/CatDog/dataset_cache",    // caching dataset in binary file for accelerating data load process 
            "cache_clean" : true,                                       // clean cache files of last training experiment
            "batch_size" : 32,                                          // batch size
            "shuffle": true                                             // shuffle training data before splitting
        }
    },
    "model": {
        "type": "SimpleModel",          // name of model architecture to train
        "show": true,                   // show model information and save model architecture in model_infor.png                       
        "args": {                       // some configrations for model
            "input_h" : 224,            
            "input_w" : 224,
            "input_c" : 3,
            "num_classes" : 2
        }
    },
    "optimizer": {                      
        "type": "Adam",                 //  choose optimizer
        "args":{
            "learning_rate" : 0.001,    // set learning rate
            "amsgrad": true
        }
    },
    "loss": {
        "type": "cce",                  // choose loss
        "args":{}
    },
    "metrics": [
        "acc"                           // choose metric
    ],
    "trainer": {
        "pretrained_weights_path" : null,   // set pretrained weights path for transfer learning or fine-tune
        "checkpoint" : null,                // checkpoint path for resume training
        "epochs": 20,                       // set epochs
        "saved_dir": "../Saved",            // default saved dir for logs and models
        "callbacks":[                       // set callbacks for calling during the training process
            {                               
                "used" : true,              // use or not
                "type" : "reduce_lr",       // callback name
                "args" : {                  // params for callback function, you can go to tensorflow webpage for more detail
                    "monitor" : "loss",
                    "factor" : 0.1,         
                    "patience" : 5,
                    "verbose" : 1,
                    "mode" : "auto",
                    "min_delta" : 1e-4
                }
            },
            {
                "used" : true,
                "type" : "early_stopping",
                "args" : {
                    "monitor" : "val_loss",
                    "patience" : 6,
                    "verbose" : 1,
                    "mode" : "auto",
                    "restore_best_weights" : true
                }
            },
            {
                "used" : true,
                "type" : "model_checkpoint",
                "args" : {
                    "monitor" : "val_loss",
                    "verbose" : 1,
                    "save_best_only" : true,
                    "save_weights_only" : true
                }
            },
            {
                "used" : true,
                "type" : "tensorboard",
                "args" : {
                    "log_dir": null, 
                    "histogram_freq": 1
                }
            }
        ]
    }  
}
```

### training
`python train.py -c /path/to/config/config.json`

### testing
`python test.py -w /path/to/weights/weights.h5`

## Customization in two Steps

### Implement in code 
* 


### Modify config json file
* data loader
* model 
* loss
* metric
* trainer
* callback

### Others
Threoretically, you can modified any thing you want, but known with how I thought may help.  
* parameters, all params passed on by variable **config in code which is dictionary like

### Data Loader
* **Writing your own data loader**

1. **Inherit ```BaseDataLoader```**

    `BaseDataLoader` is a subclass of `torch.utils.data.DataLoader`, you can use either of them.

    `BaseDataLoader` handles:
    * Generating next batch
    * Data shuffling
    * Generating validation data loader by calling
    `BaseDataLoader.split_validation()`

* **DataLoader Usage**

  `BaseDataLoader` is an iterator, to iterate through batches:
  ```python
  for batch_idx, (x_batch, y_batch) in data_loader:
      pass
  ```
* **Example**

  Please refer to `data_loader/data_loaders.py` for an MNIST data loading example.

### Trainer
* **Writing your own trainer**

1. **Inherit ```BaseTrainer```**

    `BaseTrainer` handles:
    * Training process logging
    * Checkpoint saving
    * Checkpoint resuming
    * Reconfigurable performance monitoring for saving current best model, and early stop training.
      * If config `monitor` is set to `max val_accuracy`, which means then the trainer will save a checkpoint `model_best.pth` when `validation accuracy` of epoch replaces current `maximum`.
      * If config `early_stop` is set, training will be automatically terminated when model performance does not improve for given number of epochs. This feature can be turned off by passing 0 to the `early_stop` option, or just deleting the line of config.

2. **Implementing abstract methods**

    You need to implement `_train_epoch()` for your training process, if you need validation then you can implement `_valid_epoch()` as in `trainer/trainer.py`

* **Example**

  Please refer to `trainer/trainer.py` for MNIST training.

* **Iteration-based training**

  `Trainer.__init__` takes an optional argument, `len_epoch` which controls number of batches(steps) in each epoch.

### Model
* **Writing your own model**

1. **Inherit `BaseModel`**

    `BaseModel` handles:
    * Inherited from `torch.nn.Module`
    * `__str__`: Modify native `print` function to prints the number of trainable parameters.

2. **Implementing abstract methods**

    Implement the foward pass method `forward()`

* **Example**

  Please refer to `model/model.py` for a LeNet example.

### Loss
Custom loss functions can be implemented in 'model/loss.py'. Use them by changing the name given in "loss" in config file, to corresponding name.

### Metrics
Metric functions are located in 'model/metric.py'.

You can monitor multiple metrics by providing a list in the configuration file, e.g.:
  ```json
  "metrics": ["accuracy", "top_k_acc"],
  ```


### Validation data
To split validation data from a data loader, call `BaseDataLoader.split_validation()`, then it will return a data loader for validation of size specified in your config file.
The `validation_split` can be a ratio of validation set per total data(0.0 <= float < 1.0), or the number of samples (0 <= int < `n_total_samples`).

### Tensorboard Visualization
This template supports Tensorboard visualization by using either  `torch.utils.tensorboard` or [TensorboardX](https://github.com/lanpa/tensorboardX).

1. **Install**

    If you are using Keras 1.1 or higher, install tensorboard by 'pip install tensorboard>=1.14.0'.

    Otherwise, you should install tensorboardx. Follow installation guide in [TensorboardX](https://github.com/lanpa/tensorboardX).

2. **Run training** 

    Make sure that `tensorboard` option in the config file is turned on.

    ```
     "tensorboard" : true
    ```

3. **Open Tensorboard server** 

    Type `tensorboard --logdir saved/log/` at the project root, then server will open at `http://localhost:6006`


<!-- ## TODOs

- [ ] Multiple optimizers
- [ ] Support more tensorboard functions
- [x] Using fixed random seed
- [x] Support Keras native tensorboard
- [x] `tensorboardX` logger support
- [x] Configurable logging layout, checkpoint naming
- [x] Iteration-based training (instead of epoch-based)
- [x] Adding command line option for fine-tuning -->

# Reference
* [KerasTemplate](https://github.com/victoresque/Keras-template)  
* [KerasTemplate](https://github.com/Ahmkel/Keras-Project-Template)  
* [Tensorflow && Keras](https://tensorflow.google.cn/)  