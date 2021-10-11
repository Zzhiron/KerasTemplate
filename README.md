# KerasTemplate
Combining some features easy to use of Keras-Tensorflow, Pytorch, Caffe for training and testing workflow.  

# Personal views as follows:  
## Kera-Tensorflow
1. optimized well and can make training process very fast.  
2. easy to start for beginner.  
3. supported with many tools in engineering.  
## Pytorch
1. easy to customize and debug.  
2. ops supported flexible with numpy.   
## Caffe
1. Config.json is realy easy to use when to modified training configurations but don't want to change code.   

# KerasTemplate Project

* [KerasTemplate Project](#Kerastemplate-project)
	* [Environment recommended](#environment-recommended)
	* [Folder Structure](#folder-structure)
	* [Usage](#usage)
		* [Check config file](#check-config-file)
        	* [Training](#training)
        	* [Testing](#testing)
	* [Customization in two Steps](#customization-in-two-steps)
		* [Implement in code](#implement-in-code)
		* [Modify config json file](#modify-config-json-file)
		* [Others](#others)
            		add args
            		change code
	* [Reference](#reference)

## Environment recommended
* python == 3.8.11  
* tensorflow == 2.3.0 (auto with Keras)  	(pip install tensorflow==2.3)
* tensorboard == 2.6.0  		 	(pip install tensorboard)
* pydot == 1.4.2 (for save model_info.png)  	(pip install pydot)
* graphviz == 0.17 (for save model_info.png)  	(pip install graphviz)

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
    |           └── util.py
  ```

## Usage
### Check config file
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
* **Data Loader**
* 1.inherit from ```BaseDataLoader```
* 2.handle shuffling and validation split

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

### Trainer
* **Writing your own trainer**
1. **Inherit ```BaseTrainer```**
2. **Implementing abstract methods**

### Model
* **Writing your own model**
1. **Inherit `BaseModel`**
2. **Implementing abstract methods**

### Loss
Custom loss functions can be implemented in 'model/loss.py'. Use them by changing the name given in "loss" in config file, to corresponding name.

### Metrics

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
