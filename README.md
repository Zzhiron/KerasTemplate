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
1. Config.json is really easy to use when to modify training configurations but don't want to change code.   

# KerasTemplate Project

* [KerasTemplate Project](#Kerastemplate-project)
	* [Environment recommended](#environment-recommended)
	* [Folder Structure](#folder-structure)
	* [Usage](#usage)
		* [Check config file](#check-config-file)
		* [Training](#training)
		* [Testing](#testing)
		* [Tensorboard](#tensorboard)
	* [Customization](#customization) **(Within two steps)**
		* [Data Loader](#data-loader)
		* [Model](#model)
		* [Loss](#loss)
		* [Metric](#metric)
		* [Trainer](#trainer)
		* [Callback](#callback)
		* [Others](#others)
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

### Training
Run script `python train.py -c /path/to/config/config.json`

### Testing
Run script `python test.py -w /path/to/weights/weights.h5`

### Tensorboard
When finish training, run script `tensorboard --logdir /path/to/Saved/CatDog/log`, then you can open the link [http://localhost:6006/](http://localhost:6006/) in your browser to see the log  

## Customization
**Customization in two Steps**  
```0. Check examples I offered first if you get in trouble```  
```1. Implement in code```  
```2. Modify config json file```  

### Data Loader
* 0. ```BaseDataLoader``` handle the ```tf.data.Datasets```, batch size, cache and prefetch
* 1. inherit ```BaseDataLoader```, handle shuffling and validation split yourself
* 2. modify ```type``` and ```args``` in the part of data_loader in config.json

### Model
* 0. ```BaseModel``` handle the ```build_model```, that means compile model with loss, metric, and optimizer 
* 1. inherit ```BaseModel```, which is subclass of ```tf.keras.models.Model```, implement the architecture of model yourself
* 2. modify ```type``` and ```args``` in the part of model in config.json

### Loss
* 1. implement the loss function yourself with inputs as ```(y_true, y_pred)```, or you can use the loss functions keras provided
* 2. modify ```type``` and ```args``` in the part of loss in config.json

### Metric
* 1. implement the metric function yourself with inputs as ```(y_true, y_pred)```, or you can use the metric functions keras provided
* 2. add the function name in the metric list in the part of metric in config.json

### Trainer
* 0. ```BaseTrainer``` handle the ```traning process and callback```, that means you can do something you want during the training process 
* 1. inherit ```BaseTrainer```, implement the logic you want during the training process, you can also implement the training process from scratch without using ```fit()``` method keras provided
* 2. modify ```type``` and ```args``` in the part of trainer in config.json

### Callback
* 1. implement the callback function yourself inherit ```tf.keras.callbacks.Callback```, return its name to function like I do
* 2. add your callback into callback list in config.json as follows, then modify ```type``` and ```args``` in the part of your callback in config.json
```javascript
"callbacks":[                   
            	{                               
			"used" : true,              
			"type" : "reduce_lr",       
			"args" : {                  
			    "monitor" : "loss",
			    "factor" : 0.1,         
			    "patience" : 5,
			    "verbose" : 1,
			    "mode" : "auto",
			    "min_delta" : 1e-4
			    }
            	},
------------------add your callback as follows-------------------
		{
			"used" : true,
			"type" : "your_callback_function_name",
			"args" : {}
		}
-----------------------------------------------------------------
]
```
### Others
Threoretically, you can modified any thing you want, but known with how I thought may help    
* parameters, all params passed on by variable ```config``` in code which is dictionary like  
* every sub module in ```config.json``` are dict like with ```type, args``` attribute, the former is module name,  the latter will be params passed the to module in code when use ```config.init_obj()``` method to initialize the module instance  

## TODO
- [ ] Customized Optimizers  
- [ ] Support more tensorboard functions  

# Reference
* [PytorchTemplate](https://github.com/victoresque/Keras-template)  
* [KerasTemplate](https://github.com/Ahmkel/Keras-Project-Template)  
* [Tensorflow && Keras](https://tensorflow.google.cn/)  
