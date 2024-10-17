# DARE
The code for  paper "Uncovering Hidden Correlations: Post-Training  Data Reconstruction Attacks against Vertical  Federated Learning"

## About The Project
DARE allows a vertical federated client to reconstruct the corresponding complete data by leveraging his or her own incomplete data.

## Getting Started
### Prerequisites
 requires the following packages: 
- Python 3.9.18
- Pytorch 1.10.2+cu102
- Sklearn 1.3.2
- Numpy 1.23.5
- Scipy 1.11.1


### File Structure 
```
DARE
├── data
│   ├── bank
│   ├── cifar10
│   ├── cifar100
│   └── TinyImageNet
├── models
│   ├── DRModel.py
│   ├── MAEModel.py
│   └── VFLModel.py
├── results
│   ├── MAE_saved_models
│   |   └── MAE_official_pretrained_models
│   ├── Recovery_training_saved_models
│   └── VFL_training_saved_models
├── params.py
├── utils.py
├── VFL_training.py
├── MAE_finetune.py
├── Data_recovery.py
└── test.py
```
There are several parts of the code:
- data folder: This folder contains the training and testing data for the target model.  In order to reduce the memory space, we just list the  links to theset dataset here. 
   -- Bank: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
   -- CIFAR10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
   -- CIFAR100: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
   -- Tiny-ImageNet: http://cs231n.stanford.edu/tiny-imagenet-200.zip
- models folder: This folder contains three types of model structures, including the model structure of the Data Recovery Model, the model structure of MAE Model and the model structure of VFL Model.
- results folder: This folder contains the saved parameters for the aforementioned three model architectures, including the MAE official pre-trained parameters.
- params.py: This file contains the parameter setting of the model structure.
- utils.py: This file contains the function of data loading and preprocessing.
- VFL_training.py: This file contains the function of federated learning based on VFL.
- MAE_finetune.py: This file contains the function of MAE finetuning.
- ***Data_recovery.py: This file contains the main function of data recovery based on MAE.***
- test.py: This file contains the function of testing the model performance on the target dataset.

## Parameter Setting of DARE
The attack settings are determined in the parameter **args** in **params.py**. 
- ***Vertical Federated Learning Model Training Settings***
-- args.dataset: the name of dataset
-- args.seed: random seed
-- args.save: whether to save every model
-- args.vfl_model: the bottom model type of in VFL training
-- args.vfl_epochs: number of total epochs to run
-- args.vfl_batch_size: mini-batch size (default: 128)
-- args.vfl_lr: initial learning rate
-- args.vfl_momentum: momentum for vfl training
-- args.vfl_weight_decay: weight decay (default: 5e-4)
-- args.vfl_step_gamma: gamma for step scheduler
- ***MAE Model Training Settings***
-- args.image_size: the size of input image
-- args.patch_size: patch size for mae model
-- args.mae_batch_size: batch size for MAE pre-training
-- args.mae_pretrain_epochs: number of total epochs to run: cifar(200) and tiny-imagenet(500)
-- args.mae_finetune_epochs: number of total epochs to run: cifar(200) and tiny-imagenet(500)
-- args.mae_warm_epochs: number of epochs for warm-up lr-schedule
-- args.mae_lr: learning rate for training MAE
-- args.mae_warm_start_lr: warm-up start learning rate
-- args.mae_warm_end_lr: warm-up end learning rate
- ***Data Recovery Model Training Settings***
-- args.is_recovery_supervised: whether to train supervised attack model. True(supervised training) or False(unsupervised)
-- args.attack_batch_size: batch size for data recovery
-- args.attack_epochs: number of total epochs to run
-- args.attack_lr: learning rate for training data recovery model
-- args.attack_noise_type: type of attack noise, choices=['None', 'Noise', 'Soteria']



## Execute DARE
1. Run VFL_training.py for VFL frameworks.  
2. Run MAE_pretrain.py for MAE pre-training model.  
3. Run MAE_finetune.py for MAE fine-tuning model.  
4. Run Data_recovery.py for data recovery.  





