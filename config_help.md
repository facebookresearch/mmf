This file is to help with writing a customized config file

### Default configurations
All default configuration files can be seen in _config/verbose/default.yaml_ 

**Note**: When training models, the complete config file (i.e., containing all parameters for a specified model) will be generated and stored in the results folder 

### Config Structure
The config file can be divide into 3 main parts: data, model and training_parameters

data: describe all the data files like embeddings, features and annotations needed for training and evaluation

training_parameters: describe parametes for the training process like such as max iterations, warm-up ratio etc

model: describe the configuration of models, the model structure is illustrated in the following figure




![Alt text](info/code_structure_plot.png?raw=true "model structure")





### Available configs

All available configs can be found in _config/config.py_ and _config/function_config_lib.py_
In the config file, there are many method/par pairs. This is because different methods can have different 
parameters, for example, for modal_combine (for multi modal combination), there are MFH and non_linear_elmt_multiply. If only the method is changed, the coresponding default parameters will be loaded automatically  

To change a config file, the user only needs to replace the correlated part, for example, to run model with image feature 
fine-tune, the _config/keep/detectron23_finetune.yaml_ can be used, while the verbose config file generated is
 _config/verbose/dectectron_finetune.yaml_ 
 
### Config overwrite
To avoid creating a separate config file for each minor change, pythia also allows to overwrite config files from 
command-line arguments by using _--config_overwrite_. The overwrite a string follow a semi-json format, 
i.e. no quote for keys. The following are several examples

change the steps to adjust learning rate
```bash
python train.py \
--config_overwrite '{training_parameters:{lr_steps:[17000,20000,23000],max_iter:25000}}' 
```

change the dataset to use all val2014 in training
```bash
python train.py \
--config_overwrite '{data:{imdb_file_train:[\"imdb/imdb_train2014.npy\",\"imdb/imdb_val2014.npy\"]}}'
```
