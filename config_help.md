This file is help to write a customized config file



### Default configurations
All default configuration can be seen from _config/verbose/default.yaml_ 

**Note**: when training models, the complete config file (i.e., contains all parameters for a specified model) will be generated 
and stored at the results folder 


### Config Structure
As can be seen from the config file can be divivded to mainly 3 parts: data, model and training_parameters

data: describe files from the program

training_parameters: describe parametes from training process, such as max iteration, warm-up ratio etc

model: describe the configuration of models, the model structure is illustrated in the following figure
![Alt text](info/code_structure_plot.png?raw=true "model structure")


### available configs
For all available configs, they can be found in _config/config.py_ and _config/function_config_lib.py_
 In the config file, there are a lot of method/par pairs, this is because different methods can have different 
 parameters, for example, for modal_combine, there are MFH and non_linear_elmt_multiply, if only change the method, 
 the corresponding default parameters will be load automatically  

To change a config file, the user only need replace the correlated part, for example, to run model with image feature 
fine-tune, the _config/keep/detectron23_finetune.yaml_ can be used, while the verbose configure file generated is
 _config/verbose/dectectron_finetune.yaml_ 
 


### config overwrite
To avoid creating a separated config file for each minor change, pythia also allows to overwrite config files from 
command-line arguments by using _--config_overwrite_. The overwrite a string follow a semi-json format, 
i.e. no quote for keys. The following are several examples

change the steps to adjust learning rate
```bash
starter_priority.sh 8 1 72 500G vqa_suite  train \
--config_overwrite '{training_parameters:{lr_steps:[17000,20000,23000],max_iter:25000}}' 
```

change the dataset to use all val2014 in training
```bash
starter_priority.sh 8 1 72 500G vqa_suite  train \
--config_overwrite '{data:{imdb_file_train:[\"imdb/imdb_train2014.npy\",\"imdb/imdb_val2014.npy\"]}}'
```
