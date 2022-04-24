# LightNILM

# LightNILM


**Requirements**

0. This software was tested on Window 2004

1. Create your virtual environment Python 3.5-3.8

2. Install Tensorflow = 2.0.0

    * Follow official instruction on https://www.tensorflow.org/install/
    
    * Remember a GPU support is highly recommended for training
    
3. Install Keras > 2.1.5 (Tested on Keras 2.3.1)

    * Follow official instruction on https://keras.io/
    
4. Clone this repository


### **REFIT**

Download the REFIT raw data from the original website (https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned). 
Appliances and training set composition for this project:

| Appliances      |      training               |  validation | test   |
|-----------------|:---------------------------:|:-----------:|:------:|
| kettle          | 3, 4, 7, 8, 9,12,13,19,20   |     5       |   2    |
| microwave       | 10, 12, 19                  |     17      |   4    |
| fridge          | 2, 5, 9                     |     12      |   15   |
| dish washer     | 5, 7, 9, 13, 16             |     18      |   20   |
| washing machine | 2,5,7,8,9,15,16,17          |     18      |   8    |


### **UK-DALE**

Download the UK-DALE raw data from the original website (http://jack-kelly.com/data/). 
Validation is a 13% slice from the final training building. 
Appliances and training set composition for this project:

| Appliances      |      training   |  validation | test   |
|-----------------|:---------------:|:-----------:|:------:|
| kettle          | 1               |     1       |   2    |
| microwave       | 1               |     1       |   2    |
| fridge          | 1               |     1       |   2    |
| dishwasher      | 1               |     1       |   2    |
| washingmachine  | 1               |     1       |   2    |


### **REDD**

Download the REDD raw data from the original website (http://redd.csail.mit.edu/).
Validation is a 10% slice from the final training building. 
Appliances and training set composition for this project:

| Appliances      |      training   |  validation | test   |
|-----------------|:---------------:|:-----------:|:------:|
| microwave       | 2,3             |     3       |   1    |
| fridge          | 2,3             |     3       |   1    |
| dishwasher      | 2,3             |     3       |   1    |
| washingmachine  | 2,3             |     3       |   1    |



Training default parameters:

	Input window: 599 samples
	Number of maximum: epochs 50
	Batchsize: 1000

	Early stopping：
		min epochs: 3
		patience: 4
		
	Adam optimiser:
		Learning rate: 0.001
		Beta1: 0.9
		Beta2: 0.999
		Epsilon: 10^{-8}

Train and Test (for example):

Train the whole model, randomly initialised, using 10000 data points, with ResNet model:

python train_main.py --appliance_name kettle --datadir REFIT --save_dir ./trained_model --crop 10000  --network_type resnet


Test the resnet model using 10000 data points:

python test_main.py --appliance_name kettle --datadir REFIT --trained_model_dir ./trained_model --save_results_dir ./result  --crop_dataset 10000 --network_type resnet






