import argparse
from remove_space import remove_space
from seq2point_test import Tester,tfThread
import tensorflow as tf
from tensorflow.python.tools import freeze_graph 
# Allows a model to be tested from the terminal.

# You need to input your test data directory
test_directory="~"
dir = 'D:/dataset/'
parser = argparse.ArgumentParser(description="Train a pruned neural network for energy disaggregation. ")
appName = "fridge" # kettle, fridge, washing machine, dishwasher, microwave.
parser.add_argument('--datadir',type=str,default='REFIT') #UK-DALE REFIT REDD
parser.add_argument("--appliance_name", type=remove_space, default=appName)
parser.add_argument("--batch_size", type=int, default="1024")
parser.add_argument("--crop", type=int, default="6000000")
parser.add_argument("--algorithm", type=remove_space, default="srnn") # 'mobilenet' 'densenet' 'resnet' 'srnn'
parser.add_argument("--network_type", type=remove_space, default="") 
parser.add_argument("--input_window_length", type=int, default="512")
parser.add_argument("--test_directory", type=str, default=test_directory)
parser.add_argument("--plot_result", type=bool, default=False)

arguments = parser.parse_args()
test_directory   = dir + arguments.datadir + '/' + arguments.appliance_name + '/' + arguments.appliance_name + '_test_' + '.csv'  #training, validation, test
saved_model_dir = "saved_models/" + arguments.datadir + '_s/' + arguments.appliance_name + "_" + arguments.algorithm + "_model.h5" #_best
# saved_model_dir = "saved_models/" + arguments.datadir + '_s/' + arguments.appliance_name + "_seq2point_model.h5" #_best
log_file_dir = "saved_models/" + arguments.datadir + '_s/' + arguments.appliance_name + "_" + arguments.algorithm + "_" + arguments.algorithm + ".log"
pb_output_dir = "saved_models/" + arguments.datadir + '_s/' + arguments.appliance_name + "_" + arguments.algorithm + "_model.pb" #_best

tester = Tester(arguments.appliance_name, arguments.algorithm, arguments.crop,
                arguments.batch_size, arguments.algorithm,
                test_directory, saved_model_dir, log_file_dir,
                arguments.input_window_length,arguments.plot_result,arguments.datadir)
tester.test_model()





