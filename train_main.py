import argparse
from remove_space import remove_space
from seq2point_train import Trainer

# Allows a model to be trained from the terminal.

parser = argparse.ArgumentParser(description="Train sequence-to-point learning for energy disaggregation. ")
appName = "fridge"  # kettle, fridge, washing machine, dishwasher, microwave.
dir = 'D:/dataset/'
parser.add_argument("--datadir", type=str, default='REFIT')  # UK-DALE REFIT REDD
parser.add_argument("--appliance_name", type=remove_space, default=appName)
parser.add_argument("--batch_size", type=int, default="200")
parser.add_argument("--crop", type=int, default="200000")
parser.add_argument("--valstep", type=int, default="1000")
parser.add_argument("--network_type", type=remove_space,
                    default="srnn")  # 'mobilenet' 'densenet' 'resnet' 'srnn' 'seq2point'
parser.add_argument("--epochs", type=int, default="15")
parser.add_argument("--mobile", type=bool, default=True)  # True False
parser.add_argument("--input_window_length", type=int, default="512")
parser.add_argument("--validation_frequency", type=int, default="1")

arguments = parser.parse_args()
training_directory = dir + arguments.datadir + '/' + arguments.appliance_name + '/' + arguments.appliance_name + '_training_' + '.csv'
validation_directory = dir + arguments.datadir + '/' + arguments.appliance_name + '/' + arguments.appliance_name + '_validation_' + '.csv'
print(training_directory, '----', validation_directory)
save_model_dir = "saved_models/" + arguments.datadir + '_s/' + arguments.appliance_name + "_" + arguments.network_type + "_model.h5"
print(save_model_dir)
trainer = Trainer(arguments.appliance_name, arguments.batch_size, arguments.crop, arguments.network_type,
                  training_directory, validation_directory, save_model_dir,
                  epochs=arguments.epochs, input_window_length=arguments.input_window_length,
                  validation_frequency=arguments.validation_frequency, mb=arguments.mobile, valstep=arguments.valstep)
trainer.train_model()
