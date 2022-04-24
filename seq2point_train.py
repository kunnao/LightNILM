import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from model_structure import mnloss
from data_feeder import TrainSlidingWindowGenerator
from model_structure import create_s2p_model, save_model, create_mobilenet_model, ShuffleNetv2, create_dense_model, \
    create_GRU_model


class Trainer():

    def __init__(self, appliance, batch_size, crop, network_type, training_directory, validation_directory,
                 save_model_dir,
                 epochs=10, input_window_length=599, validation_frequency=1, patience=4, min_delta=1e-5, verbose=1,
                 mb=True, valstep=1000):
        self.__appliance = appliance
        self.__algorithm = network_type
        self.__network_type = network_type
        self.__crop = crop
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__patience = patience
        self.__min_delta = min_delta
        self.__verbose = verbose
        self.__loss = "mse"  # mnloss
        self.__metrics = ["mse"]
        self.__learning_rate = 0.001
        self.__beta_1 = 0.9
        self.__beta_2 = 0.999
        self.__save_model_dir = save_model_dir

        self.__input_window_length = input_window_length
        self.__window_size = self.__input_window_length
        self.__window_offset = int(self.__window_size // 2)
        self.__max_chunk_size = 5 * 10 ** 3
        self.__validation_frequency = validation_frequency
        self.__ram_threshold = 5 * 10 ** 4
        self.__skip_rows_train = 2 * 10 ** 5
        self.__validation_steps = valstep
        self.__skip_rows_val = 10 ** 3

        self.__training_directory = training_directory
        self.__validation_directory = validation_directory

        self.__training_chunker = TrainSlidingWindowGenerator(file_name=self.__training_directory,
                                                              chunk_size=self.__max_chunk_size,
                                                              batch_size=self.__batch_size, crop=self.__crop,
                                                              shuffle=True, skip_rows=self.__skip_rows_train,
                                                              offset=self.__window_offset,
                                                              ram_threshold=self.__ram_threshold,
                                                              qo=self.__window_size % 2)
        self.__validation_chunker = TrainSlidingWindowGenerator(file_name=self.__validation_directory,
                                                                chunk_size=self.__max_chunk_size,
                                                                batch_size=self.__batch_size, crop=self.__crop,
                                                                shuffle=True, skip_rows=self.__skip_rows_val,
                                                                offset=self.__window_offset,
                                                                ram_threshold=self.__ram_threshold,
                                                                qo=self.__window_size % 2)
        self.__mobile = mb

    def train_model(self):
        steps_per_training_epoch = np.round(int(self.__training_chunker.total_num_samples / self.__batch_size),
                                            decimals=0)
        # ----------------------tensorboard and checkpoint------------------------#

        logdir = os.path.join("logs")  # 记录回调tensorboard日志打印记录
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True,
                                                     write_images=True)
        modeldir = os.path.join(self.__save_model_dir)
        modeldir, filename = os.path.split(modeldir)
        modeldir = os.path.join(modeldir, self.__appliance + '_seq2point_model_best.h5')  #
        print(modeldir)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=modeldir, monitor='val_loss',
                                                        save_best_only=True)  # False
        # ----------------------Keras Model------------------------#
        if self.__network_type == 'mobilenet':
            model = create_mobilenet_model(self.__input_window_length, True)
        elif self.__network_type == 'resnet':
            model = create_mobilenet_model(self.__input_window_length, False)
        elif self.__network_type == 'densenet':
            model = create_dense_model(self.__input_window_length, False)
        elif self.__network_type == 'seq2point':
            model = create_s2p_model(self.__input_window_length)
        elif self.__network_type == 'srnn':
            model = create_GRU_model(self.__input_window_length)
        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.__learning_rate, beta_1=self.__beta_1,
                                                         beta_2=self.__beta_2, epsilon=1e-08), loss=self.__loss,
                      metrics=self.__metrics)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=self.__min_delta,
                                                          patience=self.__patience, verbose=self.__verbose, mode="auto")
        print(model)  # mean_absolute_error
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        sess = tf.compat.v1.Session(config=config)

        callbacks = [early_stopping, checkpoint]  # , tensorboard

        training_history = self.default_train(model, callbacks, steps_per_training_epoch)

        training_history.history["val_loss"] = np.repeat(training_history.history["val_loss"],
                                                         self.__validation_frequency)

        save_model(model, self.__network_type, self.__algorithm, self.__appliance, self.__save_model_dir)

        # self.plot_training_results(training_history)

    def default_train(self, model, callbacks, steps_per_training_epoch):
        training_history = model.fit(self.__training_chunker.load_dataset(), steps_per_epoch=steps_per_training_epoch,
                                     epochs=self.__epochs,
                                     verbose=self.__verbose, callbacks=callbacks,
                                     validation_data=self.__validation_chunker.load_dataset(),
                                     validation_freq=self.__validation_frequency,
                                     validation_steps=self.__validation_steps)

        return training_history

    def plot_training_results(self, training_history):

        plt.plot(training_history.history["loss"], label="MSE (Training Loss)")
        plt.plot(training_history.history["val_loss"], label="MSE (Validation Loss)")
        plt.title('Training History')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()
