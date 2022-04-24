import numpy as np 
import pandas as pd 

# batch_size: the number of rows fed into the network at once.
# crop: the number of rows in the data set to be used in total.
# chunk_size: the number of lines to read from the file at once.

class TrainSlidingWindowGenerator():


    def __init__(self, file_name, chunk_size, shuffle, offset, batch_size=1000, 
                crop=100000, skip_rows=0, ram_threshold=5 * 10 ** 5,qo=1):
        self.__file_name = file_name
        self.__batch_size = batch_size
        self.__chunk_size = chunk_size
        self.__shuffle = shuffle
        self.__offset = offset
        self.__crop = crop
        self.__skip_rows = skip_rows
        self.__ram_threshold = ram_threshold
        self.total_size = 100
        self.__total_num_samples = crop
        self.qo = qo

    @property
    def total_num_samples(self):
        return self.__total_num_samples
    
    @total_num_samples.setter
    def total_num_samples(self, value):
        self.__total_num_samples = value

    def check_if_chunking(self):

        # Loads the file and counts the number of rows it contains.
        print("Importing training file...")
        chunks = pd.read_csv(self.__file_name, 
                            header=0, 
                            nrows=self.__crop, 
                            skiprows=self.__skip_rows)
        print("Counting number of rows...")
        self.total_size = len(chunks)
        del chunks
        print("Done.")

        print("The dataset contains ", self.total_size, " rows")

        # Display a warning if there are too many rows to fit in the designated amount RAM.
        if (self.total_size > self.__ram_threshold):
            print("There is too much data to load into memory, so it will be loaded in chunks. Please note that this may result in decreased training times.")
    

    def load_dataset(self):

        if self.total_size == 0:
            self.check_if_chunking()

        # If the data can be loaded in one go, don't skip any rows.
        if (self.total_size <= self.__ram_threshold):

            # Returns an array of the content from the CSV file.
            data_array = np.array(pd.read_csv(self.__file_name, nrows=self.__crop, skiprows=self.__skip_rows, header=0))
            inputs = data_array[:, 0]
            outputs = data_array[:, 1]

            maximum_batch_size = inputs.size - 2 * self.__offset
            self.total_num_samples = maximum_batch_size
            if self.__batch_size < 0:
                self.__batch_size = maximum_batch_size

            indicies = np.arange(maximum_batch_size)
            

            while True:
                if self.__shuffle:
                    np.random.shuffle(indicies)
                for start_index in range(0, maximum_batch_size, self.__batch_size):
                    splice = indicies[start_index : start_index + self.__batch_size]
                    input_data = np.array([inputs[index : index + 2 * self.__offset + self.qo] for index in splice])
                    output_data_seq = np.array([outputs[index : index + 2 * self.__offset + self.qo] for index in splice])
                    output_data = outputs[splice + self.__offset].reshape(-1, 1)

                    yield input_data, output_data
                    
        # Skip rows where needed to allow data to be loaded properly when there is not enough memory.
        if (self.total_size >= self.__ram_threshold):
            k = self.__crop / self.__chunk_size
            number_of_chunks = np.arange(k)
            if self.__shuffle:
                np.random.shuffle(number_of_chunks)
            i = 0
            # Yield the data in sections.
            '''for index in number_of_chunks:
                data_array = np.array(pd.read_csv(self.__file_name, skiprows=int(index) * self.__chunk_size, header=0, nrows=self.__crop))                   
                inputs = data_array[:, 0]
                outputs = data_array[:, 1]

                maximum_batch_size = inputs.size - 2 * self.__offset
                self.total_num_samples = maximum_batch_size
                if self.__batch_size < 0:
                    self.__batch_size = maximum_batch_size

                indicies = np.arange(maximum_batch_size)
                if self.__shuffle:
                    np.random.shuffle(indicies)

            while True:
                for start_index in range(0, maximum_batch_size, self.__batch_size):
                    splice = indicies[start_index : start_index + self.__batch_size]
                    input_data = np.array([inputs[index : index + 2 * self.__offset + self.qo] for index in splice])
                    output_data_seq = np.array([outputs[index : index + 2 * self.__offset + self.qo] for index in splice])
                    output_data = outputs[splice + self.__offset].reshape(-1, 1)

                    yield input_data, output_data
                    '''
        
            while True:
                data_array = np.array(pd.read_csv(self.__file_name, skiprows=int(i) * self.__chunk_size+self.__skip_rows, header=0, nrows=self.__chunk_size))
                i = i+1
                inputs = data_array[:, 0]
                outputs = data_array[:, 1]
                if inputs.size != self.__chunk_size or i > k+1:   #
                    i = 0
                maximum_batch_size = inputs.size - 2 * self.__offset
                self.total_num_samples = maximum_batch_size
                if self.__batch_size < 0:
                   self.__batch_size = maximum_batch_size

                indicies = np.arange(maximum_batch_size)
                if self.__shuffle:
                    np.random.shuffle(indicies)
                for start_index in range(0, maximum_batch_size, self.__batch_size):
                    splice = indicies[start_index : start_index + self.__batch_size]
                    input_data = np.array([inputs[index : index + 2 * self.__offset + self.qo] for index in splice])
                    output_data_seq = np.array([outputs[index : index + 2 * self.__offset + self.qo] for index in splice])
                    output_data = outputs[splice + self.__offset].reshape(-1, 1)
                    yield input_data, output_data

class ValSlidingWindowGenerator():
    def __init__(self, file_name, chunk_size, shuffle, offset, batch_size=1000, 
                crop=100000, skip_rows=0, ram_threshold=5 * 10 ** 5,qo=1):
        self.__file_name = file_name
        self.__batch_size = batch_size
        self.__chunk_size = chunk_size
        self.__shuffle = shuffle
        self.__offset = offset
        self.__crop = crop 
        self.__skip_rows = skip_rows
        self.__ram_threshold = ram_threshold
        self.total_size = 100000
        self.__total_num_samples = crop
        self.qo = qo

    @property
    def total_num_samples(self):
        return self.__total_num_samples
    
    @total_num_samples.setter
    def total_num_samples(self, value):
        self.__total_num_samples = value


    def load_dataset(self):
        i = 0
        k = self.__crop / self.__chunk_size
        while True:
            data_array = np.array(pd.read_csv(self.__file_name, skiprows=int(i) * self.__chunk_size+self.__skip_rows, header=0, nrows=self.__chunk_size))       
            i = i+1           
            inputs = data_array[:, 0]
            outputs = data_array[:, 1]
            if inputs.size != self.__chunk_size or i > k + 10:   #
                i = 0
            maximum_batch_size = inputs.size - 2 * self.__offset
            self.total_num_samples = maximum_batch_size
            if self.__batch_size < 0:
               self.__batch_size = maximum_batch_size

            indicies = np.arange(maximum_batch_size)
            if self.__shuffle:
                np.random.shuffle(indicies)
            for start_index in range(0, maximum_batch_size, self.__batch_size):
                splice = indicies[start_index : start_index + self.__batch_size]
                input_data = np.array([inputs[index : index + 2 * self.__offset + self.qo] for index in splice])
                output_data_seq = np.array([outputs[index : index + 2 * self.__offset + self.qo] for index in splice])
                output_data = outputs[splice + self.__offset].reshape(-1, 1)
                yield input_data, output_data

class TestSlidingWindowGenerator(object):

    def __init__(self, number_of_windows, inputs, targets, offset,qo=1):
        self.__number_of_windows = number_of_windows
        self.__offset = offset
        self.__inputs = inputs
        self.__targets = targets
        self.total_size = len(inputs)
        self.qo = qo

    def load_dataset(self):

        self.__inputs = self.__inputs.flatten()
        max_number_of_windows = self.__inputs.size - 2 * self.__offset

        if self.__number_of_windows < 0:
            self.__number_of_windows = max_number_of_windows

        indicies = np.arange(max_number_of_windows, dtype=int)
        for start_index in range(0, max_number_of_windows, self.__number_of_windows):
            splice = indicies[start_index : start_index + self.__number_of_windows]
            input_data = np.array([self.__inputs[index : index + 2 * self.__offset + self.qo] for index in splice])
            target_data = self.__targets[splice].reshape(-1, 1)  # + self.__offset
            yield input_data, target_data