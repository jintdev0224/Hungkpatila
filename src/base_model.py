""" CNN based on AlexNet from 2011.

CNN:
    INPUT -> Feature Extraction Steps
            -> CONV_1_RELU -> POOL_1
              -> CONV_2_RELU -> POOL_2
                -> CONV_3_RELU -> POOL_3
                  -> Classifing Steps
                    -> DROPOUT_1
                      -> FULLY_CONNECTED_MLP_RELU
                        -> DROPOUT_2
                          -> FULLY_CONNECTED_MLP_SOFTMAX

Model Summary:
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 32, 32, 16)        208       
                                                                     
     max_pooling2d (MaxPooling2D  (None, 16, 16, 16)       0         
     )                                                               
                                                                     
     conv2d_1 (Conv2D)           (None, 16, 16, 32)        2080      
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 8, 8, 32)         0         
     2D)                                                             
                                                                     
     conv2d_2 (Conv2D)           (None, 8, 8, 64)          8256      
                                                                     
     max_pooling2d_2 (MaxPooling  (None, 4, 4, 64)         0         
     2D)                                                             
                                                                     
     dropout (Dropout)           (None, 4, 4, 64)          0         
                                                                     
     flatten (Flatten)           (None, 1024)              0         
                                                                     
     dense (Dense)               (None, 500)               512500    
                                                                     
     dropout_1 (Dropout)         (None, 500)               0         
                                                                     
     dense_1 (Dense)             (None, 10)                5010                                                             
    =================================================================
    Total params: 528,054
    Trainable params: 528,054
    Non-trainable params: 0
    _________________________________________________________________
    None

"""

from dataclasses import dataclass

import keras

import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import np_utils

TRAINING_DATASET_SIZE = 5000


@dataclass
class ConvolutionalLayerSettings:
    filters: int
    kernel_size: int
    padding: str
    activation_function: str
    input_shape: tuple = None
    pool_size: int = None

@dataclass
class FullyConnectedLayerSettings:
    dropout_rate: float
    dense_layer_count: int
    activation_function: str

@dataclass
class ModelCompilationSettings:
    loss_function: str
    optimizer: str
    metrics: list

@dataclass
class ModelCheckpointSettings:
    filepath: str
    verbose: int
    save_best_only: bool

@dataclass
class ModelFitSettings:
    batch_size: int
    epochs: int
    verbose: int
    shuffle: bool

CONV_2D_1_SETTINGS = ConvolutionalLayerSettings(
    filters=16,
    kernel_size=2,
    padding='same',
    activation_function='relu',
    input_shape=(32, 32, 3),
    pool_size=2
)
CONV_2D_2_SETTINGS = ConvolutionalLayerSettings(
    filters=32,
    kernel_size=2,
    padding='same',
    activation_function='relu',
    pool_size=2
)
CONV_2D_3_SETTINGS = ConvolutionalLayerSettings(
    filters=64,
    kernel_size=2,
    padding='same',
    activation_function='relu',
    pool_size=2
)
FULLY_CONNECTED_1_SETTINGS = FullyConnectedLayerSettings(
    dropout_rate=0.3,
    dense_layer_count=500,
    activation_function='relu'
)
FULLY_CONNECTED_2_SETTINGS = FullyConnectedLayerSettings(
    dropout_rate=0.4,
    dense_layer_count=10,
    activation_function='softmax'
)
MODEL_COMPILATION_SETTINGS = ModelCompilationSettings(
    loss_function='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)
MODEL_CHECKPOINT_SETTINGS = ModelCheckpointSettings(
    filepath='model.weights.best.hdf5',
    verbose=1,
    save_best_only=True
)
MODEL_FIT_SETTINGS = ModelFitSettings(
    batch_size=32,
    epochs=100,
    verbose=2,
    shuffle=True
)

# I don't understand why I need this line and the one inside load_cifar_dataset for it to work
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
def load_cifar_dataset():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)


def rescale_images(input_set):
    return input_set.astype('float32') / 255


def show_figures():
    fig = plt.figure(figsize=(20, 5))
    for i in range(36):
        ax = fig.add_subplot(3, 12, i+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(x_train[i]))
    plt.show()


def one_hot_encoding(input_set):
    """ takes our 10 classes ["airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"] and converts them to image_1,
        image_2, ..., image_10. That's what one-hot-encoding is.
    """
    return np_utils.to_categorical(input_set, len(np.unique(input_set)))


def split_dataset(input_set, split_cutoff):
    """ split the data sets into training and validation sets
    """
    return input_set[split_cutoff:], input_set[:split_cutoff]


def feature_extraction(model):
    """ Run the feature extraction portion of the module description.
        INPUT -> Feature Extraction Steps
                -> CONV_1_RELU -> POOL_1
                  -> CONV_2_RELU -> POOL_2
                    -> CONV_3_RELU -> POOL_3
    """
    model.add(Conv2D(
        filters=CONV_2D_1_SETTINGS.filters,
        kernel_size=CONV_2D_1_SETTINGS.kernel_size,
        padding=CONV_2D_1_SETTINGS.padding,
        activation=CONV_2D_1_SETTINGS.activation_function,
        input_shape=CONV_2D_1_SETTINGS.input_shape
    ))
    model.add(MaxPooling2D(pool_size=CONV_2D_1_SETTINGS.pool_size))
    model.add(Conv2D(
        filters=CONV_2D_2_SETTINGS.filters,
        kernel_size=CONV_2D_2_SETTINGS.kernel_size,
        padding=CONV_2D_2_SETTINGS.padding,
        activation=CONV_2D_2_SETTINGS.activation_function
    ))
    model.add(MaxPooling2D(pool_size=CONV_2D_2_SETTINGS.pool_size))
    model.add(Conv2D(
        filters=CONV_2D_3_SETTINGS.filters,
        kernel_size=CONV_2D_3_SETTINGS.kernel_size,
        padding=CONV_2D_3_SETTINGS.padding,
        activation=CONV_2D_3_SETTINGS.activation_function
    ))
    model.add(MaxPooling2D(pool_size=CONV_2D_3_SETTINGS.pool_size))


def classification(model):
    """ Run the fully connected layer's classification portion of the module description.
        -> Classifing Steps
          -> DROPOUT_1
            -> FULLY_CONNECTED_MLP_RELU
              -> DROPOUT_2
                -> FULLY_CONNECTED_MLP_SOFTMAX
    """
    model.add(Dropout(FULLY_CONNECTED_1_SETTINGS.dropout_rate))
    model.add(Flatten())
    model.add(Dense(
        FULLY_CONNECTED_1_SETTINGS.dense_layer_count,
        activation=FULLY_CONNECTED_1_SETTINGS.activation_function
    ))
    model.add(Dropout(FULLY_CONNECTED_2_SETTINGS.dropout_rate))
    model.add(Dense(
        FULLY_CONNECTED_2_SETTINGS.dense_layer_count,
        activation=FULLY_CONNECTED_2_SETTINGS.activation_function
    ))


def compilation(model):
    """ Compiles the model with the MODEL_COMPILATION_SETTINGS
    """
    model.compile(
        loss=MODEL_COMPILATION_SETTINGS.loss_function,
        optimizer=MODEL_COMPILATION_SETTINGS.optimizer,
        metrics=MODEL_COMPILATION_SETTINGS.metrics
    )

def checkpoint(model):
    """ sets up a model checkpointer for debugging
    """
    return ModelCheckpoint(
        filepath=MODEL_CHECKPOINT_SETTINGS.filepath,
        verbose=MODEL_CHECKPOINT_SETTINGS.verbose,
        save_best_only=MODEL_CHECKPOINT_SETTINGS.save_best_only
    )

def model_fit(model, x_train, y_train, x_valid, y_valid):
    """ Trains the network by running the .fit() method
    """
    return model.fit(
        x_train,
        y_train,
        batch_size=MODEL_FIT_SETTINGS.batch_size,
        epochs=MODEL_FIT_SETTINGS.epochs,
        validation_data=(x_valid, y_valid),
        callbacks=[checkpoint(model)],
        verbose=MODEL_FIT_SETTINGS.verbose,
        shuffle=MODEL_FIT_SETTINGS.shuffle
    )

def main():
    # load the training and test datasets
    (x_train, y_train), (x_test, y_test) = load_cifar_dataset()
    # rescale the images to a standardized size
    x_train = rescale_images(x_train)
    x_test = rescale_images(x_test)
    # one hot encode the y test and training sets
    y_train = one_hot_encoding(input_set=y_train)
    y_test = one_hot_encoding(input_set=y_test)
    # split the datasets into training and validation sets
    (x_train, x_valid) = split_dataset(input_set=x_train, split_cutoff=TRAINING_DATASET_SIZE)
    (y_train, y_valid) = split_dataset(input_set=y_train, split_cutoff=TRAINING_DATASET_SIZE)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], '-> training sample size')
    print(x_valid.shape[0], '-> validation sample size')
    print(x_test.shape[0], '-> testing sample size')
    # initialize the model
    model = Sequential()
    # build the feature extrator
    feature_extraction(model=model)
    # build the classifier
    classification(model=model)
    # compile the model
    compilation(model=model)
    # fit/train the model
    model_fit(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid
    )
    # load the model with the best weights/outcomes
    model.load_weights(MODEL_CHECKPOINT_SETTINGS.filepath)
    #evaluate the model
    score = model.evaluate(x_test, y_test, verbose=0)

    print(f'\n\nTest Accuracy: {score[1]}')
    print(model.summary())

    # show_figures()

if __name__ == '__main__':
    main()

