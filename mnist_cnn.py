'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, LeakyReLU, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

batch_size = 128
num_classes = 10
epochs = 5



def load_mnist():
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test

def buildModel():
    input_shape = (28, 28, 1)
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, input_shape=input_shape)(inputs)
    x = LeakyReLU()(x)
    x = Conv2D(64, 3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(64, 3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(64, 3)(x)
    x = LeakyReLU()(x)
    x = Conv2D(64, 3)(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, 1)(x)
    x = LeakyReLU()(x)

    x = Conv2D(num_classes, 1)(x)
    x = Activation('softmax')(x)

    # x = Dropout(0.25) (x)
    x = Flatten()(x)
    # x = Dense(128)(x)
    # x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, x, name='mnist_cnn')
    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    model = buildModel()
    train = False

    if train:
        x_train, y_train, x_test, y_test = load_mnist()

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        model.save_weights('mnist_fcn_lrelu_sm.h5')

    else:
        model.load_weights('mnist_fcn_lrelu_sm.h5')

        from conv_filter_visualization import visualize_layer
        visualize_layer(model, 'conv2d_1', step=1, epochs=200, init='train_random')

