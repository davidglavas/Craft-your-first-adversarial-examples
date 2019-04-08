import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

def get_data():
    """
    Prepares the MNIST dataset.
    """

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # reshape to (n_samples, 28, 28, 1)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # cast pixels to floats, normalize to [0, 1] range
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    def to_categorical(y, num_classes=None, dtype='float32'):
        """
        Converts a class vector (integers) to a binary class matrix.
        """
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=dtype)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical

    # one-hot-encode the labels
    Y_train = to_categorical(y_train, 10)
    Y_test = to_categorical(y_test, 10)

    return X_train, Y_train, X_test, Y_test


def get_model():
    """
    Prepares the target neural network, a Keras 'Sequential' instance.
    """
    # MNIST model
    layers = [
        Conv2D(64, (3, 3), padding='valid', input_shape=(28, 28, 1)),
        Activation('relu'),
        Conv2D(64, (3, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(128),
        Activation('relu'),
        Dropout(0.5),
        Dense(10),
        Activation('softmax')
    ]

    model = Sequential()
    for layer in layers:
        model.add(layer)

    return model


X_train, Y_train, X_test, Y_test = get_data()
model = get_model()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adadelta',
    metrics=['accuracy']
)

model.fit(
    X_train, Y_train,
    epochs=20,
    batch_size=128,
    shuffle=True,
    verbose=1,
    validation_data=(X_test, Y_test)
)

model.save('%s_model.h5' % 'MNIST')
print('Finished training the MNIST model, it has been saved for later use.')
