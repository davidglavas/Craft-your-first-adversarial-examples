import tensorflow as tf

import numpy as np

import keras.backend as K
from keras.models import load_model
from keras.datasets import mnist

import matplotlib.pyplot as plt


def fgsm(x, predictions, eps, clip_min=None, clip_max=None, y=None):
    """
    Computes symbolic TF tensor for the adversarial examples. Note that this must
    be evaluated with a session.run call.
    :param x: the input placeholder
    :param predictions: the model's output tensor
    :param eps: the epsilon (input variation parameter)
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :param y: the output placeholder. Use None (the default) to avoid the
            label leaking effect.
    :return: a tensor for the adversarial example
    """

    # Compute loss
    if y is None:
        # In this case, use model predictions as ground truth
        y = tf.to_float(
            tf.equal(predictions,
                     tf.reduce_max(predictions, 1, keepdims=True)))
    y = y / tf.reduce_sum(y, 1, keepdims=True)
    logits, = predictions.op.inputs
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    )

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    # Take sign of gradient
    signed_grad = tf.sign(grad)

    # Multiply by constant epsilon
    scaled_signed_grad = eps * signed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_signed_grad)

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x

def fast_gradient_sign_method(sess, model, X, Y, eps, clip_min=None,
                              clip_max=None, batch_size=256):
    """
    Performs FGSM in batches.
    """
    # Define TF placeholders for the input and output
    x = tf.placeholder(tf.float32, shape=(None,) + X.shape[1:])
    y = tf.placeholder(tf.float32, shape=(None,) + Y.shape[1:])

    adv_x = fgsm(
        x, model(x), eps=eps,
        clip_min=clip_min,
        clip_max=clip_max, y=y
    )

    def batch_eval(sess, tf_inputs, tf_outputs, numpy_inputs, batch_size=None,
                   feed=None):
        """
        A helper function that computes a tensor on numpy inputs by batches.
        """

        n = len(numpy_inputs)
        assert n > 0
        assert n == len(tf_inputs)
        m = numpy_inputs[0].shape[0]
        for i in range(1, n):
            assert numpy_inputs[i].shape[0] == m
        out = []
        for _ in tf_outputs:
            out.append([])
        for start in range(0, m, batch_size):
            batch = start // batch_size

            # Compute batch start and end indices
            start = batch * batch_size
            end = start + batch_size
            numpy_input_batches = [numpy_input[start:end]
                                   for numpy_input in numpy_inputs]
            cur_batch_size = numpy_input_batches[0].shape[0]
            assert cur_batch_size <= batch_size
            for e in numpy_input_batches:
                assert e.shape[0] == cur_batch_size

            feed_dict = dict(zip(tf_inputs, numpy_input_batches))
            if feed is not None:
                feed_dict.update(feed)
            numpy_output_batches = sess.run(tf_outputs, feed_dict=feed_dict)
            for e in numpy_output_batches:
                assert e.shape[0] == cur_batch_size, e.shape
            for out_elem, numpy_output_batch in zip(out, numpy_output_batches):
                out_elem.append(numpy_output_batch)

        out = [np.concatenate(x, axis=0) for x in out]
        for e in out:
            assert e.shape[0] == m, e.shape
        return out

    print("Generating the adversarial examples. This might take a few minutes depending on your machine.")
    X_adv, = batch_eval(
        sess, [x, y], [adv_x],
        [X, Y], batch_size=256
    )

    return X_adv


def get_data():
    """
    Prepares MNIST.
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
        Converts a class vector (integers) to binary class matrix.
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


# Create TF session, set it as Keras backend
sess = tf.Session()
K.set_session(sess)
K.set_learning_phase(0)  # test time

model = load_model('%s_model.h5' % 'MNIST')

_, _, X_test, Y_test = get_data()

# FGSM attack
print('Crafting FGSM adversarial samples...')
X_adv = fast_gradient_sign_method(sess, model, X_test, Y_test, eps=0.300, clip_min=0., clip_max=1., batch_size=256)

# evaluate model on natural data
_, acc = model.evaluate(X_test, Y_test, batch_size=256, verbose=0)
print("Accuracy on the natural test set: %0.2f%%" % (100*acc))

# evaluate model on adversarial data
_, acc = model.evaluate(X_adv, Y_test, batch_size=256, verbose=0)
print("Model accuracy on the adversarial test set: %0.2f%%" % (100 * acc))

np.save('Adv_%s_%s.npy' % ('MNIST', 'FGSM'), X_adv)
print('Adversarial samples crafted and saved.')


def examineSoftmaxOutputs():
    def sideBySideMNIST(arr1, arr2):
        arr1 = arr1.reshape(28, 28)
        arr2 = arr2.reshape(28, 28)

        plt.subplot(121)  # numRows, numColumns, position of image (upper left is 1, increment by 1 as you go right)
        plt.imshow(arr1, cmap='gray')
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(arr2, cmap='gray')
        plt.axis('off')

        # plt.savefig("/data/MNISTtests.png", bbox_inches='tight', pad_inches=0)  # tight removes whitespace
        plt.show()

    naturalPred = model.predict(X_test[1].reshape(1, 28, 28, 1))
    adversarialPred = model.predict(X_adv[1].reshape(1, 28, 28, 1))

    np.set_printoptions(formatter={'float_kind':'{:f}'.format})
    print('naturalPred: ', naturalPred)
    print('adversarialPred: ', adversarialPred)

    # same as model.predict_classes(X_test[1].reshape(1, 28, 28, 1))
    print('Predicted class for natural example: ', naturalPred.argmax())
    print('Predicted class for adversarial example: ', adversarialPred.argmax())

    sideBySideMNIST(X_test[1], X_adv[1])


examineSoftmaxOutputs()
sess.close()
