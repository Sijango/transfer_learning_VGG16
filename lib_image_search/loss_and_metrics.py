import tensorflow as tf
import tensorflow.keras.backend as K


def custom_loss(y_true, y_pred):
    mse = tf.losses.mean_squared_error(y_true, y_pred)
    iou = mean_iou(y_true, y_pred)
    custom_loss.__name__ = 'custom_loss'
    return mse + (1 - iou)


def iou(y_true, y_pred, label: int):
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())

    intersection = K.sum(y_true * y_pred)

    union = K.sum(y_true) + K.sum(y_pred) - intersection

    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def mean_iou(y_true, y_pred):
    num_labels = K.int_shape(y_pred)[-1]
    total_iou = K.variable(0)

    for label in range(num_labels):
        total_iou = total_iou + iou(y_true, y_pred, label)
    mean_iou.__name__ = 'mean_iou'
    return total_iou / num_labels
