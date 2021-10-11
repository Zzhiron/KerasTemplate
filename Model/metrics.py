import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import metrics


@tf.function
def acc(y_true, y_pred):
    # y_true = K.cast(y_true, y_pred.dtype)
    # return K.mean(y_pred)
    return metrics.sparse_categorical_accuracy(y_true, y_pred)
