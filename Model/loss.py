import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import losses


@tf.function
def cce(y_true, y_pred):
    return losses.sparse_categorical_crossentropy(y_true, y_pred)
    # y_true = K.cast(y_true, y_pred.dtype)
    # return K.mean(K.square(y_pred - y_true), axis=-1)
