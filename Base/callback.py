import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard


''' 
recommended format for using callbacks

# call function here
def your_callback_name():
    return real_callback_classname

# implement function class here
class real_callback_classname(Callback)
    def __init__(self):
        # TODO here
    def on_epoch_begin(self, epoch, logs=None):
        # TODO here
'''

def reduce_lr():
    return ReduceLROnPlateau


def early_stopping():
    return EarlyStopping


def model_checkpoint():
    return ModelCheckpoint


def tensorboard():
    return TensorBoard


# customed callbacks code copied from tensorflow web page
class LRScheduler(Callback):

    def __init__(self):
        super(LRScheduler, self).__init__()
        self.schedule = LRScheduler.lr_schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        scheduled_lr = self.schedule(epoch, lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))

    @staticmethod
    def lr_schedule(epoch, lr):
        LR_SCHEDULE = [
            # (epoch to start, learning rate) tuples
            (3, 0.05),
            (6, 0.01),
            (9, 0.005),
            (12, 0.001),
        ]

        if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
            return lr
        for i in range(len(LR_SCHEDULE)):
            if epoch == LR_SCHEDULE[i][0]:
                return LR_SCHEDULE[i][1]
        return lr




