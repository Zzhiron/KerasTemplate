import tensorflow as tf
from tensorflow.keras.utils import plot_model


class BaseModel(tf.keras.models.Model):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model = None


    def call(self, input):
        raise NotImplementedError


    def build_model(self, loss, metric_list, optimizer):
        self.model.compile(optimizer=optimizer,
            loss=loss,
            metrics=metric_list
            )
        
        return self.model


    @staticmethod
    def print_model_info(model):
        a = 0
        for layer in model.layers:
            print("Layer ", a, " : ", layer.name)
            a += 1

        model.summary()
        print('Real input name is: ', model.input.name)
        print('Real output name is: ', model.output.name)
        plot_model(model, to_file="model_info.png", show_shapes=True)
        print("Model with initial learning rate: ", model.optimizer.lr.numpy())
