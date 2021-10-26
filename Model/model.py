from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model

from Base import BaseModel


class SimpleModel(BaseModel):
    def __init__(self, input_h, input_w, input_c, num_classes):
        super(SimpleModel, self).__init__()

        self.num_classes = num_classes
    
        self.input_data = Input(shape=(input_h, input_w, input_c), name="input")
        self.output_data = self._make_model(self.input_data)
        self.model = Model(self.input_data, self.classifier_output, name="mymodel")

    def call(self, input):
        return self.model(input)

    def _make_model(self, input):
        padding = "same"
        x = Conv2D(16, kernel_size=(3, 3), strides=1, activation='relu', padding=padding)(input)
        x = MaxPooling2D()(x)
        for i in range(1, 4):
            x = Conv2D(16 * (2 ** i), kernel_size=(3, 3), strides=1, activation='relu', padding=padding)(x)
            x = MaxPooling2D()(x)
        x = GlobalMaxPooling2D()(x)
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        predictions = Dense(self.num_classes, activation="sigmoid", name="output")(x)

        return predictions
