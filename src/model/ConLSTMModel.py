from demandPre.src.model.model import Model
from demandPre.src import config
from demandPre.src.model.convLstmCelll import BasicConvLSTMCell
import tensorflow as tf
import os


class ConvLstmModel(Model):

    def __init__(self, input_shape, output_shape, learning_rate=0.0002, model_name="ConvLstmModel",
                 model_path=os.path.join(config.dataset_path, "ConvLstmModel")):
        super(ConvLstmModel, self).__init__(input_shape, output_shape, learning_rate=learning_rate, model_name=model_name,
                                            model_path=model_path)

    def build(self):
        super().build()

