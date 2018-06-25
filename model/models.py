from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.layers import Input, Dense



class Model():
    def __init__(self, model_name, num_classes, height, width, weights='imagenet', include_top=False):
        self.model_name = model_name
        self.num_classes = num_classes
        self.height = height
        self.width = width
        self.include_top = include_top
        self.weights = weights


    def tune_architecture(self, model, model_input):
        """
        Tune base model architecture

        Arguments:
             model - base model for tuning,
             model_input - input tensor in model

        Returns:
           Tuned model with new layers

        """
        x = model.output
        predictions = Dense(self.num_clasess, activation='sigmoid', name='predictions')(x)
        tuned_model = Model(model_input, predictions)

        return tuned_model


    def get_model(self):
        """
        Create model for training

        Arguments:

        Returns:
            Tuned model that are ready for training

        """
        input_tensor = Input(shape=(self.height, self.width, 3))

        if (self.name == 'resnet'):
            base_model = ResNet50(input_tensor=input_tensor,
                                  weights=self.weights,
                                  include_top=include_top)
            model =self.tune_architecture(base_model, input_tensor)

        elif (self.name == 'densenet'):
            base_model = DenseNet121(input_tensor=input_tensor,
                                     weights=weight,
                                     include_top=include_top)
            model = self.tune_architecture(base_model, input_tensor)

        else:
            raise ValueError('Unknown model name')

        return model


    def train(self):
        """

        Arguments:


        Returns:

        """
        pass


    def validate(self):
        """

        Arguments:


        Returns:

        """
        pass









