from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.layers import Input, Dense



class ModelFactory():
    def __init__(self, model_names, num_clasess):
        self.model_names = model_names
        self.num_classes = num_clasess


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


    def get_model(self, name, weight='imagenet', height=224, width=224, include_top=False):
        """
        Create model for training

        Arguments:
            name - name of a model, which will be created,
            weight - name of a pre-trained weights
            height - the height of input tensor in a model,
            width - the width of input tensor in a model,
            include_top - whether to include the fully-connected layer at the top of the network

        Returns:
            Tuned model that are ready for training

        """
        input_tensor = Input(shape=(height, width, 3))

        if (name == 'resnet'):
            base_model = ResNet50(input_tensor=input_tensor,
                                  weights=weight,
                                  include_top=include_top)
            model =self.tune_architecture(base_model, input_tensor)

        elif (name == 'densenet'):
            base_model = DenseNet121(input_tensor=input_tensor,
                                     weights=weight,
                                     include_top=include_top)
            model = self.tune_architecture(base_model, input_tensor)

        else:
            raise ValueError('Unknown model name')

        return model



    def get_model_dict(self, model_param_dict):
        """
        Create a model factory

        Arguments:
            model_param_dict - dictionary of parameters for each network in a factory

        Returns:
            Dictionary of networks for training

        """
        model_dict = {}

        for model_index, name in enumerate(self.model_names):
            weight = model_param_dict[name].weight
            height = model_param_dict[name].height
            widht = model_param_dict[name].widht
            include_top = model_param_dict[name].include_top

            model_dict[name] = self.get_model(name, weight, height, widht, include_top)

        return model_dict


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

