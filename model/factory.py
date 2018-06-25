from models import Model

class ModelFactory():
    def __init__(self, model_names, num_classes):
        self.model_names = model_names
        self.num_classes = num_classes


    def add_model(self, name, height, widht, weights, include_top):
        """

        Arguments:
            name - name of a model, which will be created,
            weights - name of a pre-trained weights
            height - the height of input tensor in a model,
            width - the width of input tensor in a model,
            include_top - whether to include the fully-connected layer at the top of the network

        Returns:
            Tuned model that are ready for training

        """
        model = Model(model_name=name,
                      num_classes=self.num_classes,
                      width=widht, height=height,
                      weights=weights,
                      include_top=include_top)
        model = model.get_model()

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
            weights = model_param_dict[name].weights
            height = model_param_dict[name].height
            widht = model_param_dict[name].widht
            include_top = model_param_dict[name].include_top

            model_dict[name] = self.add_model(name, weights, height, widht, include_top)

        return model_dict


    def train_factory(self):
        """

        Arguments:


        Returns:

        """
        pass


    def validate_factory(self):
        """

        Arguments:


        Returns:

        """
        pass