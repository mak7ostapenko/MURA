from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.layers import Input, Dense
from keras.callbacks import (ModelCheckpoint, TensorBoard)



class TuneModel():
    model_dict = {'resnet': ResNet50, 'densenet': DenseNet121}

    def __init__(self, base_model_name, num_classes, height, width, weights='imagenet', include_top=False):
        self.name = base_model_name
        self.num_classes = num_classes
        self.height = height
        self.width = width
        self.include_top = include_top
        self.weights = weights
        self.start_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z')


    def tune_architecture(self, base_model, tuning_params=None):
        """
        Tune base model architecture

        Arguments:
             model - base model for tuning,
             tuning_params - ...

        Returns:
           Tuned model with new layers

        """
        x = base_model.output
        predictions = Dense(self.num_classes, activation='sigmoid', name='predictions')(x)
        tuned_model = Model(base_model.input, predictions)

        return tuned_model


    def callbacks(self):
        """

        Arguments:


        Returns:

        """
        checkpoint = ModelCheckpoint(filepath='./checkpoint/{}.hdf5'.format(self.name),
                                     verbose=1, seve_best_only=True)
        tensor_board = TensorBoard(log_dir='./logs/{}/{}/'.format(self.name,
                                                                 self.start_time))

        return [checkpoint, tensor_board]


    def get_datagen(self, params):
        """

        Arguments:


        Returns:

        """
        datagen_param = params.datagen
        gen_param = params.generator
        datagen = ImageDataGenerator(**datagen_param)
        generator = datagen.flow_from_directory(**gen_param)

        return generator


    def train(self):
        # GET CALLBACKS
        callbacks = self.callbacks()


    def build(self, train_param):
        """

        Arguments:


        Returns:

        """
        # WORK WITH DATA
        train_gen_params = train_param.datagenerator.train
        val_gen_param = train_param.datagenerator.val
        train_generator = self.get_datagen(train_gen_params)
        val_generator = self.get_datagen(val_gen_param)

        # BUILD ARCHITECTURE
        base_model = model_dict[self.name]
        input_tensor = Input(shape=(self.height, self.width, 3))
        base_model = base_model(input_tensor=input_tensor,
                                     weights=self.weights,
                                     include_top=include_top)
        self.tuned_model = self.tune_architecture(base_model)

        # GET WEIGHT if model has already trained before
        if self.model_trained:
            self.tuned_model.load_weights(filepath='./checkpoint/{}.hdf5'.format(self.name))
            for layer in self.tuned_model.layers:
                layer.set_trainable = True

        # COMPILE
        self.tuned_model.compile(
            #.............
        )





