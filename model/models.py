from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.models import Model
from tf.contrib.keras.optimizers import Adam, RMSprop
from keras.layers import Input, Dense
from keras.callbacks import (ModelCheckpoint, TensorBoard)
from keras.optimizer import Adam
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator



class TuneModel:
    def __init__(self, base_model_name, num_classes, height, width, weights='imagenet', include_top=False):
        self.name = base_model_name
        self.num_classes = num_classes
        self.height = height
        self.width = width
        self.include_top = include_top
        self.weights = weights
        self.start_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z')


    def tune_architecture(self, base_model):
        """
        Tune base model architecture

        Arguments:
             model - base model for tuning,

        Returns:
           Tuned model with new layers

        """
        x = base_model.output
        predictions = Dense(self.num_classes, activation='sigmoid', name='predictions')(x)
        tuned_model = Model(base_model.input, predictions)
        
        return tuned_model


    def get_callbacks(self):
        """
        Create and setting up callbacks for training

        Arguments:

        Returns:
            list of callbacks for using during training

        """
        checkpoint = ModelCheckpoint(filepath='./checkpoint/{}.hdf5'.format(self.name),
                                     verbose=1, seve_best_only=True)
        tensor_board = TensorBoard(log_dir='./logs/{}/{}/'.format(self.name,
                                                                 self.start_time))
        return [checkpoint, tensor_board]


    def get_datagen(self, params):
        """
        Create data generator

        Arguments:
            params - dictionary of parameters for setting up generator

        Returns:
            data generator from directory

        """
        datagen_param = params.datagen
        gen_param = params.generator
        datagen = ImageDataGenerator(**datagen_param)
        generator = datagen.flow_from_directory(**gen_param)

        return generator


    def train(self, train_params=None):
        """
        Train model

        Arguments:
            train_params - dictionary of parameters for training

        Returns:
            History of model training

        """
        # GET CALLBACKS
        callbacks = self.get_callbacks()

        # Compute a class weights
        unique, counts = np.unique(self.train_generator.classes, return_counts=True)
        weight = dict(zip(unique, counts / np.sum(counts)))

        train_history = self.tuned_model.fit_generator(
            generator=self.train_generator,
            steps_per_epoch=len(self.train_generator),
            epoch=2,
            verbose=1,
            callbacks=callbacks,
            validation_data=self.val_generator,
            validation_sptes=len(self.val_generator),
            class_weight=weight,
        )
        return train_history


    def build(self, train_param):
        """
        Build and prepare a model that will be ready for training

        Arguments:
            train_param - dictionary of parameters for training

        Returns:

        """
        model_dict = {'resnet': ResNet50, 'densenet': DenseNet121}

        # WORK WITH DATA
        train_gen_params = train_param.datagenerator.train
        val_gen_param = train_param.datagenerator.val
        self.train_generator = self.get_datagen(train_gen_params)
        self.val_generator = self.get_datagen(val_gen_param)

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
            optimizer=Adam(lr=0.001, decay=0.1),
            loss=binary_corssentropy,
            metrics=['binary_accuracy']
        )





