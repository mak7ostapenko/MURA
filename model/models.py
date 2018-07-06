from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.models import Model
from tf.contrib.keras.optimizers import Adam, RMSprop
from keras.layers import Input, Dense
from keras.callbacks import (ModelCheckpoint, TensorBoard)
from keras.optimizer import Adam
from sklearn.utils import class_weight
import numpy as np
from utils import get_datagen,class_weights
from mura import Mura


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
        tensor_board = TensorBoard(log_dir='./logs/{}/{}/'.format(self.name, self.start_time))
        return [checkpoint, tensor_board]

    def train(self, train_params):
        """
        Train model

        Arguments:
            train_params - dictionary of parameters for training

        Returns:
            History of model training

        """
        callbacks = self.get_callbacks()
        weights = class_weight(self.train_generator.classes)

        train_history = self.tuned_model.fit_generator(
            generator=self.train_generator,
            steps_per_epoch=len(self.train_generator),
            epoch=2,
            verbose=1,
            callbacks=callbacks,
            validation_data=self.val_generator,
            validation_sptes=len(self.val_generator),
            class_weight=weights,
        )
        return train_history

    def build(self, build_params):
        """
        Build and prepare a model that will be ready for training

        Arguments:
            build_params - dictionary of parameters for model building

        Returns:

        """
        model_dict = {'resnet': ResNet50, 'densenet': DenseNet121}

        # WORK WITH DATA
        train_gen_params = build_params.datagenerator.train
        val_gen_param = build_params.datagenerator.val
        self.train_generator = get_datagen(train_gen_params)
        self.val_generator = get_datagen(val_gen_param)

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

    def eval(self, eval_params):
        """
        Use trained model for evaluation with default metrics
        and custom metrics from the MURA research paper 1712.06957.pdf

        Arguments:
            eval_params - dictionary of parameters for evaluation

        Returns:
            Results of evaluation

        """
        eval_generator = self.get_datagen(eval_params)

        score, accuracy = self.tuned_model.evaluate_generator(
            generator=eval_generator,
            steps=len(eval_generator),
            verbose=1
        )
        print("Loss: ", score)
        print("Accuracy: ", accuracy)

        y_pred = tuned_model.predict_generator(generator=eval_generator,
                                               steps=len(eval_generator))
        mura = Mura(eval_generator.filenames, y_true=eval_generator.classes, y_pred=y_pred)
















