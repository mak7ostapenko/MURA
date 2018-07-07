from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.layers import Input, Dense
from keras.callbacks import (ModelCheckpoint, TensorBoard)
import numpy as np
from model.utils import data_gen, class_weights
from model.mura import Mura


class TuneModel:
    def __init__(self, base_model, name, width, height, channels,
                 include_top=False, weights='imagenet', classes=2, model_trained=False):
        self.name = name
        self.classes = classes
        self.height = height
        self.width = width
        self.channels = channels
        self.include_top = include_top
        self.weights = weights
        self.model_trained = model_trained

        input_tensor = Input(shape=(self.height, self.width, self.channels))
        self.base_model = base_model(input_tensor=input_tensor, include_top=self.include_top,
                                     weights=self.weights, classes=self.classes)

        self.start_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z')

    def tune_architecture(self):
        """
        Tune base model architecture

        Arguments:
             model - base model for tuning,

        Returns:
           Tuned model with new layers

        """
        x = self.base_model.output
        predictions = Dense(self.num_classes, activation='sigmoid', name='predictions')(x)
        model = Model(self.base_model.input, predictions)

        return model

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

    def train(self, train_params, gen_params):
        """
        Train model

        Arguments:
            train_params - dictionary of parameters for training

        Returns:
            History of model training

        """
        train_generator = get_datagen(gen_params.train)
        val_generator = get_datagen(gen_params.val)

        callbacks = self.get_callbacks()
        weights = class_weight(self.train_generator.classes)

        train_history = self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=len(train_generator),
            epoch=train_params.epoch,
            verbose=train_params.verbose,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_sptes=len(val_generator),
            class_weight=weights,
        )
        return train_history

    def build(self, build_params):
        """
        Build and prepare a model that will be ready for training

        Arguments:
            build_params -

        Returns:

        """
        # TUNE BASE ARCHITECTURE
        self.model = self.tune_architecture()

        # GET WEIGHT if model has already trained before
        if self.model_trained:
            self.model.load_weights(filepath='./checkpoint/{}.hdf5'.format(self.name))
        for layer in self.model.layers:
            layer.set_trainable = True

        self.model.compile(
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

        score, accuracy = self.model.evaluate_generator(
            generator=eval_generator,
            steps=len(eval_generator),
            verbose=1)

        print("Loss: ", score)
        print("Accuracy: ", accuracy)

        y_pred = self.model.predict_generator(generator=eval_generator,
                                               steps=len(eval_generator))
        mura = Mura(eval_generator.filenames, y_true=eval_generator.classes, y_pred=y_pred)

        metrics = mura.metrics()
        metrics_by_encounter = mura.metrics_by_encounter()
        metrics_by_study_type = mura.metrics_by_study_type()

        return metrics, metrics_by_encounter, metrics_by_study_type
















