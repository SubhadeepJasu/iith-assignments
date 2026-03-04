"""
Shallow Neural Network
"""
# Author: Subhadeep Jasu

from typing import Any
from keras.models import Sequential
from keras.layers import Dense, Input

class ShallowNN():
    """
    A single hidden layer neural network.
    """

    def __init__(
            self,
            shape:tuple,
            hidden_activation:str,
            hidden_units:int,
            n_classes:int,
            optimizer='adam'
        ):
        self.n_classes = n_classes
        self.nn_model = Sequential()

        # Input layer
        self.nn_model.add(Input(shape=shape))

        # One hidden layer
        self.nn_model.add(Dense(hidden_units, activation=hidden_activation))

        # Output layer
        self.nn_model.add(Dense(n_classes, activation='softmax'))

        self.nn_model.summary()
        self.nn_model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )


    def fit(self, x:Any, y:Any, epochs:int, validation_data:Any|None):
        """
        Train the model for the given number of epochs.
        """
        return self.nn_model.fit(x, y, epochs=epochs, validation_data=validation_data)


    def evaluate(self, x_test, y_test):
        """
        Test the model against test data to find accuracy and loss.
        """
        result = self.nn_model.evaluate(x_test, y_test)
        return {
            'accuracy': result[1],
            'loss': result[0]
        }


    def predict(self, x, batch_size:int):
        """
        Generate output prediction for input samples.
        """
        return self.nn_model.predict(x, batch_size)
