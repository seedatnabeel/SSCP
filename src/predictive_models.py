import logging

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import set_random_seed

from typing import Tuple, List, Dict, Any


logging.getLogger().setLevel(logging.INFO)

# Build a predictive model
class predictive_model:
    def __init__(self, input_shape: Tuple[int, int], epochs: int, batch_size: int, seed: int = 42) -> None:
        set_random_seed(seed)
        self.input_shape = input_shape
        self.pred_model = self._build_model()
        optimizer = keras.optimizers.Adam(lr=0.0005)
        self.pred_model.compile(optimizer=optimizer, loss="mse")
        self.epochs = epochs
        self.batch_size = batch_size

    def _build_model(self) -> Tuple[keras.Model, keras.Model]:
        """
        > We take the input data, pass it through a series of dense layers, and then output a single
        value
        
        Returns:
          The model is being returned.
        """
        
        input_data = keras.Input(shape=(self.input_shape,))

        x = layers.Dense(int(64), activation="relu", name="dense1")(input_data)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(int(64), activation="relu", name="dense2")(x)
        x = layers.Dropout(0.1)(x)
        # "encoded" is the encoded representation of the input
        x = layers.Dense(int(64), activation="relu", name="dense3")(x)
        x = layers.Dropout(0.1)(x)
        output = layers.Dense(1, name="output")(x)

        pred_model = keras.Model(input_data, output)

        return pred_model

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        The function takes in the training data and trains the autoencoder for 100 epochs with a batch
        size of 8
        Args:
          x_train: The training data
        """

        callback = EarlyStopping(monitor="loss", patience=20)
        self.pred_model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[callback],
        )

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Predict function
        Args:
          x_test: The input data
        Returns:
          prediction from model
        """
        return self.pred_model.predict(x_test)

    def extract_encoding(self, x_test):
        """
        > The function takes in a tensor of test data and returns a tensor of the encoding of the data that
         - i.e. feature extraction of an embedding layer
        
        Args:
          x_test (tf.Tensor): the test data
        
        Returns:
          The output of the second dense layer.
        """

        feature_extractor = keras.Model(
            inputs=self.pred_model.inputs,
            outputs=self.pred_model.get_layer(name="dense2").output,
        )

        return feature_extractor.predict(x_test)
