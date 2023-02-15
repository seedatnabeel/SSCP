# stdlib
import logging

# third party
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import set_random_seed
from VIME.vime_self import vime_self

from typing import Tuple, List, Dict, Any


logging.getLogger().setLevel(logging.INFO)


class AutoEncoder:
    def __init__(self, input_shape: Tuple[int, ...], reconstruct_shape: Tuple[int, ...], epochs: int, batch_size: int, seed: int = 42):
        set_random_seed(seed)
        self.input_shape = input_shape
        self.reconstruct_shape = reconstruct_shape
        self.autoencoder = self._build_model()
        optimizer = keras.optimizers.Adam(lr=0.001)
        self.autoencoder.compile(optimizer=optimizer, loss="mse")
        self.epochs = epochs
        self.batch_size = batch_size
        self.pred_model = None

    def _build_model(self):
        """
        > Function to build the autoencoder
        Returns:
          The autoencoder
        """

        input_data = keras.Input(shape=(self.input_shape,))

        decoder2 = layers.Dense(int(self.input_shape) * 2, activation="relu")(
            input_data
        )

        decoder1 = layers.Dense(int(self.input_shape) * 2, activation="relu")(decoder2)

        # "decoded" is the lossy reconstruction of the input
        decoded = layers.Dense(self.reconstruct_shape)(decoder1)

        # Define autoencoder
        autoencoder = keras.Model(input_data, decoded)

        return autoencoder

    def fit(self, x_train: np.ndarray, pred_model) -> None:
        """
        > The function takes in the training data and trains the autoencoder
        Args:
          x_train: The training data
          pred_model: The model used to extract the encoding
        """
        self.pred_model = pred_model
        decoder_input = self.pred_model.extract_encoding(x_train)
        callback = EarlyStopping(monitor="loss", patience=5)
        self.autoencoder.fit(
            decoder_input,
            x_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[callback],
        )

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        > Predict function
        Args:
          x_test: The input data
        Returns:
          predicted data
        """
        decoder_input = self.pred_model.extract_encoding(x_test)
        x_pred = self.autoencoder.predict(decoder_input)

        mse = ((x_test - x_pred) ** 2).mean(axis=1)
        return mse


class Vime_Task:
    def __init__(self, p_m: float = 0.3, alpha: float = 2, batch_size: int = 128, epochs: int = 10, seed: int = 42):
        set_random_seed(seed)
        self.p_m = p_m  # probability of corruption
        self.alpha = alpha  # control the weights of feature and mask losses
        self.batch_size = batch_size
        self.epochs = epochs
        self.vime_self_encoder = None

    def fit(self, X: np.ndarray) -> None:
        """
        > The function takes in the training data and train VIME
        Args:
          X: The training data
        """
        vime_self_parameters = dict()
        vime_self_parameters["batch_size"] = self.batch_size
        vime_self_parameters["epochs"] = self.epochs
        vime_self_encoder = vime_self(X, self.p_m, self.alpha, vime_self_parameters)
        self.vime_self_encoder = vime_self_encoder

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        > Predict function
        Args:
          X: input test data
        Returns:
          predicted mse
        """
        preds = self.vime_self_encoder.predict(X)
        mse = ((preds - X) ** 2).mean(axis=1)
        return mse
