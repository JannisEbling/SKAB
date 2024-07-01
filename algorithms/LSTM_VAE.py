import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, random_split


class LSTM_VAE(nn.Module):
    """
    A reconstruction LSTM variational autoencoder model to detect anomalies in timeseries data using reconstruction error as an anomaly score.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Examples
    -------
    >>> from LSTM_VAE import LSTM_VAE
    >>> model = LSTM_VAE()
    >>> model.fit(train_data)
    >>> predictions = model.predict(test_data)
    """

    def __init__(self, input_dim, timesteps, intermediate_dim, latent_dim):
        super(LSTM_VAE, self).__init__()

        self.input_dim = input_dim
        self.timesteps = timesteps
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim

        self.encoder_lstm = nn.LSTM(input_dim, intermediate_dim, batch_first=True)
        self.z_mean = nn.Linear(intermediate_dim, latent_dim)
        self.z_log_sigma = nn.Linear(intermediate_dim, latent_dim)

        self.decoder_lstm = nn.LSTM(latent_dim, intermediate_dim, batch_first=True)
        self.decoder_mean = nn.LSTM(intermediate_dim, input_dim, batch_first=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.LSTM):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reparameterize(self, z_mean, z_log_sigma):
        epsilon = torch.randn(z_mean.size(0), z_mean.size(1)).to(z_mean.device)
        return z_mean + torch.exp(z_log_sigma / 2) * epsilon

    def forward(self, x):
        h, _ = self.encoder_lstm(x)
        h = h[:, -1, :]  # Take the last hidden state
        z_mean = self.z_mean(h)
        z_log_sigma = self.z_log_sigma(h)
        z = self.reparameterize(z_mean, z_log_sigma)

        z = z.unsqueeze(1).repeat(1, self.timesteps, 1)  # Repeat z to match timesteps
        h_decoded, _ = self.decoder_lstm(z)
        x_decoded_mean, _ = self.decoder_mean(h_decoded)
        return x_decoded_mean, z_mean, z_log_sigma

    def loss_function(self, x, x_decoded_mean, z_mean, z_log_sigma):
        recon_loss = nn.MSELoss()(x_decoded_mean, x)
        kl_loss = -0.5 * torch.mean(
            1 + z_log_sigma - z_mean**2 - torch.exp(z_log_sigma)
        )
        return recon_loss + kl_loss

    def fit(
        self,
        data,
        epochs=20,
        validation_split=0.1,
        batch_size=1,
        early_stopping=True,
        patience=5,
    ):
        """
        Train the LSTM variational autoencoder model on the provided data.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for training.
        epochs : int, optional
            Number of training epochs (default is 20).
        validation_split : float, optional
            Fraction of the training data to be used as validation data (default is 0.1).
        batch_size : int, optional
            Batch size for training (default is 1).
        early_stopping : bool, optional
            Whether to use early stopping during training (default is True).
        """
        data = torch.tensor(data, dtype=torch.float32)
        dataset = TensorDataset(data, data)
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.RMSprop(self.parameters(), lr=0.001)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.train()
            train_loss = 0
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                x_decoded_mean, z_mean, z_log_sigma = self(batch_x)
                loss = self.loss_function(batch_x, x_decoded_mean, z_mean, z_log_sigma)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            val_loss = 0
            self.eval()
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(self.device)
                    x_decoded_mean, z_mean, z_log_sigma = self(batch_x)
                    loss = self.loss_function(
                        batch_x, x_decoded_mean, z_mean, z_log_sigma
                    )
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            if early_stopping:
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping")
                    break

    def predict(self, data):
        """
        Generate predictions using the trained LSTM variational autoencoder model.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for making predictions.

        Returns
        -------
        predictions : numpy.ndarray
            The reconstructed output predictions.
        """
        data = torch.tensor(data, dtype=torch.float32).to(self.device)
        self.eval()
        with torch.no_grad():
            x_decoded_mean, _, _ = self(data)
        return x_decoded_mean.cpu().numpy()


# from tensorflow.keras.layers import Input, Dense, Lambda, LSTM, RepeatVector
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras import backend as K
# from tensorflow.keras import losses
# from tensorflow.python.framework.ops import disable_eager_execution

# disable_eager_execution()


# class LSTM_VAE:
#     """
#     A reconstruction LSTM variational autoencoder model to detect anomalies in timeseries data using reconstruction error as an anomaly score.

#     Parameters
#     ----------
#     TenserFlow_backend : bool, optional
#         Flag to specify whether to use TensorFlow backend (default is False).

#     Attributes
#     ----------
#     None

#     Examples
#     -------
#     >>> from LSTM_VAE import LSTM_VAE
#     >>> model = LSTM_VAE()
#     >>> model.fit(train_data)
#     >>> predictions = model.predict(test_data)
#     """

#     def __init__(self):
#         pass

#     def _build_model(self, input_dim, timesteps, intermediate_dim, latent_dim):

#         self._Random(0)

#         x = Input(
#             shape=(
#                 timesteps,
#                 input_dim,
#             )
#         )

#         h = LSTM(intermediate_dim)(x)

#         self.z_mean = Dense(latent_dim)(h)
#         self.z_log_sigma = Dense(latent_dim)(h)

#         z = Lambda(self.sampling, output_shape=(latent_dim,))(
#             [self.z_mean, self.z_log_sigma]
#         )

#         decoder_h = LSTM(intermediate_dim, return_sequences=True)
#         decoder_mean = LSTM(input_dim, return_sequences=True)

#         h_decoded = RepeatVector(timesteps)(z)
#         h_decoded = decoder_h(h_decoded)

#         x_decoded_mean = decoder_mean(h_decoded)

#         vae = Model(x, x_decoded_mean)

#         encoder = Model(x, self.z_mean)

#         decoder_input = Input(shape=(latent_dim,))

#         _h_decoded = RepeatVector(timesteps)(decoder_input)
#         _h_decoded = decoder_h(_h_decoded)

#         _x_decoded_mean = decoder_mean(_h_decoded)
#         generator = Model(decoder_input, _x_decoded_mean)

#         vae.compile(optimizer="rmsprop", loss=self.vae_loss)

#         return vae, encoder, generator

#     def _Random(self, seed_value):

#         import os

#         os.environ["PYTHONHASHSEED"] = str(seed_value)

#         import random

#         random.seed(seed_value)

#         import numpy as np

#         np.random.seed(seed_value)

#         import tensorflow as tf

#         tf.random.set_seed(seed_value)

#     def sampling(self, args):
#         """
#         Sample from the latent space using the reparameterization trick.

#         Parameters
#         ----------
#         args : list
#             List of tensors [z_mean, z_log_sigma].

#         Returns
#         -------
#         z : tensorflow.Tensor
#             Sampled point in the latent space.
#         """
#         z_mean, z_log_sigma = args
#         epsilon = K.random_normal(
#             shape=(self.batch_size, self.latent_dim), mean=0.0, stddev=self.epsilon_std
#         )
#         return z_mean + z_log_sigma * epsilon

#     def vae_loss(self, x, x_decoded_mean):
#         """
#         Calculate the VAE loss.

#         Parameters
#         ----------
#         x : tensorflow.Tensor
#             Input data.
#         x_decoded_mean : tensorflow.Tensor
#             Decoded output data.

#         Returns
#         -------
#         loss : tensorflow.Tensor
#             VAE loss value.
#         """
#         mse = losses.MeanSquaredError()
#         xent_loss = mse(x, x_decoded_mean)
#         kl_loss = -0.5 * K.mean(
#             1 + self.z_log_sigma - K.square(self.z_mean) - K.exp(self.z_log_sigma)
#         )
#         loss = xent_loss + kl_loss
#         return loss

#     def fit(
#         self, data, epochs=20, validation_split=0.1, BATCH_SIZE=1, early_stopping=True
#     ):
#         """
#         Train the LSTM variational autoencoder model on the provided data.

#         Parameters
#         ----------
#         data : numpy.ndarray
#             Input data for training.
#         epochs : int, optional
#             Number of training epochs (default is 20).
#         validation_split : float, optional
#             Fraction of the training data to be used as validation data (default is 0.1).
#         BATCH_SIZE : int, optional
#             Batch size for training (default is 1).
#         early_stopping : bool, optional
#             Whether to use early stopping during training (default is True).
#         """

#         self.shape = data.shape
#         self.input_dim = self.shape[-1]
#         self.timesteps = self.shape[1]
#         self.batch_size = BATCH_SIZE
#         self.latent_dim = 100
#         self.epsilon_std = 1.0
#         self.intermediate_dim = 32

#         self.model, self.enc, self.gen = self._build_model(
#             self.input_dim,
#             timesteps=self.timesteps,
#             intermediate_dim=self.intermediate_dim,
#             latent_dim=self.latent_dim,
#         )

#         callbacks = []
#         if early_stopping:
#             callbacks.append(
#                 EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=0)
#             )

#         self.model.fit(
#             data,
#             data,
#             epochs=epochs,
#             batch_size=self.batch_size,
#             validation_split=validation_split,
#             verbose=0,
#             callbacks=callbacks,
#         )

#     def predict(self, data):
#         """
#         Generate predictions using the trained LSTM variational autoencoder model.

#         Parameters
#         ----------
#         data : numpy.ndarray
#             Input data for making predictions.

#         Returns
#         -------
#         predictions : numpy.ndarray
#             The reconstructed output predictions.
#         """

#         return self.model.predict(data)
