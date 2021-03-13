"""
An implementation of Deep Variational Canonical Correlation Analysis
intended specifically for two image datasets that may be the images of different
dimensions. Most code borrowed from James Schapman @ 
https://github.com/jameschapman19/cca_zoo

Based off the following paper:
Wang et al. https://ttic.uchicago.edu/~wwang5/papers/vcca.pdf

Matt Mauer
"""
from abc import abstractmethod
from math import sqrt
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch import optim


class BaseEncoder(nn.Module):
    @abstractmethod
    def __init__(self, latent_dims: int, variational: bool = False):
        super(BaseEncoder, self).__init__()
        self.variational = variational
        self.latent_dims = latent_dims

    @abstractmethod
    def forward(self, x):
        pass


class BaseDecoder(nn.Module):
    @abstractmethod
    def __init__(self, latent_dims: int):
        super(BaseDecoder, self).__init__()
        self.latent_dims = latent_dims

    @abstractmethod
    def forward(self, x):
        pass

class CNNEncoder(BaseEncoder):
    def __init__(self, latent_dims: int, variational: bool = False, feature_size: Iterable = (28, 28),
                 channels: list = None, kernel_sizes: list = None,
                 stride: list = None,
                 padding: list = None):
        super(CNNEncoder, self).__init__(latent_dims, variational=variational)
        if channels is None:
            channels = [1, 1]
        if kernel_sizes is None:
            kernel_sizes = [5] * (len(channels))
        if stride is None:
            stride = [1] * (len(channels))
        if padding is None:
            padding = [2] * (len(channels))
        # assume square input
        conv_layers = []
        current_size = feature_size[0]
        current_channels = 1
        for l_id in range(len(channels) - 1):
            conv_layers.append(nn.Sequential(  # input shape (1, current_size, current_size)
                nn.Conv2d(
                    in_channels=current_channels,  # input height
                    out_channels=channels[l_id],  # n_filters
                    kernel_size=kernel_sizes[l_id],  # filter size
                    stride=stride[l_id],  # filter movement/step
                    padding=padding[l_id],
                    # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if
                    # stride=1
                ),  # output shape (out_channels, current_size, current_size)
                nn.ReLU(),  # activation
            ))
            current_size = current_size
            current_channels = channels[l_id]

        if self.variational:
            self.fc_mu = nn.Sequential(
                nn.Linear(int(current_size * current_size * current_channels), latent_dims),
            )
            self.fc_var = nn.Sequential(
                nn.Linear(int(current_size * current_size * current_channels), latent_dims),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(int(current_size * current_size * current_channels), latent_dims),
            )
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape((x.shape[0], -1))
        if self.variational:
            mu = self.fc_mu(x)
            logvar = self.fc_var(x)
            return mu, logvar
        else:
            x = self.fc(x)
            return x


class CNNDecoder(BaseDecoder):
    def __init__(self, latent_dims: int, feature_size: Iterable = (28, 28), channels: list = None, kernel_sizes=None,
                 strides=None,
                 paddings=None, norm_output: bool = False):
        super(CNNDecoder, self).__init__(latent_dims)
        if channels is None:
            channels = [1, 1]
        if kernel_sizes is None:
            kernel_sizes = [5] * len(channels)
        if strides is None:
            strides = [1] * len(channels)
        if paddings is None:
            paddings = [2] * len(channels)

        if norm_output:
            activation = nn.Sigmoid()
        else:
            activation = nn.ReLU()

        conv_layers = []
        current_channels = 1
        current_size = feature_size[0]
        # Loop backward through decoding layers in order to work out the dimensions at each layer - in particular the first
        # linear layer needs to know B*current_size*current_size*channels
        for l_id, (channel, kernel, stride, padding) in reversed(
                list(enumerate(zip(channels, kernel_sizes, strides, paddings)))):
            conv_layers.append(nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=channel,  # input height
                    out_channels=current_channels,
                    kernel_size=kernel_sizes[l_id],
                    stride=strides[l_id],  # filter movement/step
                    padding=paddings[l_id],
                    # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if
                    # stride=1
                ),
                activation,
            ))
            current_size = current_size
            current_channels = channel

        # reverse layers as constructed in reverse
        self.conv_layers = nn.Sequential(*conv_layers[::-1])
        self.fc_layer = nn.Sequential(
            nn.Linear(latent_dims, int(current_size * current_size * current_channels)),
        )

    def forward(self, x):
        x = self.fc_layer(x)
        x = x.reshape((x.shape[0], self.conv_layers[0][0].in_channels, -1))
        x = x.reshape(
            (x.shape[0], self.conv_layers[0][0].in_channels, int(sqrt(x.shape[-1])), int(sqrt(x.shape[-1]))))
        x = self.conv_layers(x)
        return x

class DCCA_base(nn.Module):
    def __init__(self, latent_dims: int, post_transform=False):
        super(DCCA_base, self).__init__()
        self.latent_dims = latent_dims
        self.post_transform = post_transform
        self.schedulers = [None]

    @abstractmethod
    def update_weights(self, *args):
        """
        A complete update of the weights used every batch
        :param args: batches for each view separated by commas
        :return:
        """
        pass

    @abstractmethod
    def forward(self, *args):
        """
        :param args: batches for each view separated by commas
        :return: views encoded to latent dimensions
        """
        pass

class DVCCA(DCCA_base):
    """
    https: // arxiv.org / pdf / 1610.03454.pdf
    With pieces borrowed from the variational autoencoder implementation @
    # https: // github.com / pytorch / examples / blob / master / vae / main.py
    """

    def __init__(self, latent_dims: int, encoders: Iterable[BaseEncoder] = (CNNEncoder, CNNEncoder),
                 decoders: Iterable[BaseDecoder] = (CNNDecoder, CNNDecoder),
                 private_encoders: Iterable[BaseEncoder] = (CNNEncoder, CNNEncoder), learning_rate=1e-3, private=False,
                 mu=0.5, post_transform=True, encoder_optimizers=None, decoder_optimizers=None,
                 private_encoder_optimizers=None,
                 encoder_schedulers=None, decoder_schedulers=None, private_encoder_schedulers=None):
        super().__init__(latent_dims, post_transform=post_transform)
        self.private = private
        self.mu = mu
        self.latent_dims = latent_dims
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.schedulers = []
        if encoder_schedulers:
            self.schedulers.extend(encoder_schedulers)
        if decoder_schedulers:
            self.schedulers.extend(decoder_schedulers)
        self.encoder_optimizers = encoder_optimizers
        if self.encoder_optimizers is None:
            self.encoder_optimizers = optim.Adam(self.encoders.parameters(), lr=learning_rate)
        self.decoder_optimizers = decoder_optimizers
        if self.decoder_optimizers is None:
            self.decoder_optimizers = optim.Adam(self.decoders.parameters(), lr=learning_rate)
        if private:
            self.private_encoders = nn.ModuleList(private_encoders)
            self.private_encoder_optimizers = private_encoder_optimizers
            if self.private_encoder_optimizers is None:
                self.private_encoder_optimizers = optim.Adam(self.private_encoders.parameters(), lr=learning_rate)
            if private_encoder_schedulers:
                self.schedulers.extend(private_encoder_schedulers)

    def update_weights(self, *args):
        """
        :param args:
        :return:
        """
        self.encoder_optimizers.zero_grad()
        self.decoder_optimizers.zero_grad()
        if self.private:
            self.private_encoder_optimizers.zero_grad()
        loss = self.loss(*args)
        loss.backward()
        self.encoder_optimizers.step()
        self.decoder_optimizers.step()
        if self.private:
            self.private_encoder_optimizers.step()
        return loss

    def forward(self, *args, mle=True):
        """
        :param args:
        :param mle:
        :return:
        """
        # Used when we get reconstructions
        mu, logvar = self.encode(*args)
        if mle:
            z = mu
        else:
            z_dist = dist.Normal(mu, torch.exp(0.5 * logvar))
            z = z_dist.rsample()
        # If using single encoder repeat representation n times
        if len(self.encoders) == 1:
            z = z * len(args)
        if self.private:
            mu_p, logvar_p = self.encode_private(*args)
            if mle:
                z_p = mu_p
            else:
                z_dist = dist.Normal(mu_p, torch.exp(0.5 * logvar_p))
                z = z_dist.rsample()
            z = [torch.cat([z_] + z_p, dim=-1) for z_ in z]
        return z

    def encode(self, *args):
        """
        :param args:
        :return:
        """
        mu = []
        logvar = []
        for i, encoder in enumerate(self.encoders):
            mu_i, logvar_i = encoder(args[i])
            mu.append(mu_i)
            logvar.append(logvar_i)
        return mu, logvar

    def encode_private(self, *args):
        """
        :param args:
        :return:
        """
        mu = []
        logvar = []
        for i, private_encoder in enumerate(self.private_encoders):
            mu_i, logvar_i = private_encoder(args[i])
            mu.append(mu_i)
            logvar.append(logvar_i)
        return mu, logvar

    def decode(self, z):
        """
        :param z:
        :return:
        """
        x = []
        for i, decoder in enumerate(self.decoders):
            x_i = decoder(z)
            x.append(x_i)
        return tuple(x)

    def recon(self, *args):
        """
        :param args:
        :return:
        """
        z = self(*args)
        return [self.decode(z_i) for z_i in z][0]

    def loss(self, *args):
        """
        :param args:
        :return:
        """
        mus, logvars = self.encode(*args)
        if self.private:
            mus_p, logvars_p = self.encode_private(*args)
            losses = [self.vcca_private_loss(*args, mu=mu, logvar=logvar, mu_p=mu_p, logvar_p=logvar_p) for
                      (mu, logvar, mu_p, logvar_p) in
                      zip(mus, logvars, mus_p, logvars_p)]
        else:
            losses = [self.vcca_loss(*args, mu=mu, logvar=logvar) for (mu, logvar) in
                      zip(mus, logvars)]
        return torch.stack(losses).mean()

    def vcca_loss(self, *args, mu, logvar):
        """
        :param args:
        :param mu:
        :param logvar:
        :return:
        """
        batch_n = mu.shape[0]
        z_dist = dist.Normal(mu, torch.exp(0.5 * logvar))
        z = z_dist.rsample()
        kl = torch.mean(-0.5 * torch.sum(1 + logvar - logvar.exp() - mu.pow(2), dim=1), dim=0)
        recons = self.decode(z)
        bces = torch.stack(
            [F.binary_cross_entropy(recon, arg, reduction='sum') / batch_n for recon, arg in
             zip(recons, args)]).sum()
        return kl + bces

    def vcca_private_loss(self, *args, mu, logvar, mu_p, logvar_p):
        """
        :param args:
        :param mu:
        :param logvar:
        :return:
        """
        batch_n = mu.shape[0]
        z_dist = dist.Normal(mu, torch.exp(0.5 * logvar))
        z = z_dist.rsample()
        z_p_dist = dist.Normal(mu_p, torch.exp(0.5 * logvar_p))
        z_p = z_p_dist.rsample()
        kl_p = torch.stack(
            [torch.mean(-0.5 * torch.sum(1 + logvar_p - logvar_p.exp() - mu_p.pow(2), dim=1), dim=0) for
             i, _ in enumerate(self.private_encoders)]).sum()
        kl = torch.mean(-0.5 * torch.sum(1 + logvar - logvar.exp() - mu.pow(2), dim=1), dim=0)
        z_combined = torch.cat([z, z_p], dim=-1)
        recon = self.decode(z_combined)
        bces = torch.stack(
            [F.binary_cross_entropy(recon[i], args[i], reduction='sum') / batch_n for i, _ in
             enumerate(self.decoders)]).sum()
        return kl + kl_p + bces


        