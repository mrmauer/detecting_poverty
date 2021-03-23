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
from torch.utils.data import DataLoader

import copy
import itertools
import numpy as np
import pandas as pd


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
    def __init__(self, latent_dims: int, variational: bool = False, feature_size: Iterable = (1, 28, 28),
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
        current_channels = feature_size[0]
        for l_id in range(len(channels)):
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
                nn.LeakyReLU(),  # activation
            ))
            current_channels = channels[l_id]

        if self.variational:
            self.fc_mu = nn.Sequential(
                nn.Linear(current_channels, latent_dims),
            )
            self.fc_var = nn.Sequential(
                nn.Linear(current_channels, latent_dims),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(current_channels, latent_dims),
            )
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        if self.variational:
            mu = self.fc_mu(x)
            logvar = self.fc_var(x)
            return mu, logvar
        else:
            x = self.fc(x)
            return x


class CNNDecoder(BaseDecoder):
    def __init__(self, latent_dims: int, feature_size: Iterable = (1, 28, 28), channels: list = None, kernel_sizes=None,
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
            activation = nn.LeakyReLU()

        conv_layers = []
        current_channels = feature_size[0]
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
            current_channels = channel

        # reverse layers as constructed in reverse
        self.conv_layers = nn.Sequential(*conv_layers[::-1])
        self.fc_layer = nn.Sequential(
            nn.Linear(latent_dims, int(current_channels)),
        )

    def forward(self, x):
        x = self.fc_layer(x)
        x = x.reshape((x.shape[0], self.conv_layers[0][0].in_channels, -1))
        x = x.reshape(
            (x.shape[0], self.conv_layers[0][0].in_channels, int(sqrt(x.shape[-1])), int(sqrt(x.shape[-1]))))
        x = self.conv_layers(x)
        return x

class DVCCA(nn.Module):
    """
    https: // arxiv.org / pdf / 1610.03454.pdf
    With pieces borrowed from the variational autoencoder implementation @
    # https: // github.com / pytorch / examples / blob / master / vae / main.py
    """

    def __init__(self, latent_dims: int, 
                 encoders: Iterable[BaseEncoder] = (CNNEncoder, CNNEncoder),
                 decoders: Iterable[BaseDecoder] = (CNNDecoder, CNNDecoder),
                 learning_rate=1e-3,
                 encoder_optimizers=None, decoder_optimizers=None,
                 encoder_schedulers=None, decoder_schedulers=None):
        super().__init__()
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
        

    def update_weights(self, *args):
        """
        :param args:
        :return:
        """
        self.encoder_optimizers.zero_grad()
        self.decoder_optimizers.zero_grad()
        loss = self.loss(*args)
        loss.backward()
        self.encoder_optimizers.step()
        self.decoder_optimizers.step()
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


class DeepWrapper():

    def __init__(self, model, device: str = 'cuda', tensorboard_tag=''):
        self.model = model
        self.device = device
        if not torch.cuda.is_available() and self.device == 'cuda':
            self.device = 'cpu'
        self.latent_dims = model.latent_dims

    def fit(self, train_dataset, val_dataset, labels=None, val_split=0.2, batch_size=1, patience=0, epochs=1, train_correlations=True):
        """
        :param views: EITHER 2D numpy arrays for each view separated by comma with the same number of rows (nxp)
                        OR torch.torch.utils.data.Dataset
                        OR 2 or more torch.utils.data.Subset separated by commas
        :param labels:
        :param val_split: the ammount of data used for validation
        :param batch_size: the minibatch size
        :param patience: if 0 train to num_epochs, else if validation score doesn't improve after patience epochs stop training
        :param epochs: maximum number of epochs to train
        :param train_correlations: if True generate training correlations
        :return:
        """
        self.batch_size = batch_size

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        # First we get the model class.
        # These have a forward method which takes data inputs and outputs the variables needed to calculate their
        # respective loss. The models also have loss functions as methods but we can also customise the loss by calling
        # a_loss_function(model(data))
        num_params = sum(p.numel() for p in self.model.parameters())
        print('total parameters: ', num_params)
        best_model = copy.deepcopy(self.model.state_dict())
        self.model.float().to(self.device)
        min_val_loss = torch.tensor(np.inf)
        epochs_no_improve = 0
        early_stop = False
        all_train_loss = []
        all_val_loss = []

        for epoch in range(1, epochs + 1):
            if not early_stop:
                epoch_train_loss = self.train_epoch(train_dataloader)
                print('====> Epoch: {} Average train loss: {:.4f}'.format(
                    epoch, epoch_train_loss))
                epoch_val_loss = self.val_epoch(val_dataloader)
                print('====> Epoch: {} Average val loss: {:.4f}'.format(
                    epoch, epoch_val_loss))
                if epoch_val_loss < min_val_loss or epoch == 1:
                    min_val_loss = epoch_val_loss
                    best_model = copy.deepcopy(self.model.state_dict())
                    print('Min loss %0.2f' % min_val_loss)
                    epochs_no_improve = 0
                if any(self.model.schedulers):
                    for scheduler in self.model.schedulers:
                        try:
                            scheduler.step()
                        except:
                            scheduler.step(epoch_train_loss)
                else:
                    epochs_no_improve += 1
                    # Check early stopping condition
                    if epochs_no_improve == patience and patience > 0:
                        print('Early stopping!')
                        early_stop = True
                        self.model.load_state_dict(best_model)

                # all_train_loss.append(epoch_train_loss)
                # all_val_loss.append(epoch_val_loss)
        #         if self.tensorboard:
        #             self.writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        #             self.writer.add_scalar('Loss/test', epoch_val_loss, epoch)
        # if self.tensorboard:
        #     self.writer.close()
        # cca_zoo.plot_utils.plot_training_loss(all_train_loss, all_val_loss)
        if train_correlations:
            self.train_correlations = self.predict_corr(train_dataset, train=True)
        return self

    def train_epoch(self, train_dataloader: torch.utils.data.DataLoader):
        """
        Train a single epoch
        :param train_dataloader: a dataloader for training data
        :return: average loss over the epoch
        """
        self.model.train()
        train_loss = 0
        for batch_idx, (dataA, dataZ) in enumerate(train_dataloader):
            # data = [d.float().to(self.device) for d in list(data)]
            # loss = self.model.update_weights(*data)
            dataA = dataA.to(self.device)
            dataZ = dataZ.to(self.device)
            loss = self.model.update_weights(dataA, dataZ)
            train_loss += loss.item()
        return train_loss / len(train_dataloader)

        # for batch_idx, (dataA, dataZ) in enumerate(train_loader):
        #     dataA = dataA.to(self.device)
        #     dataZ = dataZ.to(self.device)
        #     optimizer.zero_grad()
        #     recon_batch, mu, logvar = self.model(dataA)
        #     loss = self._loss_function(recon_batch, dataZ, mu, logvar)
        #     loss.backward()
        #     train_loss += loss.item()
        #     optimizer.step()
        # return train_loss/(batch_idx+1)

    def val_epoch(self, val_dataloader: torch.utils.data.DataLoader):
        """
        Validate a single epoch
        :param val_dataloader: a dataloder for validation data
        :return: average validation loss over the epoch
        """
        self.model.eval()
        for param in self.model.parameters():
            param.grad = None
        total_val_loss = 0
        for batch_idx, (data, label) in enumerate(val_dataloader):
            data = [d.float().to(self.device) for d in list(data)]
            loss = self.model.loss(*data)
            total_val_loss += loss.item()
        return total_val_loss / len(val_dataloader)

    def predict_corr(self, *views, train=False):
        """
        :param views: EITHER numpy arrays separated by comma. Each view needs to have the same number of features as its
         corresponding view in the training data
                        OR torch.torch.utils.data.Dataset
                        OR 2 or more torch.utils.data.Subset separated by commas
        :return: numpy array containing correlations between each pair of views for each dimension (#views*#views*#latent_dimensions)
        """
        transformed_views = self.transform(*views, train=train)
        all_corrs = []
        for x, y in itertools.product(transformed_views, repeat=2):
            all_corrs.append(np.diag(np.corrcoef(x.T, y.T)[:self.latent_dims, self.latent_dims:]))
        all_corrs = np.array(all_corrs).reshape(
            (len(transformed_views), len(transformed_views), self.latent_dims))
        return all_corrs

    def transform(self, *views, labels=None, train=False):
        if type(views[0]) is np.ndarray:
            test_dataset = cca_zoo.data.CCA_Dataset(*views, labels=labels)
        elif isinstance(views[0], torch.utils.data.Dataset):
            test_dataset = views[0]
        if self.batch_size > 0:
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size)
        else:
            test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(test_dataloader):
                data = [d.float().to(self.device) for d in list(data)]
                z = self.model(*data)
                if batch_idx == 0:
                    z_list = [z_i.detach().cpu().numpy() for i, z_i in enumerate(z)]
                else:
                    z_list = [np.append(z_list[i], z_i.detach().cpu().numpy(), axis=0) for
                              i, z_i in enumerate(z)]
        # For trace-norm objective models we need to apply a linear CCA to outputs
        if self.model.post_transform:
            if train:
                self.cca = cca_zoo.wrappers.MCCA(latent_dims=self.latent_dims)
                self.cca.fit(*z_list)
                z_list = self.cca.transform(*z_list)
            else:
                z_list = self.cca.transform(*z_list)
        return z_list

    def predict_view(self, *views, labels=None):
        if type(views[0]) is np.ndarray:
            test_dataset = cca_zoo.data.CCA_Dataset(*views, labels=labels)
        elif isinstance(views[0], torch.utils.data.Dataset):
            test_dataset = views[0]

        test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(test_dataloader):
                data = [d.float().to(self.device) for d in list(data)]
                x = self.model.recon(*data)
                if batch_idx == 0:
                    x_list = [x_i.detach().cpu().numpy() for i, x_i in enumerate(x)]
                else:
                    x_list = [np.append(x_list[i], x_i.detach().cpu().numpy(), axis=0) for
                              i, x_i in enumerate(x)]
        return x_list






