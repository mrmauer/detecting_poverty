# NEEDS WORK
    # dimensions of Conv Net running into Errors...

import torch.utils.data as tud
from torch import optim, nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import numpy as np
import torch
from sklearn import decomposition
from numpy.random import permutation as rpm
from data_loaders import MNIST


# the autoencoder network
class CVEDNet(nn.Module):
    def __init__(
            self, latent_dim=64, init_channels=8, kernel_size=4, 
            image_in_channels=1, image_out_channels=1
        ):
        super(CVEDNet, self).__init__()
 
        # encoder
        # H = ceil((H + 2*padding - kernel)/stride + 1)
        # H = 28
        self.enc1 = nn.Conv2d(
            in_channels=image_in_channels, out_channels=init_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        # H = (28 + 2 - 4)/2 + 1 = 14
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        # H = (14 + 2 - 4)/2 + 1 = 7
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        # H = (7 + 2 - 4)/2 + 1 = ceil(3.5) = 4...
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4, out_channels=64, kernel_size=kernel_size-1, 
            stride=2, padding=0
        )
        # H = (4 + 0 - 4)/2 + 1 = 1
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)
        # decoder 
        # output dimensions of each ConvTranspose2d:
        #     H = (H - 1)*stride + kernel_size - 2*padding
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=1, padding=0
        )
        # H = 0 + 3 - 0 = 3
        # H = 0 + 4 - 0 = 4
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        # H = (2)2 + 3 - 2 = 5
        # H = (3)2 + 4 - 2 = 8
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        # H = (4)2 + 3 - 2 = 9
        # H = (7)2 + 2 = 16
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=image_out_channels, kernel_size=kernel_size, 
            stride=2, padding=3
        )
        # H = (8)2 + 3 - 2 = 17
        # H = (15)2 + 4 - 2(padding=3) = 28

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x):
        # encoding
        x = F.leaky_relu(self.enc1(x))
        x = F.leaky_relu(self.enc2(x))
        x = F.leaky_relu(self.enc3(x))
        x = F.leaky_relu(self.enc4(x))
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        z = z.view(-1, 64, 1, 1)
 
        # decoding
        x = F.leaky_relu(self.dec1(z))
        x = F.leaky_relu(self.dec2(x))
        x = F.leaky_relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))
        return reconstruction, mu, log_var

class ConvVED(object):
    """Convolutional Variational Encoder Decoder

    Parameters
    ----------
    n_inputs: int, feature size of input data
    n_components: int, feature size of output
    lr: float, learning rate (default: 0.001)
    batch_size: int, batch size (default: 128)
    cuda: bool, whether to use GPU if available (default: True)
    path: string, path to save trained model (default: "vae.pth")
    kkl: float, float, weight on loss term -KL(q(z|x)||p(z)) (default: 1.0)
    kv: float, weight on variance term inside -KL(q(z|x)||p(z)) (default: 1.0)
    """
    def __init__(
            self, n_components, lr=1.0e-3, batch_size=16, data_loader=MNIST,
            cuda=True, path="vae.pth", kkl=1.0, kv=1.0, init_channels=8, 
            kernel_size=4, image_in_channels=1, image_out_channels=1
        ):
        self.model = CVEDNet(
            latent_dim=n_components,
            init_channels=init_channels, 
            kernel_size=kernel_size,
            image_in_channels=image_in_channels,
            image_out_channels=image_out_channels
        )
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = batch_size
        self.lr = lr
        self.path = path
        self.kkl = kkl
        self.kv = kv
        self.data_loader = data_loader
        self.initialize()

    def fit(self, Xr, Xd, epochs):
        """Fit ConvVED from data Xr
        Parameters
        ----------
        :in:
        Xr: 2d array of shape (n_data, n_dim). Training data
        Xd: 2d array of shape (n_data, n_dim). Dev data, used for early stopping
        epochs: int, number of training epochs
        """
        train_loader = tud.DataLoader(
            self.data_loader(Xr),
            batch_size=self.batch_size, shuffle=True)
        dev_loader = tud.DataLoader(
            self.data_loader(Xd),
            batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        best_dev_loss = np.inf
        for epoch in range(1, epochs + 1):
            train_loss = self._train(train_loader, optimizer)
            dev_loss, _ = self._evaluate(dev_loader)
            if dev_loss < best_dev_loss:
                torch.save(self.model, self.path)
            print('Epoch: %d, train loss: %.4f, dev loss: %.4f' % (
                epoch, train_loss, dev_loss))
        return

    def transform(self, X):
        """Transform X
        Parameters
        ----------
        :in:
        X: 2d array of shape (n_data, n_dim)
        :out:
        Z: 2d array of shape (n_data, n_components)
        """
        try:
            self.model = torch.load(self.path)
        except Exception as err:
            print("Error loading '%s'\n[ERROR]: %s\nUsing initial model!" % (self.path, err))
        test_loader = tud.DataLoader(
            self.data_loader(X), batch_size=self.batch_size, shuffle=False)
        _, Z = self._evaluate(test_loader)
        return Z

    def _train(self, train_loader, optimizer):
        self.model.train()
        train_loss = 0
        for batch_idx, (dataA, dataZ) in enumerate(train_loader):
            dataA = dataA.to(self.device)
            dataZ = dataZ.to(self.device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(dataA)
            loss = self._loss_function(recon_batch, dataZ, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        return train_loss/(batch_idx+1)

    def _evaluate(self, loader):
        self.model.eval()
        loss = 0
        fs = []
        with torch.no_grad():
            for batch_idx, (dataA, dataZ) in enumerate(loader):
                dataA = dataA.to(self.device)
                dataZ = dataZ.to(self.device)
                recon_batch, mu, logvar = self.model(dataA)
                loss += self._loss_function(recon_batch, dataZ, mu, logvar).item()
                fs.append(mu)
        fs = torch.cat(fs).cpu().numpy()
        return loss/(batch_idx+1), fs

    def _loss_function(self, recon_x, x, mu, logvar):
        """VAE Loss
        Parameters
        ----------
        :in:
        recon_x: 2d tensor of shape (batch_size, n_dim), reconstructed input
        x: 2d tensor of shape (batch_size, n_dim), input data
        mu: 2d tensor of shape (batch_size, n_components), latent mean
        logvar: 2d tensor of shape (batch_size, n_components), latent log-variance
        :out:
        l: 1d tensor, VAE loss
        """
        n, d = mu.shape
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')/n
        KLD = -0.5*(d + self.kv*(logvar-logvar.exp()).sum()/n - mu.pow(2).sum()/n)
        l = BCE + self.kkl*KLD
        return l

    def initialize(self):
        """
        Model Initialization
        """
        def _init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.model.apply(_init_weights)
        return

def conv_ved_emb(train_subset, dev_subset, train_data, dev_data, test_data, 
            n_components, lr=1.0e-3, batch_size=128,
            epochs=15, kkl=1.0, kv=1.0, path='ConvVED.pt', 
        ):
    """
    Train and extract feature with Convolutional VAE
    ------
    Input
        train_subset, dev_subset: 2d array of shape (n_data, n_dim), VAE 
            training and development data
        train_data, dev_data, test_data: 2d array of shape (n_data, n_dim), 
            training/dev/test set of the dataset, where trained VAE will be 
            used to extract features
        n_components: int, feature dimension
        lr: float, learning rate (default: 0.001)
        batch_size: int, batch size to train VAE (default: 128)
        epochs: int, training epochs (default: 20)
        kkl: float, weight (lambda_KL) on -KL(q(z|x)||p(z)) (default: 1.0)
        kv: float, weight (lambda_var) on variance term inside -KL(q(z|x)||p(z)) 
            (default: 1.0)
        path: string, path to save trained model (default: "VAE.pt")
    
    Output
        train_features, dev_features, test_features: 2d array of shape 
            (n_data, n_dim), extracted features of the training/dev/test set
    """
    print("Using Convolutional VED")
    model = ConvVED(
        n_components=n_components, 
        lr=lr, 
        batch_size=batch_size, 
        kkl=kkl, 
        kv=kv, 
        path=path
    )
    model.fit(train_subset, Xd=dev_subset, epochs=epochs)
    train_features = model.transform(train_data)
    dev_features = model.transform(dev_data)
    test_features = model.transform(test_data)
    return train_features, dev_features, test_features