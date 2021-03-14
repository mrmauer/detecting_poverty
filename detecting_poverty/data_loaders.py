import torch
import torch.utils.data as tud
import numpy as np
import geoio
from utils import create_space
import os
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


class ToTensor(object):
    def __init__(self, bands, height, width):
        self.x = width
        self.y = height
        self.z = bands

    def __call__(self, sample):
        m = torch.from_numpy(sample).type(torch.float).reshape(
            (self.z, self.y, self.x))
        return m

class MNIST(tud.Dataset):
    def __init__(self, X, transform=ToTensor):
        self.X = X
        self.transform = transform(1, 28, 28)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return (x, x)

class LandsatViirs(tud.Dataset):
    """
    A data loader that uses Google Earth Engine to sample pairs of Landsat
    and VIIRS images for matching land areas. 
    """
    def __init__(
            self, df, landsat_transform, viirs_transform
        ):
        self.df = df[['image_name', 'image_lat', 'image_lon']]
        self.landsat_transform = landsat_transform
        self.viirs_transform = viirs_transform

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        lat, lon = self.df.loc[idx, ['image_lat', 'image_lon']]
        img, country = self.df.loc[idx, ['image_name', 'country']]
        if self.transform:
            landsat = self.landsat_transform((img, country))
            viirs = self.viirs_transform((lat, lon))
        return landsat, viirs

class LandsatTransform:
    """
    A callable object that, given a pair of coordinates, returns a the 
    Landsat image formatted as a 3D Tensor [bands, height, width].
    """
    def __init__(self, base_path, width=256, height=256):
        self.base_path = base_path
        self.width = width
        self.height = height

    def __call__(self, image_name, country='ng'):
        path = '/'.join([self.base_path, country, image_name])
        im = Image.open(path) # use pillow to open a file
        img = img.resize((self.width, self.height)) # resize the file to 256x256
        # img = img.convert('RGB') #convert image to RGB channel
        # img = np.asarray(img).transpose(-1, 0, 1) 
        # ^^^ we have to change the dimensions from width x height x channel 
        # (WHC) to channel x width x height (CWH)
        # img = torch.from_numpy(np.asarray(img)/255)
        return pil_to_tensor(img)

class ViirsTransform:
    """
    A callable object that, given a pair of coordinates, returns a the 
    VIIIRS Day/Night Band image formatted as a 3D Tensor [bands, height, width].
    """
    def __init__(self, tif):
        """
        Inputs:
            tif (geoio.GeoImage)
        """
        self.tif = tif
        self.data = tig.get_data()

    def __call__(self, coord):
        """
        Input:
            coord (tuple of 2 floats)

        Returns a 3D tensor.
        """
        min_lat, min_lon, max_lat, max_lon = create_space(
            coord[0], coord[1])
        xminPixel, ymaxPixel = self.tif.proj_to_raster(min_lon, min_lat)
        # xmaxPixel, yminPixel = self.tif.proj_to_raster(max_lon, max_lat)
        if (xminPixel<0) or (ymaxPixel-21<0) or \
                (ymaxPixel>self.data.shape[0]) or \
                (xminPixel+21>self.data.shape[1]):
            return False, None

        array = self.data[ymaxPixel-21:ymaxPixel,xminPixel:xminPixel+21]
        return True, torch.tensor(array)






