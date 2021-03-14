import torch
import torch.utils.data as tud
import numpy as np
import geoio
from utils import create_space
import os
from PIL import Image
import torchvision.transforms.functional as TF


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
        self.df = df
        self.landsat_transform = landsat_transform
        self.viirs_transform = viirs_transform
        self.idxs = df.index.to_list()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx = self.idxs[idx]
        cols = ['image_lat', 'image_lon', 'image_name', 'country']
        lat, lon, img, country = self.df.loc[idx, cols]
        landsat = self.landsat_transform(img, country)
        viirs = self.viirs_transform((lat, lon), country)
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
        path = '/'.join([self.base_path, country, 'images copy', image_name])
        img = Image.open(path) # use pillow to open a file
        img = img.resize((self.width, self.height)) # resize the file to 256x256
        # img = img.convert('RGB') #convert image to RGB channel
        # img = np.asarray(img).transpose(-1, 0, 1) 
        # ^^^ we have to change the dimensions from width x height x channel 
        # (WHC) to channel x width x height (CWH)
        # img = torch.from_numpy(np.asarray(img)/255)
        return TF.pil_to_tensor(img)

class ViirsTransform:
    """
    A callable object that, given a pair of coordinates, returns a the 
    VIIIRS Day/Night Band image formatted as a 3D Tensor [bands, height, width].
    """
    def __init__(self, tifs):
        """
        Inputs:
            tif (geoio.GeoImage)
        """
        self.tifs = {
            'ng' : tifs[1],
            'eth' : tifs[1],
            'mw' : tifs[0]
        }
        tif0_data = tifs[0].get_data()
        tif1_data = tifs[1].get_data()
        self.arrays = {
            'ng' : tif1_data,
            'eth' : tif1_data,
            'mw' : tif0_data
        }

    def __call__(self, coord, country):
        """
        Input:
            coord (tuple of 2 floats)

        Returns a 3D tensor.
        """
        min_lat, min_lon, max_lat, max_lon = create_space(
            coord[0], coord[1])
        xminPixel, ymaxPixel = self.tifs[country].proj_to_raster(min_lon, min_lat)
        # xmaxPixel, yminPixel = self.tif.proj_to_raster(max_lon, max_lat)
        # if (xminPixel<0) or (ymaxPixel-21<0) or \
        #         (ymaxPixel>self.data.shape[0]) or \
        #         (xminPixel+21>self.data.shape[1]):
        #     return False, None

        array = self.arrays[country][ymaxPixel-21:ymaxPixel,xminPixel:xminPixel+21]
        return torch.tensor(array.reshape((-1,21,21)))






