import torch
import torch.utils.data as tud
import numpy as np
import geoio
from utils import create_space


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
            self, coords, landsat_transform, viirs_transform
        ):
        self.coords = coords
        self.landsat_transform = landsat_transform
        self.viirs_transform = viirs_transform

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        if self.transform:
            landsat = self.landsat_transform(coord)
            viirs = self.viirs_transform(coord)
        return landsat, viirs

class LandsatTransform:
    """
    A callable object that, given a pair of coordinates, returns a the 
    Landsat image formatted as a 3D Tensor [bands, height, width].
    """
    def __init__(self, base_path):
        self.path = base_path

    def __call__(self, coord):
        pass

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






