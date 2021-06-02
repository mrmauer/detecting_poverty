import torch
import torch.utils.data as tud
import numpy as np
import geoio
from utils import create_space
import os
from PIL import Image, ImageFile
import torchvision.transforms.functional as TF
from collections import defaultdict
import rasterio
import pyproj

ImageFile.LOAD_TRUNCATED_IMAGES = True

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

class Landsat(tud.Dataset):
    """
    A data loader that samples pairs of Landsat images.
    """
    def __init__(self, df, landsat_transform):
        self.df = df
        self.landsat_transform = landsat_transform
        self.idxs = df.index.to_list()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx = self.idxs[idx]
        img, country = self.df.loc[idx, ['image_name', 'country']]
        landsat = self.landsat_transform(img, country)
        return landsat, landsat


class LandsatViirs(tud.Dataset):
    """
    A data loader that samples pairs of Landsat
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
        cols = ['# lon', 'lat', 'path', 'row', 'B2_link', 'B3_link', 'B4_link']
        lon, lat, path, row, B2_url, B3_url, B4_url = self.df.loc[idx, cols]
        pathrow = str(path) + str(row)
        landsat = self.landsat_transform((lon, lat), pathrow,
                                         (B2_url, B3_url, B4_url))
        viirs = self.viirs_transform((lat, lon), country)
        return landsat, viirs

class LandsatTransform:
    """
    A callable object that, given a pair of coordinates, returns the
    Landsat image formatted as a 3D Tensor [bands, height, width].
    """
    def __init__(self):
        """
        Inputs: None
        """
        self.pathrow_imgs = defaultdict(0)

    def _update_dict(pathrow, img_tensor):
        """
        Given a path-row and a 3d tensor representing an image for a single
        pathrow scene (band1_url, band2_url, band3_url), add the path-row and
        tensor as a key-value pair to the pathrow_imgs dictionary

        Inputs:
            pathrow (string)
            img_tensor (3d tensor)
        """
        self.pathrow_imgs[pathrow] = img_tensor

    def _get_tif_from_s3(img_urls):
        """
        Given a tuple of urls to R, G, and B TIF bands for a single scene,
        return a single TIF
        """
        r_url, g_url, b_url = img_urls

        b2 = rasterio.open(r_url)
        b3 = rasterio.open(g_url)
        b4 = rasterio.open(b_url)

        with rasterio.open('rgb.tiff', 'w', driver='Gtiff', width=b2.width,
                            height=b2.height, count=3, crs=b2.crs,
                            tranform=b2.transform, dtype='float32') as rgb:
            rgb.write(b2.read(1), 1)
            rgb.write(b3.read(1), 2)
            rgb.write(b4.read(1), 3)

        b2.close()
        b3.close()
        b4.close()

        return rgb

    def __call__(self, coord, pathrow, img_urls):
        """
        Extracts the 21x21 sub-array from the Landsat scene corresponding
        to the provided coordinates, and returns a normalized 3D tensor.

        Input:
            coord (tuple of 2 floats)
            pathrow (str)
            img_urls: tuple of urls to R, G, and B TIF bands to a single scene

        Returns a 3D tensor.
        """
        r_url, b_url, g_url = img_urls
        lon, lat = coords

        # If we already have the 3d tensor, just return it
        if self.pathrow_imgs[pathrow] != 0:
            return self.pathrow_imgs[pathrow]

        # Get TIF with 3 bands
        tif = self._get_tif_from_s3(img_urls)
        tif_data = tif.read()

        # Extract 224x224 subarray from landsat scene
        min_lat, min_lon, max_lat, max_lon = create_space(lat, lon)
        utm = pyproj.Proj(tif.crs)
        lonlat = pyproj.Proj(init='epsg:4326')
        east, north = pyproj.transform(lonlat, utm, lon, lat)

        row, col = tif.index(east, north)

        # how to reshape tif?
        array = rgb_data[:, row: row+224, col: col+224]

        # Convert array to tensor
        landsat_tensor = torch.tensor(array).type(torch.FloatTensor)

        # Add tensor to dict
        self._update_dict(pathrow, landsat_tensor)

        return landsat_tensor


############################# DEPRECATED #####################################

class deprecated_LandsatTransform:
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
        img = Image.open(path)
        img = img.resize((self.width, self.height))
        return TF.pil_to_tensor(img.convert('RGB')).type(torch.FloatTensor)/255

class deprecated_ViirsTransform:
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
        Extracts the 21x21 sub-array from the the full VIIRS tile set
        corresponding to the provided coordinates,
        and returns a normalized 3D tensor.

        Normalization:
            output = ln(input + 1)/ln(max)
            Where max = 92,000 radiance in our dataset.
            Maps to [0,1] and reduces outsize influence of outliers.
        Input:
            coord (tuple of 2 floats)
            country (str): One of ['eth', 'mw', 'ng']
        Returns a 3D tensor.
        """
        min_lat, min_lon, max_lat, max_lon = create_space(
            coord[0], coord[1])
        xminPixel, ymaxPixel = self.tifs[country].proj_to_raster(min_lon, min_lat)
        xminPixel, ymaxPixel = int(xminPixel), int(ymaxPixel)
        array = self.arrays[country][:, ymaxPixel-21:ymaxPixel, xminPixel:xminPixel+21]
        viirs_tensor = torch.tensor(array.reshape((-1,21,21))).type(torch.FloatTensor)
        return torch.log(viirs_tensor + 1) / 11.43

class LandsatDataset(tud.Subset):
    """
    A data loader that samples pairs of Landsat
    and VIIRS images for matching land areas.
    """
    def __init__(
            self, df, landsat_transform
        ):
        self.df = df
        self.landsat_transform = landsat_transform
        self.idxs = df.index.to_list()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx = self.idxs[idx]
        cols = ['image_lat', 'image_lon', 'image_name', 'country']
        lat, lon, img, country = self.df.loc[idx, cols]
        landsat = self.landsat_transform(img, country)
        return landsat

class ViirsDataset(tud.Subset):
    """
    A data loader that samples pairs of Landsat
    and VIIRS images for matching land areas.
    """
    def __init__(
            self, df, viirs_transform
        ):
        self.df = df
        self.viirs_transform = viirs_transform
        self.idxs = df.index.to_list()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx = self.idxs[idx]
        cols = ['image_lat', 'image_lon', 'image_name', 'country']
        lat, lon, img, country = self.df.loc[idx, cols]
        viirs = self.viirs_transform((lat, lon), country)
        return viirs
