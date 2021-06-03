import torch
import torch.utils.data as tud
import numpy as np
from utils import create_space
import os
# from PIL import Image, ImageFile
import torchvision.transforms.functional as TF
from collections import defaultdict
import rasterio
from rasterio.windows import Window
import pyproj

# ImageFile.LOAD_TRUNCATED_IMAGES = True

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

    def __zero_pad_pathrow(self, path, row):
        return ("000" + str(path))[-3:] + ("000" + str(row))[-3:]

    def __getitem__(self, idx):
        idx = self.idxs[idx]
        cols = ['# lon', 'lat', 'path', 'row', 'B2_link', 'B3_link', 'B4_link']
        lon, lat, path, row, B2_url, B3_url, B4_url = self.df.loc[idx, cols]
        # 0 pad !!!!            < --------------------------------------------------------v-------
        pathrow = self.__zero_pad_pathrow(path, row)
        landsat = self.landsat_transform((lon, lat), pathrow,
                                         (B2_url, B3_url, B4_url))
        viirs = self.viirs_transform((lon, lat))
        return landsat, viirs

class LandsatTransform:
    """
    A callable object that, given a pair of coordinates, returns the
    Landsat image formatted as a 3D Tensor [bands, height, width].
    """
    def __init__(self, xdim=224, ydim=224):
        """
        Inputs: None
        """
        self.coord_imgs = {}
        # self.pathrow_tifs = {}
        self.xdim = xdim
        self.ydim = ydim

        if not os.path.isdir('landsat'):
            os.mkdir('landsat')

    # def _update_dict(pathrow, img_tensor):
    #     """
    #     Given a path-row and a 3d tensor representing an image for a single
    #     pathrow scene (band1_url, band2_url, band3_url), add the path-row and
    #     tensor as a key-value pair to the pathrow_imgs dictionary

    #     Inputs:
    #         pathrow (string)
    #         img_tensor (3d tensor)
    #     """
    #     self.pathrow_imgs[pathrow] = img_tensor

    def _get_tif(self, img_urls, pathrow):
        """
        Given a tuple of urls to R, G, and B TIF bands for a single scene,
        return a single TIF
        """
        rgb_path = 'landsat/' + pathrow + '.tiff'

        if os.path.isfile(rgb_path):
            return rasterio.open(rgb_path)

        r_url, g_url, b_url = img_urls

        b2 = rasterio.open(r_url)
        b3 = rasterio.open(g_url)
        b4 = rasterio.open(b_url)

        # with rasterio.open(pathrow+'.tiff', 'w+', driver='Gtiff', width=b2.width,
        #                     height=b2.height, count=3, crs=b2.crs,
        #                     tranform=b2.transform, dtype='float32') as rgb:
        rgb = rasterio.open(pathrow+'.tiff', 'w+', driver='Gtiff', width=b2.width,
                            height=b2.height, count=3, crs=b2.crs,
                            tranform=b2.transform, dtype='float32')
        rgb.write(b2.read(1), 1)
        rgb.write(b3.read(1), 2)
        rgb.write(b4.read(1), 3)

        b2.close()
        b3.close()
        b4.close()

        # self.pathrow_tifs[pathrow] = rgb
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
        lon, lat = coord

        # If we already have the 3d tensor, just return it
        if coord in self.coord_imgs:
            return self.coord_imgs[coord].new_tensor()
        # if we already created the RGB tif for the pathrow, use that
        # elif pathrow in self.pathrow_tifs:
        #     tif = pathrow_tifs[pathrow]
        # otherwise create the RGB tif from tifs stored in S3
        else:
            # Get TIF with 3 bands
            tif = self._get_tif(img_urls, pathrow)
            # tif_data = tif.read()

        # Extract 224x224 subarray from landsat scene
        min_lat, min_lon, max_lat, max_lon = create_space(lat, lon)
        utm = pyproj.Proj(tif.crs)
        lonlat = pyproj.Proj(init='epsg:4326')
        east, north = pyproj.transform(lonlat, utm, max_lon, max_lat)
        west, south = pyproj.transform(lonlat, utm, min_lon, min_lat)

        north_idx, west_idx = tif.index(west, north)
        south_idx, east_idx = tif.index(east, south)

        # REPLACE WITH A WINDOW READ !!!!!!!!!
        # raw_array = rgb_data[:, north_idx:south_idx+1, west_idx:east_idx+1] # <-------------------
        raw_array = tif.read(window=Window(
            west_idx, north_idx, 
            abs(west_idx - east_idx), 
            abs(north_idx - south_idx)
        ))

        tif.close()

        # Convert array to tensor
        landsat_tensor = torch.tensor(raw_array).type(torch.FloatTensor)

        # resize tensor
        landsat_tensor = TF.resize(
            landsat_tensor,
            size = (self.ydim, self.xdim)
        )

        # Add tensor to dict
        # self._update_dict(pathrow, landsat_tensor)
        self.coord_imgs[coord] = landsat_tensor

        return landsat_tensor.new_tensor()

class ViirsTransform:
    """
    A callable object that, given a pair of coordinates, returns a the
    VIIIRS Day/Night Band image formatted as a 3D Tensor [bands, height, width].
    """
    def __init__(self, tif, subsetBBox=None):
        """
        Inputs:
            tif (rasterio.DatasetReader)
            subsetBBox: tuple of 2 coords (float,float) representing 
                ((min lon, min lat), (max lon, max lat)) of the desired subset.
                All points must be in WGS84.
        """
        self.tif = tif
        if subsetBBox:
            (west, south), (east, north) = subsetBBox
            miny, minx = tif.index(west, south)
            maxy, maxx = tif.index(east, north)
            height = abs(maxy - miny)
            width = abs(maxx - minx)
            self.col_offset = minx
            self.row_offset = maxy
            self.tif_data = tif.read(window=Window(minx, maxy, width, height))
        else:
            self.col_offset = 0
            self.row_off = 0
            self.tif_data = tif.read()

    def __call__(self, coord):
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
        Returns a 3D tensor.
        """
        lon, lat = coord

        min_lat, min_lon, max_lat, max_lon = create_space(lat, lon)

        row, col = self.tif.index(min_lon, max_lat)
        row -= self.row_offset
        col -= self.col_offset
        array = self.tif_data[:, row:row+21, col:col+21]
        viirs_tensor = torch.tensor(array.reshape((-1,21,21))).type(torch.FloatTensor)
        
        return torch.clamp(torch.log(viirs_tensor + 1) / 11.43, min=0, max=1)


############################# DEPRECATED #####################################

# class deprecated_LandsatTransform:
#     """
#     A callable object that, given a pair of coordinates, returns a the
#     Landsat image formatted as a 3D Tensor [bands, height, width].
#     """
#     def __init__(self, base_path, width=256, height=256):
#         self.base_path = base_path
#         self.width = width
#         self.height = height

#     def __call__(self, image_name, country='ng'):
#         path = '/'.join([self.base_path, country, image_name])
#         img = Image.open(path)
#         img = img.resize((self.width, self.height))
#         return TF.pil_to_tensor(img.convert('RGB')).type(torch.FloatTensor)/255

# class deprecated_ViirsTransform:
#     """
#     A callable object that, given a pair of coordinates, returns a the
#     VIIIRS Day/Night Band image formatted as a 3D Tensor [bands, height, width].
#     """
#     def __init__(self, tifs):
#         """
#         Inputs:
#             tif
#         """
#         self.tifs = {
#             'ng' : tifs[1],
#             'eth' : tifs[1],
#             'mw' : tifs[0]
#         }
#         tif0_data = tifs[0].get_data()
#         tif1_data = tifs[1].get_data()
#         self.arrays = {
#             'ng' : tif1_data,
#             'eth' : tif1_data,
#             'mw' : tif0_data
#         }

#     def __call__(self, coord, country):
#         """
#         Extracts the 21x21 sub-array from the the full VIIRS tile set
#         corresponding to the provided coordinates,
#         and returns a normalized 3D tensor.

#         Normalization:
#             output = ln(input + 1)/ln(max)
#             Where max = 92,000 radiance in our dataset.
#             Maps to [0,1] and reduces outsize influence of outliers.
#         Input:
#             coord (tuple of 2 floats)
#             country (str): One of ['eth', 'mw', 'ng']
#         Returns a 3D tensor.
#         """
#         min_lat, min_lon, max_lat, max_lon = create_space(
#             coord[0], coord[1])
#         xminPixel, ymaxPixel = self.tifs[country].proj_to_raster(min_lon, min_lat)
#         xminPixel, ymaxPixel = int(xminPixel), int(ymaxPixel)
#         array = self.arrays[country][:, ymaxPixel-21:ymaxPixel, xminPixel:xminPixel+21]
#         viirs_tensor = torch.tensor(array.reshape((-1,21,21))).type(torch.FloatTensor)
#         return torch.clamp(
#             (torch.log(viirs_tensor + 1) / 11.43),
#             min=0, max=1
#         )

# class LandsatDataset(tud.Subset):
#     """
#     A data loader that samples pairs of Landsat
#     and VIIRS images for matching land areas.
#     """
#     def __init__(
#             self, df, landsat_transform
#         ):
#         self.df = df
#         self.landsat_transform = landsat_transform
#         self.idxs = df.index.to_list()

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         idx = self.idxs[idx]
#         cols = ['image_lat', 'image_lon', 'image_name', 'country']
#         lat, lon, img, country = self.df.loc[idx, cols]
#         landsat = self.landsat_transform(img, country)
#         return landsat

# class ViirsDataset(tud.Subset):
#     """
#     A data loader that samples pairs of Landsat
#     and VIIRS images for matching land areas.
#     """
#     def __init__(
#             self, df, viirs_transform
#         ):
#         self.df = df
#         self.viirs_transform = viirs_transform
#         self.idxs = df.index.to_list()

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         idx = self.idxs[idx]
#         cols = ['image_lat', 'image_lon', 'image_name', 'country']
#         lat, lon, img, country = self.df.loc[idx, cols]
#         viirs = self.viirs_transform((lat, lon), country)
#         return viirs

# class Landsat(tud.Dataset):
#     """
#     A data loader that samples pairs of Landsat images.
#     """
#     def __init__(self, df, landsat_transform):
#         self.df = df
#         self.landsat_transform = landsat_transform
#         self.idxs = df.index.to_list()

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         idx = self.idxs[idx]
#         img, country = self.df.loc[idx, ['image_name', 'country']]
#         landsat = self.landsat_transform(img, country)
#         return landsat, landsat

# class ToTensor(object):
#     def __init__(self, bands, height, width):
#         self.x = width
#         self.y = height
#         self.z = bands

#     def __call__(self, sample):
#         m = torch.from_numpy(sample).type(torch.float).reshape(
#             (self.z, self.y, self.x))
#         return m

# class MNIST(tud.Dataset):
#     def __init__(self, X, transform=ToTensor):
#         self.X = X
#         self.transform = transform(1, 28, 28)

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         x = self.X[idx]
#         if self.transform:
#             x = self.transform(x)
#         return (x, x)
