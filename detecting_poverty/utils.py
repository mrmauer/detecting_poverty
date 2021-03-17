import torch
import numpy as np
from PIL import Image
import random
import math
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt

def adjusted_classes(y_scores, threshold):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.

    Inputs:
        y_scores (1D array): Values between 0 and 1.
        threshold (float): probability threshold

    Returns:
        array of 0s and 1s
    """
    return (y_scores >= threshold).astype(int)

def confusion_matrix(Ytrue, Ypred):
    """
    Display a color weighted confusion matrix for binary classification.
    """
    sns.heatmap(metrics.confusion_matrix(Ytrue, Ypred), annot=True)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

def roc_auc(model, Xtest, Ytest):
    """
    Display the ROC curve.
    """
    metrics.plot_roc_curve(model, Xtest, Ytest)  
    plt.show() 

def precision_recall(model, Xtest, Ytest):
    """
    Display the Precision-Recall curve.
    """
    metrics.plot_precision_recall_curve(model, Xtest, Ytest)  
    plt.show() 

class LatLonBBDraw():
    '''
    A callable class to provide random samples of fixed size bounding
    boxes from within a larger (super) bounding box.
    '''
    def __init__(self, lat_range, lon_range, dimension):
        """
        Create parameters for sampling function.

        Inputs:
            lat_range (tuple of two floats): north and south boundaries
                of super bounding box.
            lon_range (tuple of two floats): west and east boundaries.
            dimension (float): width (and height) in degrees of desired bounding
                box sub samples.
        """
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.dimension = dimension
    def __call__(self):
        """
        Returns:
            A length 4 tuple of floats: (west, south, east, north)
        """
        lat = random.uniform(self.lat_range[0], self.lat_range[1])
        lon = random.uniform(self.lon_range[0], self.lon_range[1])
        return (lon, lat, lon+self.dimension, lat+self.dimension)

class LatLonDraw():
    '''
    A callable class to provide random samples of lat/lon pairs 
    from within a bounding box.
    '''
    def __init__(self, lat_range, lon_range):
        """
        Create parameters for sampling function.

        Inputs:
            lat_range (tuple of two floats): north and south boundaries
                of super bounding box.
            lon_range (tuple of two floats): west and east boundaries.
        """
        self.lat_range = lat_range
        self.lon_range = lon_range
    def __call__(self):
        """
        Returns:
            A length 2 tuple of floats: (lat, lon)
        """
        lat = random.uniform(self.lat_range[0], self.lat_range[1])
        lon = random.uniform(self.lon_range[0], self.lon_range[1])
        return (lon, lat)

def resize(digits, row_size, column_size):
    """
    Resize images from input scale to row-size x clumn_size
    @row_size,column_size : scale_size intended to be
    """

    return np.array(
        [
            np.array(Image.fromarray(_).resize((row_size, column_size)))
            for _ in digits
        ]
    )


def gen_solution(test_lst, fname):
    """
    Generate csv file for Kaggle submission
    ------
    :in:
    test_lst: 1d array of (n_data), predicted test labels
    fname: string, name of output file
    """

    heads = ['Id', 'Category']
    with open(fname, 'w') as fo:
        fo.write(heads[0] + ',' + heads[1] + '\n')
        for ind in range(len(test_lst)):
            fo.write(heads[0] + ' ' + str(ind + 1) + ',' + str(test_lst[ind]) + '\n')
            
def create_space(lat, lon, s=10):
    """Creates a s km x s km square centered on (lat, lon)"""
    v = (180/math.pi)*(500/6378137)*s # roughly 0.045 for s=10
    return lat - v, lon - v, lat + v, lon + v

    