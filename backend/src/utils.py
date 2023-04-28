

import numpy as np

from skimage.measure import find_contours
from skimage.draw import polygon_perimeter


def get_distance_transform(binary_image, tf_type="signed"):
    from scipy.ndimage import distance_transform_edt
    out = np.zeros_like(binary_image)
    if tf_type == "signed" or tf_type == "inner" or tf_type == "unsigned":
        mask = binary_image
        dtf = distance_transform_edt(binary_image)
        out += mask * dtf
        if tf_type == "signed":
            out *= -1
    if tf_type == "signed" or tf_type == "outer" or tf_type == "unsigned":
        mask = 1 - binary_image
        dtf = distance_transform_edt(1 - binary_image)
        out += mask * dtf
    return out


def get_border_raster(binary_mask):
    raster = np.zeros(binary_mask.shape)
    contours = find_contours(binary_mask, 0.5)
    for c in contours:
        rr, cc = polygon_perimeter(c[:, 0], c[:, 1], binary_mask.shape)
        raster[rr, cc] = 1
    return raster