"""
Utilities for generating ensembles of ellipses.
"""

import math
import numpy as np
from skimage.measure import find_contours

import matplotlib.pyplot as plt


def generate_ellipse_sdf(num_cols: int, num_rows: int,
                         ellipticity: float = 0.0,
                         isovalue: float = 0.5,
                         translation_x: float = 0.0,
                         translation_y: float = 0.0,
                         rotation_deg: int = 0,
                         noise_params: dict = None, **kwargs) -> np.ndarray:
    """
    Generates the signed distance field of an ellipse with the specified parameters.
    We assume that the x-axis goes from left (negative) to right (positive).
    We assume that the y-axis foes from bottom (negative) to top (positive).
    We use array like conventions:
      - SDF[0, 0] corresponds to the top left (x: -1, y: 1) corner of the field.
      - SDF[num_cols-1, num_rows-1] corresponds to the bottom right (x: 1, y: -1) corner of the field.
    :param num_cols:
    :param num_rows:
    :param ellipticity:
    :param isovalue:
    :param translation:
    :param rotation_deg:
    :param noise_params:
    :return:
    """

    rotation_rad = np.deg2rad(rotation_deg)

    a = 1
    b = -((ellipticity ** 2) * (a ** 2) - (a ** 2))

    # Mesh creation

    # - Create domain coordinates X \in [-1,1] and Y \in [-1, 1]
    X = np.repeat(np.linspace(-1, 1, num_cols).reshape(1, -1), num_rows, axis=0)
    Y = np.repeat(np.linspace(1, -1, num_rows).reshape(-1, 1), num_cols, axis=1)

    # - Rotate domain by rotation_deg degrees
    X1 = X*math.cos(rotation_rad) - Y*math.sin(rotation_rad)
    Y1 = X*math.sin(rotation_rad) + Y*math.cos(rotation_rad)

    # - Translate the domain using the translation vector
    X1 = X1 - translation_x  # Subtracting the translation vector has the opposite visual effect
    Y1 = Y1 - translation_y  # Subtracting the translation vector has the opposite visual effect

    # - Scale domain (from circle to ellipse)
    X1 = X1 / a
    Y1 = Y1 / b

    # Distance field calculation

    dist = np.sqrt(np.square(X1) + np.square(Y1))
    signed_dist = isovalue - dist

    # TODO: theta-correlated noise operations
    theta = np.arctan2(Y1, X1)
    if noise_params is None:
        noise_params = {"l0.freq": 1, "l0.weight": 1}
    noisy_dist = signed_dist + noise_params["l0.weight"]*np.sin(noise_params["l0.freq"]*theta)

    return noisy_dist


def get_ensemble_params(num_members, ellipticities=None, isovalues=None, translations_x=None, translations_y=None, rotations_deg=None, noise_params=None, metadata=None):
    """
    Transforms individual lists of params into a list of dicts of params that can be passed
    to `generate_ellipse_sdf` as **kwargs.
    If the value of the parameter is None, a default is used.
    If the value of the parameter is an element, the element is tiled to form an array of size num_members.
    """
    def check_numeric_param(param):
        if param is None:
            param = 0.0
        if type(param) is not list and type(param) is not np.ndarray:
            param_type = type(param)
            param = np.repeat(param, num_members)
        return param

    ellipticities = check_numeric_param(ellipticities)
    isovalues = check_numeric_param(isovalues)
    translations_x = check_numeric_param(translations_x)
    translations_y = check_numeric_param(translations_y)
    rotations_deg = check_numeric_param(rotations_deg)

    if noise_params is None:
        noise_params = {"l0.freq": 1, "l0.weight": 0}
    if type(noise_params) is dict:
        noise_params_list = [dict() for i in range(num_members)]
        for k in noise_params.keys():
            for i, v in enumerate(noise_params[k]):
                noise_params_list[i][k] = v
        noise_params = noise_params_list

    if metadata is None:
        metadata = dict()
    if type(metadata) is dict:
        metadata_list = [dict() for i in range(num_members)]
        for k in metadata.keys():
            for i, v in enumerate(metadata[k]):
                metadata_list[i][k] = v
        metadata = metadata_list

    params = []
    for member_id in range(num_members):
        params.append(dict(
            ellipticity=ellipticities[member_id],
            isovalue=isovalues[member_id],
            translation_x=translations_x[member_id],
            translation_y=translations_y[member_id],
            rotation_deg=rotations_deg[member_id],
            noise_params=noise_params[member_id],
            metadata=metadata[member_id]
        ))
    return params