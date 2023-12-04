"""
This Python file implements the revised version of inclusion depth
There are three versions of inclusion depth:
 - Vanilla (strict)
 - Modified
 - Linear-time
"""

from time import time
import numpy as np
from skimage.measure import find_contours

def compute_depths(data,
                   modified=False,
                   return_data_mat=False,
                   times_dict=None):
    """
    Calculate depth of a list of contours using the inclusion depth (BoD) method.
    :param data: list
        List of contours to calculate the BoD from. A contour is assumed to be
        represented as a binary mask with ones denoting inside and zeros outside
        regions.
    :param modified: bool
        Whether to use or not the modified BoD (mBoD). This reduces the sensitivity of
        the method to outliers but yields more informative depth estimates when curves
        cross over a lot.
    :param return_data_mat: ndarray
        If true, in addition to the depths, returns the raw depth matrix which is of
        dimensions 2 x n x n, where n = len(data) and the first dimension indexes the
        in/out proportion counts.
    :param times_dict: dict
        If a dict is passed as `times_dict`, the times that different stages of the method
        take are logged to this dict.
    :return:
        depths: ndarray
        depth_matrix: ndarray
    """

    if modified:
        return inclusion_depth_modified(data, times_dict=times_dict)
    else:
        return inclusion_depth_strict(data, times_dict=times_dict)


def inclusion_depth_strict(data,
                          times_dict=None):
    record_times = False
    if times_dict is not None and type(times_dict) == dict:
        record_times = True

    if record_times:
        t_start_preproc = time()

    num_contours = len(data)

    # - Get fractional inside/outside tables
    if record_times:
        t_end_preproc = time()
        t_start_core = time()

    depths = []
    for i in range(num_contours):
        binary_mask_i = data[i]
        in_count = 0
        out_count = 0
        for j in range(num_contours):
            binary_mask_j = data[j]
            intersect = ((binary_mask_i + binary_mask_j) == 2).astype(float)

            # inside/outside check
            is_in = np.all(binary_mask_i == intersect)
            is_out = np.all(binary_mask_j == intersect)
            if is_in and not is_out:
                in_count += 1
            if is_out and not is_in:
                out_count += 1

        depths.append(np.minimum(in_count/num_contours, out_count/num_contours))

    if record_times:
        t_end_core = time()

    if record_times:
        times_dict["time_preproc"] = t_end_preproc - t_start_preproc
        times_dict["time_core"] = t_end_core - t_start_core

    return np.array(depths, dtype=float)


def inclusion_depth_modified(data,
                            times_dict=None):
    record_times = False
    if times_dict is not None and type(times_dict) == dict:
        record_times = True

    if record_times:
        t_start_preproc = time()

    num_contours = len(data)
    # from skimage.segmentation import find_boundaries
    # contours_boundaries = [find_boundaries(c, mode="inner", connectivity=1) for c in data]
    # boundaries_coords = [np.where(cb == 1) for cb in contours_boundaries]

    # - Get fractional inside/outside tables
    depth_matrix_in = np.zeros((num_contours, num_contours))
    depth_matrix_out = np.zeros((num_contours, num_contours))

    if record_times:
        t_end_preproc = time()
        t_start_core = time()

    for i in range(num_contours):
        binary_mask_i = data[i]

        for j in range(num_contours):
            binary_mask_j = data[j]

            # the smaller eps_out becomes the less outside it is
            # so when it is zero, we now it is inside
            # we add a larger number to in matrix the more inside i is
            eps_out = (binary_mask_i - binary_mask_j)
            eps_out = (eps_out > 0).sum()
            eps_out = eps_out / (binary_mask_i.sum() + np.finfo(float).eps)
            depth_matrix_in[i, j] = 1 - eps_out

            # the smaller eps_in becomes, the less j is outside of i
            # so when it is zero, we know i is outside of j
            # we add a larger number to out matrix the more outside i is
            eps_in = (binary_mask_j - binary_mask_i)
            eps_in = (eps_in > 0).sum()
            eps_in = eps_in / (binary_mask_j.sum() + np.finfo(float).eps)
            depth_matrix_out[i, j] = 1 - eps_in

    depth_matrix = np.stack([depth_matrix_in, depth_matrix_out])
    depths = depth_matrix.mean(axis=2).min(
        axis=0)

    if record_times:
        t_end_core = time()

    if record_times:
        times_dict["time_preproc"] = t_end_preproc - t_start_preproc
        times_dict["time_core"] = t_end_core - t_start_core

    return depths


