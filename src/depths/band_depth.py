from time import time
from itertools import combinations
from functools import reduce
import numpy as np
from scipy.optimize import bisect


def compute_depths(data,
                   band_size=2,
                   modified=False,
                   target_mean_depth: float = 1 / 6,
                   return_data_mat=False,
                   times_dict=None):
    """
    Calculate depth of a list of contours using the contour band depth (CBD) method.
    :param data: list
        List of contours to calculate the CBD from. A contour is assumed to be
        represented as a binary mask with ones denoting inside and zeros outside
        regions.
    :param band_size: int
        Number of contours to consider when forming the bands.
    :param modified: bool
        Whether to use or not the modified CBD (mCBD). This reduces the sensitivity of
        the method to outliers but yields more informative depth estimates when curves
        cross over a lot.
    :param target_mean_depth: float
        Only applicable if using mCBD. If None, returns the depths as is.
        If a float is specified, finds a threshold of the depth matrix that yields a
        target mean depth of `target_mean_depth` using binary search.
    :param return_data_mat: ndarray
        If true, in addition to the depths, returns the raw depth matrix which is of
        dimensions n x (n combined band_size), where n = len(data).
    :param times_dict: dict
        If a dict is passed as `times_dict`, the times that different stages of the method
        take are logged to this dict.
    :return:
        depths: ndarray
        depth_matrix: ndarray
    """

    if modified:
        return contour_band_depth_modified(data, band_size=band_size, target_mean_depth=target_mean_depth, times_dict=times_dict)
    else:
        return contour_band_depth_strict(data, band_size=band_size, times_dict=times_dict)


def contour_band_depth_strict(data,
                              band_size=2,
                              times_dict=None):
    record_times = False
    if times_dict is not None and type(times_dict) == dict:
        record_times = True

    if record_times:
        t_start_preproc = time()

    num_contours = len(data)
    subsets = get_subsets(num_contours, band_size)
    num_subsets = len(subsets)

    # Compute fractional containment table

    # - Each subset defines a band. To avoid redundant computations we
    #   precompute these bands.
    band_components = []
    for i, subset in enumerate(subsets):
        bc = get_band_components([data[j] for j in subset])  # {union:, intersection:, band:}
        bc["members_id"] = i
        bc["members_idx"] = subset
        band_components.append(bc)

    if record_times:
        t_end_preproc = time()
        t_start_core = time()

    # - Get fractional containment tables
    depth_matrix = np.zeros((num_contours, num_subsets))

    for i, binary_mask in enumerate(data):
        for j, subset in enumerate(subsets):

            bc = band_components[j]
            idx_subset = bc["members_idx"]
            union = bc["union"]
            intersection = bc["intersection"]

            in_band = 0
            if i in idx_subset:  # contour is in band border
                in_band = 1
            else:
                intersect_in_contour = np.all(((intersection + binary_mask) == 2).astype(float) == intersection)
                contour_in_union = np.all(((union + binary_mask) == 2).astype(float) == binary_mask)
                if intersect_in_contour and contour_in_union:
                    in_band = 1

            depth_matrix[i, j] = in_band

    depths = depth_matrix.mean(axis=1)

    if record_times:
        t_end_core = time()

    if record_times:
        times_dict["time_preproc"] = t_end_preproc - t_start_preproc
        times_dict["time_core"] = t_end_core - t_start_core

    return depths


def contour_band_depth_modified(data,
                                band_size=2,
                                target_mean_depth: float = 1 / 6,
                                times_dict=None):
    record_times = False
    if times_dict is not None and type(times_dict) == dict:
        record_times = True

    if record_times:
        t_start_preproc = time()

    num_contours = len(data)
    subsets = get_subsets(num_contours, band_size)
    num_subsets = len(subsets)

    # Compute fractional containment table

    # - Each subset defines a band. To avoid redundant computations we
    #   precompute these bands.
    band_components = []
    for i, subset in enumerate(subsets):
        bc = get_band_components([data[j] for j in subset])  # {union:, intersection:, band:}
        bc["members_id"] = i
        bc["members_idx"] = subset
        band_components.append(bc)

    # - Get fractional containment tables
    depth_matrix = np.zeros((num_contours, num_subsets))

    if record_times:
        t_end_preproc = time()
        t_start_core = time()

    for i, binary_mask in enumerate(data):
        for j, subset in enumerate(subsets):

            bc = band_components[j]
            idx_subset = bc["members_idx"]
            union = bc["union"]
            intersection = bc["intersection"]

            if i in idx_subset:  # contour is in band border
                p_outside_of_band = 0
            else:
                lc_frac = (intersection - binary_mask)
                lc_frac = (lc_frac > 0).sum()
                lc_frac = lc_frac / (intersection.sum() + np.finfo(float).eps)

                rc_frac = (binary_mask - union)
                rc_frac = (rc_frac > 0).sum()
                rc_frac = rc_frac / (binary_mask.sum() + np.finfo(float).eps)

                p_outside_of_band = np.maximum(lc_frac, rc_frac)

            depth_matrix[i, j] = p_outside_of_band

    def mean_depth_deviation(mat, threshold, target):
        return target - (((mat < threshold).astype(float)).sum(axis=1) / num_subsets).mean()

    depth_matrix_t = depth_matrix.copy()
    if target_mean_depth is None:  # No threshold
        print("[cbd] Using modified band depths without threshold")
        depth_matrix_t = 1 - depth_matrix_t
    else:
        print(f"[cbd] Using modified band depth with specified threshold {target_mean_depth}")
        try:
            t = bisect(lambda v: mean_depth_deviation(depth_matrix_t, v, target_mean_depth), depth_matrix_t.min(),
                       depth_matrix_t.max())
        except:
            print("[cbd] Binary search failed to converge")
            t = depth_matrix_t.mean()
        print(f"[cbd] Using t={t}")

        depth_matrix_t = (depth_matrix < t).astype(float)

    depths = depth_matrix_t.mean(axis=1)

    if record_times:
        t_end_core = time()

    if record_times:
        times_dict["time_preproc"] = t_end_preproc - t_start_preproc
        times_dict["time_core"] = t_end_core - t_start_core

    return depths


def get_band_components(band_members):
    """
    Computes necessary components to perform the
    band test.
    :param band_members: list
        Binary masks of contours that make the band.
    :return:
        band_components: dict with union and intersection of band members
    """
    num_band_members = len(band_members)
    subset_array = np.concatenate([bm[:, :, np.newaxis] for bm in band_members], axis=-1)
    subset_sum = subset_array.sum(axis=-1)

    union = (subset_sum > 0)
    intersection = (subset_sum == num_band_members)
    union = union.astype(float)
    intersection = intersection.astype(float)
    # Commented because these are not needed to perform the band checking
    # band = union + intersection
    # band[band != 1.0] = 0.0
    # band = band.astype(float)

    return dict(union=union, intersection=intersection)


def get_subsets(num_members, subset_size=2):
    """
    Returns a list of subsets of the indicated size.
    subset_size can be an int or a list of ints with the sizes.
    Min subset_size is 2 and max subset_size is num_members//2
    """
    member_idx = np.arange(num_members).tolist()
    if type(subset_size) == int:
        subset_size = [subset_size, ]

    subsets = []
    for ss in subset_size:
        for idx in combinations(member_idx, ss):
            subsets.append(idx)

    return subsets