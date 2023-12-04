"""
This Python file implements the revised version of boundary depth
There are three versions of boundary depth:
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
    Calculate depth of a list of contours using the boundary depth (BoD) method.
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
        return boundary_depth_modified(data, times_dict=times_dict)
    else:
        return boundary_depth_strict(data, times_dict=times_dict)


def boundary_depth_strict(data,
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


def boundary_depth_modified(data,
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
        # boundary_mask_i = contours_boundaries[i]
        # boundary_coords_i = boundaries_coords[i]  # Could improve performanec

        for j in range(num_contours):
            binary_mask_j = data[j]
            # boundary_mask_j = contours_boundaries[j]
            # boundary_coords_j = boundaries_coords[j]   # Could improve performanec

            # v1
            # in12 = in_func(boundary_mask_i, binary_mask_j)  # fraction bi in j
            # in21 = in_func(boundary_mask_j, binary_mask_i)  # fraction bj in i
            # out12 = out_func(boundary_mask_i, binary_mask_j)  # fraction bi in -j
            # out21 = out_func(boundary_mask_j, binary_mask_i)   # fraction bj in -i
            #
            # depth_matrix_in[i, j] = np.minimum(in12, out21)
            # depth_matrix_out[i, j] = np.minimum(in21, out12)

            # v2
            # depth_matrix_in[i, j] = in_func(boundary_mask_i, binary_mask_j)  # fraction bi in j
            # depth_matrix_out[i, j] = out_func(boundary_mask_i, binary_mask_j)  # fraction bi in -j

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
        axis=0)  # np.minimum(depth_matrix.mean(axis=0), depth_matrix_out.mean(axis=1)).flatten()

    if record_times:
        t_end_core = time()

    if record_times:
        times_dict["time_preproc"] = t_end_preproc - t_start_preproc
        times_dict["time_core"] = t_end_core - t_start_core

    return depths


def in_func(b1, m2):
    in_pos = b1 * m2
    return in_pos.sum()/b1.sum()


def out_func(b1, m2):
    out_pos = b1 * (1 - m2)
    return out_pos.sum()/b1.sum()


def boundary_depth_fast(binary_masks, ensemble, modified):
    """
    O(N) version of boundary depths.
    The limitation of this faster version is that it does not consider topological validity.
    The fast version has precomputed in/out maps
    Then it adjust the maps for the current contour, removing elements not nested with it
    Then, computing the depth of the contour boils down do checking these maps
    """

    from skimage.segmentation import find_boundaries
    contours_boundaries = [find_boundaries(c, mode="inner", connectivity=1) for c in binary_masks]
    boundaries_coords = [np.where(cb == 1) for cb in contours_boundaries]

    # - in/out fields O(N)
    in_field = np.zeros_like(binary_masks[0], dtype=float)
    out_field = np.zeros_like(binary_masks[0], dtype=float)

    for i, binary_mask in enumerate(binary_masks):
        in_field += binary_mask
        out_field += (1 - binary_mask)

    depths = []

    for i, binary_mask in enumerate(binary_masks):
        boundary_mask = contours_boundaries[i]
        boundary_coords = boundaries_coords[i]

        in_boundary_tally = in_field[boundary_coords].flatten()
        out_boundary_tally = out_field[boundary_coords].flatten()

        if modified:
            in_boundary_tally /= len(ensemble)
            out_boundary_tally /= len(ensemble)
            prop_in = in_boundary_tally.sum() / boundary_mask.sum()
            prop_out = out_boundary_tally.sum() / boundary_mask.sum()
            depths.append(np.minimum(prop_in, prop_out))
        else:
            num_in = in_boundary_tally.min()
            num_out = out_boundary_tally.min()
            depths.append(np.minimum(num_in / len(ensemble), num_out / len(ensemble)))

    return depths


def boundary_depth_experimental_bit_manipulation(binary_masks, ensemble, modified):
    # compute maps

    # - in/out fields O(N)
    in_ids = np.zeros_like(binary_masks[0].astype(int), dtype="O")
    out_ids = np.zeros_like(binary_masks[0].astype(int), dtype="O")
    for i, contour in enumerate(binary_masks):
        in_ids = in_ids + contour.astype(int) * 2 ** i
        out_ids = out_ids + (1 - contour).astype(int) * 2 ** i

    depths = []

    for contour in binary_masks:
        contour = contour.astype(int)
        contour_in_ids = in_ids[contour == 1]
        contour_out_ids = in_ids[contour == 0]

        # partially inside tells me that at least a pixel of the shape is inside
        partial_in = np.bitwise_or.reduce(contour_in_ids)
        # fully inside tells me that all pixels of the shape are inside
        fully_in = np.bitwise_and.reduce(contour_in_ids)
        # partially outside tells me that a pixel of the shape is outside
        partial_out = np.bitwise_or.reduce(contour_out_ids)

        # the shape is contained: partially inside or fully inside and not outside
        valid1 = np.bitwise_and(np.bitwise_or(partial_in, fully_in), np.bitwise_not(partial_out))
        # the shape contains: fully inside and partially outside
        valid2 = np.bitwise_and(fully_in, partial_out)

        valid_in_mask = np.bitwise_and(in_ids, np.bitwise_or(valid1, valid2))
        valid_out_mask = np.bitwise_and(out_ids, np.bitwise_or(valid1, valid2))

        unique_vals_out = np.unique(valid_in_mask[contour == 1])
        unique_vals_in = np.unique(valid_out_mask[contour == 1])

        num_out = np.array([count_set_bits_large(num) for num in unique_vals_out]).min()
        num_in = np.array([count_set_bits_large(num) for num in unique_vals_in]).max()

        # 5. compute depths
        depths.append(np.minimum(num_in / len(ensemble), num_out / len(ensemble)))

    return depths


# Brian Kerninghan algorithm for counting set bits
# can use a vectorized numpy operation but need to figure out how
# to handle the bits given that we have arbitrary precission
def count_set_bits_large(num):
    count = 0
    while num != 0:
        num &= num - 1
        count += 1
    return count