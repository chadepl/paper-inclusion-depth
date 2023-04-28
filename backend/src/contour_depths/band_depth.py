"""
Given a list of contours (binary masks), we compute the depth of each member.
This is a list with len == len(list of contours)
The main method (compute_band_depths) allows providing precomputed data to speed up repeated computations.
"""
from time import time
from itertools import combinations
from functools import reduce
import numpy as np
from scipy.optimize import bisect


def compute_depths(contours, ensemble=None, subset_size=2,
                   modified=False, target_mean_depth: float = 1/6,
                   return_data_mat=False, times_dict=None):
    """

    :param contours:
    :param ensemble:
    :param subset_size:
    :param modified: uses areas instead of strict containment relationships.
    :param target_mean_depth: if modified is true finds a threshold that attains this depth.
    :param return_data_mat:
    :return:
    """
    record_times = False
    if times_dict is not None and type(times_dict) == dict:
        record_times = True

    if record_times:
        t_start_preproc = time()

    if ensemble is None:
        # If reference data cloud (ensemble) was not provided
        # then uses contours as ensemble, which results in computing
        # the centrality of all provided contours
        ensemble = contours

    contours_size = len(contours)
    ensemble_size = len(ensemble)

    subsets = get_subsets(ensemble_size, subset_size)
    num_subsets = len(subsets)

    # Compute fractional containment table

    # - Each subset defines a band. To avoid redundant computations we
    #   precompute these bands.
    band_components = []
    for i, subset in enumerate(subsets):
        bc = get_band_components([ensemble[j] for j in subset])
        bc["members_id"] = i
        bc["members_idx"] = subset
        band_components.append(bc)

    # Prevent redundancy
    # - check which members of `contours` are in `ensemble`
    # - we do not consider those members
    redundancy_mat = np.zeros((contours_size, ensemble_size))
    for i, contour_a in enumerate(contours):
        for j, contour_b in enumerate(ensemble):
            if np.all(contour_a == contour_b):
                redundancy_mat[i, j] = 1

    if record_times:
        t_end_preproc = time()
        t_start_core = time()

    # - Get fractional containment tables
    depth_matrix = np.zeros((contours_size, num_subsets))
    depth_mask = np.ones((contours_size, num_subsets))  # records cells that should remain as 0
    depth_data = dict()

    for i, contour in enumerate(contours):
        for j, subset in enumerate(subsets):

            bc = band_components[j]

            idx_subset = bc["members_idx"]
            i_in_subset = reduce(lambda a, b: a == 1 or b == 1, [redundancy_mat[i, b] for b in idx_subset])

            if i_in_subset:
                exclude_cell = True
                depth_mask[i, j] = 0
            else:
                exclude_cell = False

            union = bc["union"]
            intersection = bc["intersection"]

            lc_frac = (intersection - contour)
            lc_frac = (lc_frac > 0).sum()
            lc_frac = lc_frac / (intersection.sum() + np.finfo(float).eps)

            rc_frac = (contour - union)
            rc_frac = (rc_frac > 0).sum()
            rc_frac = rc_frac / (contour.sum() + np.finfo(float).eps)

            if not exclude_cell:
                depth_matrix[i, j] = np.maximum(lc_frac, rc_frac)

            depth_data[(i, j)] = dict(member_id=i,
                                      subset_id=j,
                                      subset=subset,
                                      lc_frac=lc_frac,
                                      rc_frac=rc_frac,
                                      exclude=exclude_cell)

    if record_times:
        t_end_core = time()

    # binary search to ensure depth_matrix has a
    if modified:
        def mean_depth_deviation(mat, threshold, target):
            return target - (((mat < threshold).astype(float) * depth_mask).sum(axis=1)/depth_mask.sum(axis=1)).mean()

        depth_matrix_t = depth_matrix.copy()
        if target_mean_depth is None:  # No threshold
            print("[bd] Using modified band depths without threshold")
            depth_matrix_t = 1 - depth_matrix_t
        else:
            print(f"[bd] Using modified band depth with specified threshold {target_mean_depth}")
            try:
                t = bisect(lambda v: mean_depth_deviation(depth_matrix_t, v, target_mean_depth), depth_matrix_t.min(),
                           depth_matrix_t.max())
            except:
                print("[bd] Binary search failed to converge")
                t = ((depth_matrix_t * depth_mask).sum() / depth_mask.sum(axis=1).reshape(-1, 1)).mean()
            print(f"[bd] Using t={t}")

            depth_matrix_t = (depth_matrix < t).astype(float)

    else:  # not modified version
        print("[bd] Using strict band depths")
        depth_matrix_t = (depth_matrix == 0).astype(float)

    depth_matrix_t = depth_matrix_t * depth_mask
    depths = depth_matrix_t.sum(axis=1) / depth_mask.sum(axis=1)

    if record_times:

        times_dict["time_preproc"] = t_end_preproc - t_start_preproc
        times_dict["time_core"] = t_end_core - t_start_core

    if return_data_mat:
        return depths, depth_matrix_t
    else:
        return depths


def get_band_components(band_members):
    num_band_members = len(band_members)
    subset_array = np.concatenate([bm[:, :, np.newaxis] for bm in band_members], axis=-1)
    subset_sum = subset_array.sum(axis=-1)

    union = (subset_sum > 0)
    intersection = (subset_sum == num_band_members)
    union = union.astype(float)
    intersection = intersection.astype(float)
    band = union + intersection
    band[band != 1.0] = 0.0
    band = band.astype(float)

    return dict(union=union, intersection=intersection, band=band)

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


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from backend.src.datasets.circles import circles_different_radii_distribution
    from backend.src.datasets.bd_paper import get_contaminated_contour_ensemble_shape
    from backend.src.vis_utils import plot_contour_spaghetti

    #ensemble = circles_different_radii_distribution(50, 300, 300)
    ensemble = get_contaminated_contour_ensemble_shape(50, 300, 300)

    plot_contour_spaghetti(ensemble)
    plt.show()

    depths_strict, dd_strict = compute_depths(ensemble, modified=False, target_mean_depth=None,  return_data_mat=True)
    depths_modified, dd_modified = compute_depths(ensemble, modified=True, target_mean_depth=None, return_data_mat = True)
    depths_modified_t, dd_modified_t = compute_depths(ensemble, modified=True, target_mean_depth=1/6, return_data_mat=True)

    fig, axs = plt.subplots(ncols=3)
    for i, ax in enumerate(axs):
        ax.set_title(["BD strict", "BD modified \n no t", "BD modified \n t=1/6"][i])
    plot_contour_spaghetti(ensemble, arr=depths_strict, is_arr_categorical=False, ax=axs[0])
    plot_contour_spaghetti(ensemble, arr=depths_modified, is_arr_categorical=False, ax=axs[1])
    plot_contour_spaghetti(ensemble, arr=depths_modified_t, is_arr_categorical=False, ax=axs[2])
    plt.show()

    depths = pd.DataFrame([depths_strict, depths_modified, depths_modified_t])
    depths = depths.T
    depths.columns = ["BD strict", "BD modified \n no t", "BD modified \n t=1/6"]
    sns.pairplot(depths)
    plt.show()
