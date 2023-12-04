"""
Here we propose a new way to compute depths for contour data.
Assumptions on the data: closed and consistently oriented curves
They yield similar/same graph-based orderings as CBD but:
 - Are more efficient to compute
 - Encode in the depth scores the notion of distance between elements
   and also orientation (left/right) and (up/down).
 - Provide multiple levels of analysis because they are build from local
   segments all the way up to complete shapes.

The steps are:
 - For each C_i in E, obtain its boundary points, we call this B_i
 - We define a function F_i^j for each pair of B_i, B_j that indicates for each vertex in B_i
   whether it is inside or outside of the contour defined by B_j.
 - Based on these functions/graphs defined along the contours borders, we compute pairwise distances
   between curves
 - Based on these functions/graphs we compute their graph depth, which is based on per point depths.
   For the per point depths we use for now only the in/out relationship between curves. Nevertheless,
   we could also use metric information, directional information, and derivatives of the functions.
 - Finally, we allow the localization of the previous computations by specifying a graph-based threshold/
   radius.
"""
from time import time
import numpy as np
from numpy.linalg import norm
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import find_contours
from skimage.measure import label

from ..src.utils import get_distance_transform


def get_border_functions(contours, ensemble, sdfs=None, return_sdfs=False, return_border_coords=False):
    num_rows, num_cols = ensemble[0].shape

    if sdfs is None:
        sdfs = [get_distance_transform(m, tf_type="signed") for m in ensemble]
    sdfs_interps = [RegularGridInterpolator((np.arange(num_rows), np.arange(num_cols)), sdf) for sdf in sdfs]

    functions = dict()
    borders = dict()
    for i, contour in enumerate(contours):
        borders[i] = np.concatenate(find_contours(contour, level=0.5), axis=0)
        functions[i] = dict()
        for j, sdf_interp in enumerate(sdfs_interps):
            functions[i][j] = sdf_interp(borders[i])

    to_return = [functions, ]
    if return_sdfs:
        to_return.append(sdfs)
    if return_border_coords:
        to_return.append(borders)

    return to_return[0] if len(to_return) == 1 else to_return


def compute_depths(contours, ensemble=None, modified=False, global_criteria="nestedness",
                   border_functions=None, return_point_depths=False, return_border_coords=False, times_dict=None,
                   use_fast=False):
    record_times = False
    if times_dict is not None and type(times_dict) == dict:
        record_times = True

    if record_times and not use_fast:
        t_start_preproc = time()

    if ensemble is None:
        # If reference data cloud (ensemble) was not provided
        # then uses contours as ensemble, which results in computing
        # the centrality of all provided contours
        ensemble = contours

    contours_raw = contours
    ensemble_raw = ensemble

    # Separate disconnected components into separate contours: we need a map from connected components to their contours
    # the rest of the operations we will perform on the updated contours/ensemble we aggregate in the end
    contours = []
    contours_dict = dict()
    for i, c in enumerate(contours_raw):
        lc = label(c)
        num_labels = lc.max()
        for l in range(1, num_labels + 1):
            if i not in contours_dict:
                contours_dict[i] = []
            contours_dict[i].append(len(contours))
            contours.append((lc == l).astype(float))

    ensemble = []
    ensemble_dict = dict()
    for i, c in enumerate(ensemble_raw):
        lc = label(c)
        num_labels = lc.max()
        for l in range(1, num_labels + 1):
            if i not in ensemble_dict:
                ensemble_dict[i] = []
            ensemble_dict[i].append(len(ensemble))
            ensemble.append((lc == l).astype(float))

    if use_fast:
        print("Using O(N) method")
        global_depths, point_depths, border_coords = compute_depths_fast(contours, ensemble=ensemble, mode="inner",
                                                                         modified=False,
                                                                         return_point_depths=True,
                                                                         return_border_coords=True,
                                                                         times_dict=times_dict)
    else:
        print("Using O(N^2) method")

        if record_times:
            t_end_preproc = time()
            t_start_core = time()

        ensemble_size = len(ensemble)
        if border_functions is None:
            if global_criteria == "l2_dist":
                border_functions, ensemble_sdfs, border_coords = get_border_functions(contours, ensemble,
                                                                                      return_sdfs=True,
                                                                                      return_border_coords=True)
            else:
                border_functions, border_coords = get_border_functions(contours, ensemble, return_sdfs=False,
                                                                       return_border_coords=True)
        else:
            if global_criteria == "l2_dist":
                border_functions, ensemble_sdfs, border_coords = get_border_functions(contours, ensemble,
                                                                                      return_sdfs=True,
                                                                                      return_border_coords=True)

        if global_criteria == "nestedness":
            global_mat = np.zeros((len(contours), len(ensemble)))
            for i, contour_i in enumerate(contours):
                for j, contour_j in enumerate(ensemble):
                    bf_ij = border_functions[i][j]  # contained relationships of i with respect to the ensemble
                    bf_ji = border_functions[j][i]  # contained relationships of i with respect to the ensemble

                    # Topological relationships "ala" Hausdorff
                    pos_cij = (np.sign(bf_ij) > 0).astype(int).mean()
                    neg_cij = (np.sign(bf_ij) < 0).astype(int).mean()

                    pos_cji = (np.sign(bf_ji) > 0).astype(int).mean()
                    neg_cji = (np.sign(bf_ji) < 0).astype(int).mean()

                    min_pos = np.minimum(pos_cij, pos_cji)
                    min_neg = np.minimum(neg_cij, neg_cji)
                    max_pn = np.maximum(min_pos,
                                        min_neg)  # 0 means structures are nested, increasing values deviate from nestedness
                    if not modified:
                        max_pn = 0 if max_pn > 0 else 1  # strict topological version

                    global_mat[i, j] = 1 - max_pn  # nestedness
        elif global_criteria == "l2_dist":
            global_mat = np.zeros((len(contours), len(ensemble)))
            for i, contour_i in enumerate(contours):
                sdf_i = get_distance_transform(contour_i, tf_type="signed")
                for j, contour_j in enumerate(ensemble):
                    sdf_j = ensemble_sdfs[j]

                    global_mat[i, j] = norm((sdf_i - sdf_j).flatten())  # higher is worse
            global_mat = global_mat / global_mat.max()  # ranges between 0 and 1
            global_mat = 1 - global_mat
        else:
            global_mat = np.ones((len(contours), len(ensemble)))  # matrix of ones

        global_depths = []
        point_depths = []
        for i, contour in enumerate(contours):
            bf_i = border_functions[i]  # contained relationships of i with respect to the ensemble

            bf_arr = []
            for j, bf in bf_i.items():
                bf_arr.append(bf.reshape(1, -1))
            bf_arr = np.concatenate(bf_arr, axis=0)

            signed_arr = np.sign(bf_arr)
            weights = global_mat[i, :].copy().reshape(-1, 1)
            weights[i] = 0
            if weights.sum() > 0:  # Some structures are nested or partially nested
                weighted_arr = weights * signed_arr
                weighted_sum = weighted_arr.sum(axis=0)
                weighted_mean = np.abs(weighted_sum) / weights.sum()
                l1d = 1 - weighted_mean  # l1 depth
                global_depths.append(l1d.min())
            else:  # Component is disjoint, component should have depth 0
                global_depths.append(0)
                l1d = np.zeros(signed_arr.shape[1])

            point_depths.append(l1d.tolist())

        if record_times:
            t_end_core = time()

            times_dict["time_preproc"] = t_end_preproc - t_start_preproc
            times_dict["time_core"] = t_end_core - t_start_core

    # Aggregation stage
    agg_global_depths = []
    agg_point_depths = []
    for i, components_js in contours_dict.items():
        ds = np.zeros(len(components_js))
        point_ds = []
        for j_id, j in enumerate(components_js):
            ds[j_id] = global_depths[j]
            # print(point_depths)
            point_ds += point_depths[j]
        agg_global_depths.append(ds.min())  # TODO: is this the right aggregation mechanism?
        agg_point_depths.append(point_ds)

    to_return = [agg_global_depths]
    if return_point_depths:
        to_return.append(agg_point_depths)
    if return_border_coords:
        to_return.append(border_coords)

    return to_return[0] if len(to_return) == 1 else to_return


def compute_depths_fast(contours, ensemble=None, mode="inner", modified=False,
                        return_point_depths=False, return_border_coords=False, times_dict=None):
    """
    To accelerate the per-boundary point inside/outside lookups we
    use a summary that we build by setting 0 pixels in ensemble members
    to -1 and adding all the ensemble members along the member dimension.
    Then, we go over each boundary pixel, adjust the sum and average it
    to obtain our uni-variate depth at a point. We can traverse the boundary
    in several ways, all which have implications:
    - Using find_contours to get points on the boundary. These are not exactly
      on the grid, so we must use a nearest neighbor interpolator to get the con-
      tainment relationships.
       - pitfall: errors due to interpolation
    - Using find_boundaries to find inner boundaries. Then, we consult the
      summary at each point, subtract 1 and compute the average.
        - pitfall: erros due to underestimation
    - Using find_boundaries to find outer boundaries. Then, we consult the
      summary at each point and compute the average.
        - pitfall: erros due to overestimation
    Note: find_contours is in general 3x faster than find_boundaries
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

    num_members = len(ensemble)
    lookup_in = np.zeros_like(ensemble[0])  # how many contours contain the boundary point
    lookup_out = np.zeros_like(ensemble[0])  # how many contours do not contain the boundary point
    for e in ensemble:
        m = e.copy()
        # m[e == 0] = -1  # L1 inspired vectors on the inside and outside and compute average
        lookup_in += m
        lookup_out += (1 - m)

    # If using mode interpolator then we need interpolator
    if mode == "interpolator":
        from scipy.interpolate import RegularGridInterpolator
        lookup_in_interpolator = RegularGridInterpolator(
            (np.arange(ensemble[0].shape[0]), np.arange(ensemble[0].shape[0])),
            lookup_in, method="nearest")
        lookup_out_interpolator = RegularGridInterpolator(
            (np.arange(ensemble[0].shape[0]), np.arange(ensemble[0].shape[0])),
            lookup_out, method="nearest")

    # We now compute the depth of provided contours
    if mode == "inner":
        from skimage.segmentation import find_boundaries
    elif mode == "interpolator":
        from skimage.measure import find_contours

    if record_times:
        t_end_preproc = time()
        t_start_core = time()

    global_depths = []
    point_depths = []
    border_coords = []
    for c in contours:

        if mode == "inner":
            b = find_boundaries(c.astype(int), connectivity=1, mode=mode)
            b_vals_in = lookup_in[np.where(b == 1)]
            b_vals_out = lookup_out[np.where(b == 1)]
            border_coords.append(np.where(b == 1))
        elif mode == "interpolator":
            b = np.concatenate(find_contours(c, level=0.5), axis=0)
            b_vals_in = lookup_in_interpolator(b)
            b_vals_out = lookup_out_interpolator(b)
            border_coords.append(b)

        contours_contain = b_vals_in - 1  # number of contours outside at a given point
        contours_not_contain = b_vals_out  # number of contours outside at a given point

        # L1 depth
        l1_scores = contours_contain + -1 * contours_not_contain
        l1_scores = np.abs(l1_scores) / num_members
        b_depths = 1 - l1_scores  # l1 depths

        # Needed for half-space variant min(outside, inside)
        # contours_contain /= num_members  # proportion out per boundary point
        # contours_not_contain /= num_members  # proportion in per boundary point
        # b_depths = np.minimum(contours_contain, contours_not_contain)  # Tukey depth, others are possible

        if modified:
            global_depths.append(np.minimum(contours_contain.mean(), contours_not_contain.mean()))
        else:
            global_depths.append(b_depths.min())  # Infimum according to Mosler

        point_depths.append(b_depths.tolist())

    if record_times:
        t_end_core = time()

        times_dict["time_preproc"] = t_end_preproc - t_start_preproc
        times_dict["time_core"] = t_end_core - t_start_core

    to_return = [global_depths]
    if return_point_depths:
        to_return.append(point_depths)
    if return_border_coords:
        to_return.append(border_coords)

    return to_return[0] if len(to_return) == 1 else to_return


def get_border_dmat(border_functions):
    num_members = len(border_functions)
    D = np.zeros((num_members, num_members))
    for i in range(num_members):
        for j in range(0, i):
            bf_ij = border_functions[i][j]
            bf_ji = border_functions[j][i]
            d = np.minimum(np.min(bf_ij), np.min(bf_ji))
            D[i, j] = d
            D[j, i] = d
    return D


def get_border_dmat_v2(border_functions):
    num_members = len(border_functions)
    dmat = np.zeros((num_members, num_members))
    for i in range(num_members):
        for j in range(num_members):
            dmat[i, j] = np.where(border_functions[i][j] >= 0)[0].size / border_functions[i][j].size

    from scipy.spatial.distance import pdist, squareform
    dmat = squareform(pdist(dmat))
    return dmat


def get_hist_feat_mat(border_functions, density=True, bins=10):
    num_members = len(border_functions)
    feats = []
    for i in range(num_members):
        i_cloud = np.concatenate([bf.reshape(1, -1) for bf in border_functions[i].values()], axis=0).mean(axis=0)
        feats.append(np.histogram(i_cloud, bins=bins, density=density)[0].reshape(1, -1))
    return np.concatenate(feats, axis=0)


def get_border_dmat_v3(border_functions, density=True, bins=10, metric="euclidean"):
    feats = get_hist_feat_mat(border_functions, density, bins)
    from scipy.spatial.distance import pdist, squareform
    dmat = squareform(pdist(feats, metric=metric))
    return dmat


if __name__ == "__main__":
    from time import time
    import pandas as pd
    from skimage.measure import find_contours
    import matplotlib.pyplot as plt
    import seaborn as sns
    from backend.src.datasets.circles import circles_different_radii_spreads, circles_multiple_radii_modes, \
        circles_with_outliers
    from backend.src.datasets.bd_paper import get_contaminated_contour_ensemble_magnitude, \
        get_contaminated_contour_ensemble_shape, get_contaminated_contour_ensemble_topological
    from backend.src.vis_utils import plot_contour_spaghetti, plot_contour_boxplot

    num_members = 50
    ensemble = circles_different_radii_spreads(num_members, 300, 300, high_spread=False)
    # ensemble = circles_multiple_radii_modes(num_members, 300, 300, num_modes=2)
    #    ensemble = circles_with_outliers(num_members, 300, 300, num_outliers=10)
    #     _, _, ensemble = get_han_dataset(300, 300)
    # ensemble = get_contaminated_contour_ensemble_magnitude(num_members, 300, 300)
    # ensemble = get_contaminated_contour_ensemble_shape(num_members, 300, 300)
    ensemble = get_contaminated_contour_ensemble_topological(num_members, 300, 300, p_contamination=0.10)

    times = []
    t_start = time()
    depths_slow = compute_depths(ensemble, global_criteria=None, modified=False, return_point_depths=False)
    t_end = time()
    times.append(t_end - t_start)

    t_start = time()
    depths_slow_v2 = compute_depths(ensemble, global_criteria="nestedness", modified=True, return_point_depths=False)
    t_end = time()
    times.append(t_end - t_start)

    t_start = time()
    depths_fast_v1 = compute_depths(ensemble, use_fast=True, return_point_depths=False)
    t_end = time()
    times.append(t_end - t_start)

    print(f"depths_slow took {times[0]} seconds")
    print(f"depths_slow_v2 took {times[2]} seconds")
    print(f"depths_fast_v1 took {times[1]} seconds")

    print(depths_slow)
    print(depths_slow_v2)
    print(depths_fast_v1)

    # print(np.array(depths_slow) - np.array(depths_fast_v1))

    depths = pd.DataFrame([depths_slow, depths_slow_v2, depths_fast_v1])  # , depths_fast_v2])
    depths = depths.T
    depths.columns = ["slow", "slow_v2", "fast_v1"]
    sns.pairplot(depths)
    plt.show()

    fig, axs = plt.subplots(ncols=3, layout="tight")
    plot_contour_boxplot(ensemble, depths=depths_slow, outlier_type="threshold", epsilon_out=0.1, ax=axs[0])
    plot_contour_boxplot(ensemble, depths=depths_slow_v2, outlier_type="threshold", epsilon_out=0.1, ax=axs[1])
    plot_contour_boxplot(ensemble, depths=depths_fast_v1, outlier_type="threshold", epsilon_out=0.1, ax=axs[2])
    plt.show()

    fig, axs = plt.subplots(ncols=3, layout="tight")
    plot_contour_spaghetti(ensemble, arr=depths_slow, is_arr_categorical=False, ax=axs[0])
    plot_contour_spaghetti(ensemble, arr=depths_slow_v2, is_arr_categorical=False, ax=axs[1])
    plot_contour_spaghetti(ensemble, arr=depths_fast_v1, is_arr_categorical=False, ax=axs[2])
    plt.show()

    fig, axs = plt.subplots(ncols=2)
    for i, ax in enumerate(axs):
        ax.imshow(np.ones_like(ensemble[0]), cmap="gray_r")
        ax.set_axis_off()
        ax.set_title(["Fast V1", "Fast V2"][i])
        plot_contour_spaghetti(ensemble, [depths_fast_v1, depths_fast_v2][i], is_arr_categorical=False, ax=ax)
    # for i, d in enumerate():
    #     for j, e in enumerate(ensemble):
    #         for c in find_contours(e):
    #             axs[i].plot(c[:, 1], c[:, 0], c=plt.cm.get_cmap("inferno")(d[j]))
    plt.show()

    ###########################################
    # Baseline clustering (without refinement)
    ###########################################

    border_functions = get_border_functions(ensemble, ensemble)
    v1 = get_border_dmat(border_functions)
    v2 = get_border_dmat_v2(border_functions)
    v3 = get_border_dmat_v3(border_functions)

    fig, axs = plt.subplots(ncols=3)
    axs[0].matshow(v1)
    axs[1].matshow(v2)
    axs[2].matshow(v3)
    for i, title in enumerate(["V1", "V2", "V3"]):
        axs[i].set_title(title)
    plt.show()

    from sklearn_extra.cluster import KMedoids
    from scipy.cluster.hierarchy import linkage

    kmedoids_v1 = KMedoids(n_clusters=2, random_state=0).fit(v2)
    kmedoids_v2 = KMedoids(n_clusters=2, random_state=0).fit(v3)
    # initial_clustering = kmedoids.medoid_indices
    # initial_clustering = kmedoids.labels_

    fig, axs = plt.subplots(ncols=2, nrows=2)
    axs[0, 0].set_title("DMat V1: in/out proportion")
    axs[0, 1].set_title("DMat V2: histogram of local depths")
    axs[0, 0].matshow(v2)
    axs[0, 1].matshow(v3)
    for i, clust in enumerate([kmedoids_v1.labels_, kmedoids_v2.labels_]):
        for j, e in enumerate(ensemble):
            for c in find_contours(e):
                axs[1, i].plot(c[:, 1], c[:, 0], c=plt.cm.get_cmap("tab10")(clust[j]))
    for ax in axs.flatten():
        ax.set_axis_off()
        ax.set_axis_off()
    plt.show()
