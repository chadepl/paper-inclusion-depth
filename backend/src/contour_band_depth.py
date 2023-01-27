"""
Implements methods involved in the set band depth methodology.
Methods in this module expect as input binary masks.
Per member depth can be used to classify them using order statistics.
This classification could be then used, for instance, to construct boxplots.
"""

from itertools import combinations
import numpy as np
import pandas as pd

##################
# SET BAND DEPTH #
##################


def get_subsets(num_members, subset_size=2):
    """
    Returns a dict indexing the subsets used for computing bands.
    For each subset it includes: the indices of its members, the size.
    subset_size can be an int or a list of ints with the sizes.
    Min size is 2 and max size is num_members//2
    """
    member_idx = np.arange(num_members).tolist()
    if type(subset_size) == int:
        subset_size = [subset_size, ]
    i = 0
    subsets = dict()
    for ss in subset_size:
        for idx in combinations(member_idx, ss):
            subsets[i] = dict(subset_id=i, idx=idx, size=len(idx))
            i += 1
    return subsets


def get_contour_band_depths(ensemble_members, subset_size=2, random_frac=1.0):
    """
    This function returns the left and right containment matrices for a given
    subset size.
    If tidy, then the matrices are deconstructed into an array of dicts with
    the attributes: [{member_id, subset_size, subset_id, subset, lc_frac, rc_frac}, ...]
    If random frac is below 1.0, then it samples that proportion of subsets.
    """

    # Init
    num_members = len(ensemble_members)
    member_idx = np.arange(num_members)
    if type(subset_size) == dict:  # Already passed subset in structure
        subsets = subset_size
    else:
        subsets = get_subsets(num_members, subset_size)
    num_subsets = len(subsets)

    # Compute fractional containment table

    # - Each subset defines a band. To avoid redundant computations we
    #   precompute these bands.
    band_components = []
    for subset_id, subset_data in subsets.items():
        subset_data["band_components"] = get_band_components(ensemble_members, subset_data["idx"])

    # - Get fractional containment tables
    depth_data = []

    for member_id in member_idx:
        member = ensemble_members[member_id]
        for subset_id, subset_data in subsets.items():
            subset = subset_data["idx"]
            subset_size = subset_data["size"]

            if member_id in subset:
                exclude_cell = 1
            else:
                exclude_cell = 0

            bc = subset_data["band_components"]
            intersection = bc["intersection"]
            union = bc["union"]

            # lc_frac = (intersection != member).sum() / intersection.sum()
            # rc_frac = (member != union).sum() / member.sum()

            lc_frac = (member - intersection)
            lc_frac = lc_frac < 0  # parts of intersection mask that fell outside of member
            lc_frac = lc_frac.sum() / intersection.sum()

            rc_frac = (union - member)
            rc_frac = rc_frac < 0  # parts of member mask that fell outside of union
            rc_frac = rc_frac.sum() / member.sum()

            depth_data.append(dict(member_id=member_id,
                                   subset_id=subset_id,
                                   subset=subset,
                                   subset_size=subset_size,
                                   lc_frac=lc_frac,
                                   rc_frac=rc_frac,
                                   exclude=exclude_cell))

    out_tidy = dict(num_members=num_members,
                    num_subsets=num_subsets,
                    depth_data=depth_data,
                    subset_data=subsets)
    return out_tidy


def get_band_components(ensemble_members, subset_idx):
    subset_array = np.concatenate([ensemble_members[i][:, :, np.newaxis] for i in subset_idx], axis=-1)
    subset_sum = subset_array.sum(axis=-1)
    union = (subset_sum > 0)
    union_neg = np.logical_not(union)
    intersection = (subset_sum == len(subset_idx))
    union = union.astype(float)
    union_neg = union_neg.astype(float)
    intersection = intersection.astype(float)
    band = union + intersection
    band[band != 1.0] = 0.0
    band = band.astype(float)

    return dict(subset_idx=subset_idx,
                union=union,
                union_neg=union_neg,
                intersection=intersection,
                band=band)


def get_depth_matrix(depth_data, raw_quantity="max_lc_rc", threshold=None):

    # - prepare data
    depth_df = pd.DataFrame.from_records(depth_data["depth_data"])
    depth_df = depth_df.drop(["subset", "subset_size"], axis=1)
    subset_data = depth_data["subset_data"]
    subset_df = pd.DataFrame.from_records(list(subset_data.values()))

    # - depth is computed based on the max amount points falling outside the band, be it inside or outside
    depth_df["max_lc_rc"] = depth_df[["lc_frac", "rc_frac"]].max(axis=1)

    if threshold is None:
        pass  # no threshold, output raw data
    elif type(threshold) is float or type(threshold) is int:
        threshold = float(threshold)
        contains_th = depth_df["max_lc_rc"].to_numpy().copy()
        contains_th[np.logical_and(depth_df["max_lc_rc"].to_numpy() < threshold, depth_df["exclude"].to_numpy() == 0)] = 1
        contains_th[np.logical_and(depth_df["max_lc_rc"].to_numpy() >= threshold, depth_df["exclude"].to_numpy() == 0)] = 0
        depth_df["contains_th"] = contains_th
    elif threshold == "auto":
        print("Automatic, entropy based threshold generation")

    # - inspecting raw frac matrices
    if threshold is None:
        if raw_quantity not in ["lc_frac", "rc_frac", "max_lc_rc"]:
            raise Exception("`raw_quantity` should be one of: [lc_frac, rc_frac, max_lc_rc, contains_th]")
    else:
        raw_quantity = "contains_th"

    pivot = depth_df.pivot(index="member_id", columns="subset_id", values=raw_quantity)

    return pivot.to_numpy()



if __name__ == "__main__":

    from ellipse_generation import load_ensemble_ellipses, plot_ensemble_ellipses_overlay
    from skimage.measure import find_contours

    import matplotlib.pyplot as plt

    from utils import get_distance_transform, get_border_raster

    num_members = 15
    ensemble_data = load_ensemble_ellipses(num_members=num_members,
                                           num_cols=300,
                                           num_rows=300,
                                           params_set=1,
                                           random_state=42)
    members_data = [m["data"] for m in ensemble_data["ensemble"]["members"]]
    members_feat = [m["features"] for m in ensemble_data["ensemble"]["members"]]
    members_contours = [find_contours(m, 0.5)[0] for m in members_data]

    members_sdfs = []
    for member in members_data:
        members_sdfs.append(get_distance_transform(np.logical_not(get_border_raster(member))))
    sdfs_array = np.array([m.flatten() for m in members_sdfs])

    plot_ensemble_ellipses_overlay(num_members, members_contours, members_feat, alpha=0.5)

    #  Compute contour band depths

    lc_arr, rc_arr, subsets = get_contour_band_depth_matrix(members_data, 2, False, 1.0)
    mc_arr = np.maximum(lc_arr, rc_arr)

    fig, ax = plt.subplots(layout="tight")
    # ax.matshow(lc_arr, vmin=0, vmax=mc_arr.max())
    # ax.matshow(rc_arr, vmin=0, vmax=mc_arr.max())
    ax.matshow(mc_arr, vmin=0, vmax=mc_arr.max())
    ax.set_axis_off()
    plt.show()

    # - Rearrange rows in matrix to see if there are clear groups
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    Z = linkage(mc_arr, "ward")

    dn = dendrogram(Z)
    #plt.matshow(Z)
    #plt.hlines(4, 0, 180)
    plt.show()

    fc = fcluster(Z, 4, criterion="distance")
    mat_reordering = np.argsort(fc)
    plt.matshow(mc_arr[mat_reordering])
    plt.tight_layout()
    plt.axis("off")
    plt.show()

    colors = ["#1b9e77", "#d95f02", "#7570b3", "red"]
    for i in range(num_members):
        plt.plot(members_contours[i][:,1], members_contours[i][:,0], c=colors[fc[i]-1])
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # - sdf distance matrix
    from scipy.spatial.distance import pdist, squareform

    sdf_dists = squareform(pdist(sdfs_array))
    Z = linkage(sdf_dists, "ward")
    dn = dendrogram(Z)
    plt.hlines(7500, 0, 200)
    plt.show()

    fc = fcluster(Z, 7500, criterion="distance")
    mat_reordering = np.argsort(fc)

    plt.imshow(sdf_dists[mat_reordering])
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    colors = ["#1b9e77", "#d95f02", "#7570b3", "red"]
    for i in range(num_members):
        plt.plot(members_contours[i][:,1], members_contours[i][:,0], c=colors[fc[i]-1])
    plt.axis("off")
    plt.tight_layout()
    plt.show()


