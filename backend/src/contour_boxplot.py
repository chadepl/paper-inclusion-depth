"""
Input:
 - Ensemble [Array, ...]
 - Js = [2, ]

Output:
    contour_boxplot = {
        median: {data: Array},
        band50: {data: Array},
        band100: {data: Array},
        outliers: [{data: Array}, ]
        depth_data = [{}, ] // matrix of information to data depths
    }
"""

from itertools import combinations
import numpy as np


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

def get_depth_data(ensemble_members, subset_sizes=(2,), tidy_depth_data=True):

    contour_boxplot = dict()
    num_members = len(ensemble_members)
    member_idx = np.arange(len(ensemble_members)).tolist()

    # Find combinations of ensemble members to form bands

    subsets = []
    for ss in subset_sizes:
        subsets += combinations(member_idx, ss)
    num_subsets = len(subsets)

    # Compute fractional containment table

    # - Compute per-subset components for fractional containment table
    band_components = []
    for subset_id, subset_idx in enumerate(subsets):
        band_components.append(get_band_components(ensemble_members, subset_idx))

    # - Get tidy fractional containment table
    subset_sizes = []
    left_containment_array = np.zeros((num_members, num_subsets))
    right_containment_array = np.zeros((num_members, num_subsets))
    for member_id, member in enumerate(ensemble_members):
        for subset_id, subset_idx in enumerate(subsets):
            subset_sizes.append(len(subset_idx))

            bc = band_components[subset_id]
            intersection = bc["intersection"]
            union = bc["union"]

            left_containment_array[member_id, subset_id] = (intersection != member).sum() / member.sum()
            right_containment_array[member_id, subset_id] = (member != union).sum() / union.sum()

    #  - If tidy is true, tidy data matrices
    if tidy_depth_data:
        depth_data = []
        for member_id, member in enumerate(ensemble_members):
            for subset_id, subset_idx in enumerate(subsets):
                depth_data.append(
                    dict(member_id=member_id,
                         subset_id=subset_id,
                         subset_size=subset_sizes[subset_id],
                         lc_frac=left_containment_array[member_id, subset_id],
                         rc_frac=right_containment_array[member_id, subset_id])
                )

    # Assign variables
    contour_boxplot["num_members"] = num_members
    contour_boxplot["num_subsets"] = num_subsets
    if tidy_depth_data:
        contour_boxplot["depth_data"] = depth_data
    else:
        contour_boxplot["lc_array"] = left_containment_array.flatten().tolist()
        contour_boxplot["rc_array"] = right_containment_array.flatten().tolist()
        contour_boxplot["subset_sizes"] = subset_sizes

    return contour_boxplot


if __name__ == "__main__":

    from ellipse_generation import generate_ensemble_ellipses

    ensemble_data = generate_ensemble_ellipses(20, 300, 300)
    ensemble = [m["data"] for m in ensemble_data["ensemble"]["members"]]
    band_components = get_contour_boxplot(ensemble)

    mat_cells = [c for c in band_components if c["subset_size"] == 2]
    mat = np.zeros((20, 190))
    for c in mat_cells:
        mat[c["mid"], c["sid"]] = c["frac_lc"]

    import matplotlib.pyplot as plt

    plt.matshow(mat)
