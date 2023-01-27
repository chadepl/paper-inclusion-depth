"""
Given two members, computes their similarity.
Could also be extended to triplets or subsets of higher
cardinality.
"""

import numpy as np

from utils import get_border_raster, get_distance_transform

def dice_coefficient(arr1, arr2):
    """
    TODO Weight it with sdf or other fields
    """
    A = (arr1.flatten() > 0.5)
    B = (arr2.flatten() > 0.5)
    num = 2 * np.logical_and(A, B).sum()
    den = A.sum() + B.sum()
    return num / den

def surface_dice_coefficient(mask1, mask2, tolerance=10):

    # Get the borders of the masks as raster
    cm1_raster = get_border_raster(mask1)
    cm2_raster = get_border_raster(mask2)

    # Compute distance transform
    m1_dt = get_distance_transform(np.logical_not(cm1_raster))
    m2_dt = get_distance_transform(np.logical_not(cm2_raster))

    # Mask surfaces with distance tolarances
    m1_bm = m2_dt.copy()
    m1_bm[m2_dt > tolerance] = 0
    m1_bm[m2_dt <= tolerance] = 1
    m1_sm_raster = cm1_raster * m1_bm

    m2_bm = m1_dt.copy()
    m2_bm[m1_dt > tolerance] = 0
    m2_bm[m1_dt <= tolerance] = 1
    m2_sm_raster = cm2_raster * m2_bm

    # Compute surface dice coefficient
    sdc_num = m1_sm_raster.sum() + m2_sm_raster.sum()
    sdc_den = cm1_raster.sum() + cm2_raster.sum()

    return sdc_num / sdc_den


##########
# D-MATS #
##########

def get_similarity_matrix(ensemble_masks, metric="dice", metric_kwargs=None, tidy=True):
    if metric_kwargs is None:
        metric_kwargs = dict()

    num_members = len(ensemble_masks)
    pairs = combinations(np.arange(num_members), 2)
    out = []
    for pair in pairs:
        row = dict()
        row["member_i"] = pair[0]
        row["member_j"] = pair[1]
        if metric == "dice":
            row["value"] = dice_coefficient(ensemble_masks[pair[0]], ensemble_masks[pair[1]], **metric_kwargs)
        elif metric =="surface_dice":
            row["value"] = surface_dice_coefficient(ensemble_masks[pair[0]], ensemble_masks[pair[1]], **metric_kwargs)
        out.append(row)

    if not tidy:
        out_mat = np.ones((num_members, num_members))
        for row in out:
            mi = row["member_i"]
            mj = row["member_j"]
            out_mat[mi, mj] = row["value"]
            out_mat[mj, mi] = row["value"]
        return out_mat

    return out


def get_similarity_based_depths(ensemble_members, similarities, tidy=True):
    num_members = len(ensemble_members)
    subsets = list(combinations(np.arange(len(ensemble_members)), 2))
    num_subsets = len(subsets)
    depths = np.zeros((num_members, num_subsets))
    if tidy:
        out_tidy = []
    for member_id in np.arange(len(ensemble_members)):
        for subset_id, subset in enumerate(subsets):
            d12 = similarities[member_id, subset[0]]
            d13 = similarities[member_id, subset[1]]
            d23 = similarities[subset[0], subset[1]]
            #depths[member_id, subset_id] = 1.0 if np.mean([d12, d13]) <= d23 else 0.0
            depths[member_id, subset_id] = np.mean([d12, d13])
            if tidy:
                out_tidy.append(dict(member_id=member_id, subset_id=subset_id, subset=subset, value=depths[member_id, subset_id]))

    if tidy:
        return out_tidy

    return depths


if __name__ == "__main__":

    from skimage.measure import find_contours
    import matplotlib.pyplot as plt

    from ellipse_generation import load_ensemble_ellipses
    from utils import get_border_raster

    demo = ["surface-dice"][0]

    if demo == "surface-dice":
        num_members = 25
        num_cols = num_rows = 300
        ensemble = load_ensemble_ellipses(num_members, num_cols, num_rows, params_set=1)

        member_masks = [m["data"] for m in ensemble["ensemble"]["members"]]
        member_contours = [find_contours(m, 0.5)[0] for m in member_masks]

        # (0, 1), (0,23), (0,24)
        mid1 = 0
        mid2 = 24

        mask1 = member_masks[mid1]
        mask2 = member_masks[mid2]

        contour1 = member_contours[mid1]
        contour2 = member_contours[mid2]

        border1 = get_border_raster(mask1)
        border2 = get_border_raster(mask2)

        sdf1 = get_distance_transform(np.logical_not(border1))
        sdf2 = get_distance_transform(np.logical_not(border2))

        dists1 = border2 * sdf1
        dists1 = dists1[np.where(dists1 > 0)]
        dists2 = border1 * sdf2
        dists2 = dists2[np.where(dists2 > 0)]

        fig, axs = plt.subplots(ncols=2, nrows=4, layout="tight", figsize=(6, 8))
        axs[0, 0].imshow(mask1)
        axs[1, 0].imshow(border1)
        axs[2, 0].imshow(sdf1)
        axs[2, 0].plot(contour2[:, 1], contour2[:, 0], c="red")

        axs[0, 1].imshow(mask2)
        axs[1, 1].imshow(border2)
        axs[2, 1].imshow(sdf2)
        axs[2, 1].plot(contour1[:, 1], contour1[:, 0], c="red")

        for ax in axs.flatten():
           ax.set_axis_off()

        axs[3, 0].hist(dists1, bins=30, range=[0, num_rows])
        axs[3, 1].hist(dists2, bins=30, range=[0, num_rows])

        axs[3, 0].set_axis_on()
        axs[3, 1].set_axis_on()

        plt.show()