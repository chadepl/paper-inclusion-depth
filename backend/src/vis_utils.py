
from functools import reduce
import pandas as pd
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .contour_band_depth import get_band_components

#colors = ["#1b9e77", "#d95f02", "#7570b3", "red"]
colors = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494','#b3b3b3', "red"]


def plot_contour(contour, ax=None, plot_line=True, line_kwargs=None, plot_markers=False, markers_kwargs=None):
    if ax is None:
        pobj = plt
    else:
        pobj = ax

    if line_kwargs is None:
        line_kwargs = {"color": "black"}

    if markers_kwargs is None:
        markers_kwargs = {"color": "black"}

    for c in contour:
        if plot_line:
            pobj.plot(c[:, 1], c[:, 0], **line_kwargs)
        if plot_markers:
            pobj.scatter(c[:, 1], c[:, 0], **markers_kwargs)


def plot_contour_spaghetti(masks, memberships=None, highlight=None, ax=None, alpha=0.5, linewidth=1):
    num_members = len(masks)
    contours = [find_contours(m, 0.5) for m in masks]
    if memberships is None:
        memberships = [0 for _ in range(num_members)]

    if reduce(lambda a, b: a == b, [type(m) is int for m in memberships]) and type(memberships[0]) is int:
        cs = [colors[m] for m in memberships]
    else:
        cs = [cm.magma(m) for m in memberships]

    if highlight is None:
        highlight = list()
    elif type(highlight) is int:
        highlight = [highlight, ]

    if ax is None:
        fig, ax = plt.subplots(layout="tight", figsize=(10, 10))

    for i, contour in enumerate(contours):
        plot_contour(contour, line_kwargs=dict(linewidth=linewidth, color=cs[i], alpha=alpha, zorder=0))

    for i in highlight:
        plot_contour(contours[i], plot_line=False, plot_markers=True, markers_kwargs=dict(color="red", s=1, zorder=1))

    ax.set_axis_off()
    return ax


def plot_contour_boxplot(masks, cbp_depths, epsilon_out=0.3, ax=None):
    """
    Renders a contour boxplot using depth data and the provided masks.
    If a list of member_ids is supplied (subset_idx), the contour boxplot
    is constructed only from these elements.
    TODO: implement automatic ways of determining epsilon_th and epsilon_out and set them as default
    """

    # - classification
    cbp_median = np.array([np.argmax(cbp_depths), ])
    cbp_outliers = np.where(cbp_depths <= epsilon_out)[0]  # should be 0
    cbp_band100 = np.setdiff1d((np.argsort(cbp_depths)[::-1]),
                               np.union1d(cbp_median, cbp_outliers))
    cbp_band50 = cbp_band100[:cbp_depths.size // 2]

    cbp_bands = np.setdiff1d(np.arange(cbp_depths.size), np.union1d(cbp_outliers, cbp_median))

    cbp_classification = np.zeros_like(cbp_depths)
    cbp_classification[cbp_median] = 0
    cbp_classification[cbp_bands] = 1
    cbp_classification[cbp_outliers] = 2

    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10), layout="tight")
    ax.set_axis_off()

    # Visual encoding
    if len(cbp_band50 >= 2):
        b50 = get_band_components(masks, cbp_band50)["band"]
    if len(cbp_band100 >= 2):
        b100 = get_band_components(masks, cbp_band100)["band"]

    for outlier_id in cbp_outliers:
        contours = find_contours(masks[outlier_id])
        plot_contour(contours, line_kwargs=dict(c="red", linestyle="dashed"), ax=ax)

    if len(cbp_band100 >= 2):
        contours = find_contours(b100)
        plot_contour(contours, line_kwargs=dict(c="purple", linewidth=2), ax=ax)

    if len(cbp_band50 >= 2):
        contours = find_contours(b50)
        plot_contour(contours, line_kwargs=dict(c="plum", linewidth=2), ax=ax)

    contours = find_contours(masks[cbp_median[0]])
    plot_contour(contours, line_kwargs=dict(c="gold", linewidth=3), ax=ax)

    # trimmed mean
    cbp_trimmed_mean_idx = np.setdiff1d((np.argsort(cbp_depths)[::-1]), cbp_outliers)
    masks_arr = np.array([m.flatten() for m in [masks[i] for i in cbp_trimmed_mean_idx]])
    masks_mean = masks_arr.mean(axis=0)
    contours = find_contours(masks_mean.reshape(masks[0].shape), level=0.5)
    plot_contour(contours, line_kwargs=dict(c="dodgerblue", linewidth=3), ax=ax)

    return ax


def plot_grid_masks(masks, ax=None, cmap="gray"):
    num_members = len(masks)
    num_rows, num_cols = masks[0].shape
    num_rows_grid = int(np.floor(np.sqrt(num_members)))
    num_cols_grid = int(np.ceil(np.sqrt(num_members)))

    if ax is None:
        fig, ax = plt.subplots()

    canvas = np.zeros((num_rows_grid * num_rows, num_cols_grid * num_cols))

    for mid, mask in enumerate(masks):
        r = mid // num_cols_grid
        c = mid - r * num_cols_grid

        r0 = r * num_rows
        r1 = r0 + num_rows
        c0 = c * num_cols
        c1 = c0 + num_cols

        canvas[r0:r1, c0:c1] = mask

    ax.imshow(canvas, cmap=cmap)

    ax.set_axis_off()

    return ax


def plot_band_checking_procedure(member_masks, subset_data, member_id=0, subset_id=0):

    member = member_masks[member_id]
    subset = subset_data[subset_id]
    subset_bc = subset["band_components"]
    subset_members = [member_masks[i] for i in subset["idx"]]

    member_contours = find_contours(member, 0.5)
    subset_member_contours = [find_contours(s, 0.5) for s in subset_members]

    fig, axs = plt.subplots(ncols=3, figsize=(10, 4))

    axs[0].imshow(subset_bc["band"], cmap="gray")
    plot_contour(member_contours, line_kwargs=dict(color="#1b9e77"), ax=axs[0])
    axs[0].set_axis_off()
    axs[0].set_title(f"Band formed by {subset['idx']}")

    axs[1].imshow(subset_bc["intersection"], cmap="gray")
    plot_contour(member_contours, line_kwargs=dict(color="#1b9e77"), ax=axs[1])
    for m in subset_member_contours:
        plot_contour(m, line_kwargs=dict(color="#d95f02"), ax=axs[1])
    axs[1].set_axis_off()
    axs[1].set_title(f"lc_frac \n intersect \in member")

    axs[2].imshow(subset_bc["union"], cmap="gray")
    plot_contour(member_contours, line_kwargs=dict(color="#1b9e77"), ax=axs[2])
    for m in subset_member_contours:
        plot_contour(m, line_kwargs=dict(color="#d95f02"), ax=axs[2])

    axs[2].set_axis_off()
    axs[2].set_title(f"rc_frac \n member \in union")

    fig.suptitle("Is the member contained in the band?")

    return fig, axs




def plot_dendrogram(model, **kwargs):
    from scipy.cluster.hierarchy import dendrogram
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    plt.show()