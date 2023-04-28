

from time import time
import numpy as np
import pandas as pd
from skimage.draw import ellipse, rectangle
from skimage.measure import find_contours
from scipy.interpolate import RegularGridInterpolator
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

from backend.src.utils import get_distance_transform
from backend.src.contour_depths import band_depth, border_depth
from backend.src.datasets.circles import circles_with_outliers, circles_different_radii_distribution
from backend.src.datasets.bd_paper import get_contaminated_contour_ensemble_shape, get_contaminated_contour_ensemble_magnitude, get_contaminated_contour_ensemble_center
from backend.src.datasets.papers import ensemble_ellipses_cbp
from backend.src.datasets.han_ensembles import get_han_slice_ensemble
from backend.src.vis_utils import plot_contour_spaghetti, plot_contour_boxplot

# Depth setup
overview = ["band_depth", "boundary_depth"][1]

if overview == "band_depth":
    labels = [
        "strict",
        "modified",
        "modified_t"]
    fns = [
        lambda ensemble: band_depth.compute_depths(ensemble, modified=False, target_mean_depth=None),
        lambda ensemble: band_depth.compute_depths(ensemble, modified=True, target_mean_depth=None),
        lambda ensemble: band_depth.compute_depths(ensemble, modified=True, target_mean_depth=1/6)
    ]
    vmin, vmax = [0.0, 1.0]
elif overview == "boundary_depth":
    labels = [
        "base",
        "strict-nest",
        "modified-nest",
        "modified-l2",
        #"fast-strict",
        #"fast-modified"
    ]
    fns = [
        lambda ensemble: border_depth.compute_depths(ensemble, modified=False, global_criteria=None),
        lambda ensemble: border_depth.compute_depths(ensemble, modified=False, global_criteria="nestedness"),
        lambda ensemble: border_depth.compute_depths(ensemble, modified=True, global_criteria="nestedness"),
        lambda ensemble: border_depth.compute_depths(ensemble, modified=True, global_criteria="l2_dist"),
        #lambda ensemble: border_depth.compute_depths_fast(ensemble, modified=False),
        #lambda ensemble: border_depth.compute_depths_fast(ensemble, modified=True)
    ]
    vmin, vmax = [0.0, 1.0]

# Data loading
num_members = 50
num_rows = num_cols = 300
pos = []

#ensemble = circles_different_radii_distribution(num_members, num_rows, num_cols)
#ensemble = get_contaminated_contour_ensemble_center(num_members, num_rows, num_cols)
ensemble = get_contaminated_contour_ensemble_magnitude(num_members, num_rows, num_cols, case=1)
#ensemble = get_contaminated_contour_ensemble_shape(num_members, num_rows, num_cols)
#ensemble = get_contaminated_contour_ensemble_shape(num_members, num_rows, num_cols, return_labels=False, scale=0.01, freq=0.01, p_contamination=0.1)  # shape in
#ensemble = get_contaminated_contour_ensemble_shape(num_members, num_rows, num_cols, return_labels=False, scale=0.05, freq=0.05, p_contamination=0.1) # shape out
#ensemble = circles_with_outliers(num_members, num_rows, num_cols, num_outliers=4)
#ensemble = ensemble_ellipses_cbp(num_members, num_rows, num_cols)
#img, gt, ensemble = get_han_slice_ensemble(num_rows, num_cols)
#ensemble = ensemble[:20]

# Get depths
times = []
depths = []
for l, f in zip(labels, fns):
    times.append(time())
    depths.append(f(ensemble))
    times[-1] = time() - times[-1]
    print(f"{l}: {times[-1]} seconds")


# Plots
fig, axs = plt.subplots(ncols=len(labels), figsize=(len(depths)*4, 4))
for i, d in enumerate(depths):
    axs[i].set_title(labels[i])
    idx_head = np.argmax(d)
    idx_tail = np.argmin(d)
    to_highlight = [idx_head, idx_tail]
    to_highlight = [idx_head, ]
    to_highlight = [idx_tail, ]
    to_highlight = None
    plot_contour_spaghetti(ensemble, highlight=to_highlight, arr=d, is_arr_categorical=False, vmin=vmin, vmax=vmax, ax=axs[i])
    #plot_contour_boxplot(ensemble, depths[i], epsilon_out=10)
    #plot_contour_boxplot(ensemble, depths[i], outlier_type="threshold", epsilon_out=0)
plt.show()


depths_df = pd.DataFrame(depths)
depths_df = depths_df.T
depths_df.columns = labels
pplot = sns.pairplot(depths_df)
pplot.set(xlim=(vmin, vmax), ylim=(vmin, vmax))
plt.show()




# sdfs = [get_distance_transform(m, tf_type="signed") for m in ensemble_members]
# sdfs_interps = [RegularGridInterpolator((np.arange(num_rows), np.arange(num_cols)),  sdf) for sdf in sdfs]
#
# reference_id = 4
# plt.imshow(np.zeros_like(ensemble_members[0]), cmap="gray")
# for i, member in enumerate(ensemble_members):
#     if reference_id == i:
#         color = "orange"
#     else:
#         color = "gray"
#     for c in find_contours(member, 0.5):
#         plt.plot(c[:, 1], c[:, 0], c=color)
# plt.axis("off")
# plt.show()
#
#
#
# # SDFs
#
# plt.imshow(sdfs[reference_id])
# plt.show()
#
# reference_id = 0
# interp_vals_ref_sdf = []  # we check other's contour points in ref SDF
# interp_vals_other_sdf = []  # we check ref contour points in other SDFS
# contour_sdf = find_contours(sdfs[reference_id], level=0)
# for i, sdf in enumerate(sdfs):
#     contour_other = find_contours(sdf, level=0)
#
#     interp_vals_ref_sdf.append(sdfs_interps[reference_id](np.concatenate(contour_other, axis=0)))
#     interp_vals_other_sdf.append(sdfs_interps[i](np.concatenate(contour_sdf, axis=0)))
#
# fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
# axs[0].set_title(f"Check other curves sample points in SDF of contour {reference_id}")
# axs[0].set_ylabel("Distance to closest point in other contour")
# for i, iv in enumerate(interp_vals_ref_sdf):
#     axs[0].plot(iv, label=f"{i}")
# axs[1].set_title(f"Check sample points of contour {reference_id} in other curves SDFs")
# for iv in interp_vals_other_sdf:
#     axs[1].plot(iv)
# axs[0].legend()
# plt.show()
#
#
#
# per_point_depths = []
# for pi in range(len(interp_vals_other_sdf[0])):
#     points = [iv[pi] for iv in interp_vals_other_sdf]
#     d = lp_depth.compute_depths(np.array(points).reshape(-1, 1))
#     per_point_depths.append(d[reference_id])
#
#
#
# # SDF functional depths
# # We will iterate over each reference_id and compute their ego-depths
#
#
#
# t_start = time()
# border_fn = border_depth.get_border_functions(ensemble_members)
# border_depths = border_depth.compute_depths(border_fn)
# t_end = time()
# border_dmat = border_depth.get_border_dmat(border_fn)
# global_depths = np.array(border_depths[0])
# point_depths = [np.array(d) for d in border_depths[1]]
# print(f"Border depths took {t_end - t_start} seconds")
#
# plt.matshow(border_dmat)
# plt.show()
#
# t_start = time()
# band_depths = band_depth.compute_depths(ensemble_members)
# t_end = time()
# print(f"Band depths took {t_end - t_start} seconds")
#
#
# plt.scatter(global_depths, band_depths)
# plt.xlabel("Border depths")
# plt.ylabel("Band depths")
# plt.title("Border vs Band Depths")
# plt.show()
#
#
# fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
# axs[0].imshow(np.zeros_like(sdfs[0]), cmap="gray")
# axs[1].imshow(np.zeros_like(sdfs[0]), cmap="gray")
# axs[0].set_title("Contour Band Depth")
# axs[1].set_title("New Depths")
# axs[0].set_axis_off()
# axs[1].set_axis_off()
# for i, sdf in enumerate(sdfs):
#
#     for c in find_contours(sdf, level=0):
#         axs[0].plot(c[:, 1], c[:, 0], c=mpl.colormaps["inferno"]((band_depths[i] - band_depths.min())/(band_depths.max() - band_depths.min())))
#         axs[1].plot(c[:, 1], c[:, 0], c=mpl.colormaps["inferno"]((global_depths[i] - global_depths.min())/(global_depths.max() - global_depths.min())))
# plt.show()
#
#
# # The new depths also afford a piecewise perspective
# reference_id = 22
# reference_depths = point_depths[reference_id]
# if reference_depths.max() > reference_depths.min():
#     reference_depths = (reference_depths - reference_depths.min())/(reference_depths.max() - reference_depths.min())
# contours = find_contours(ensemble_members[reference_id], 0.5)
#
# fig, ax = plt.subplots()
# ax.imshow(np.zeros_like(sdfs[0]), cmap="gray")
# colors = [mpl.colormaps["inferno"](d) for d in reference_depths]
# for c in contours:
#     ax.plot(c[:, 1], c[:, 0], c="white")
#     ax.scatter(c[:, 1], c[:, 0], c=colors)
# plt.show()
