"""
In this experiment we explore different methods
to find local variations in an ensemble of iso-contours.

Inputs:
- Ensemble of iso-contours (un-parametrized lines in 2D grid)

Assumptions:
- We have a "good" clustering of the iso-contours
- Each cluster has a median and a mean shape
- We know which are the outlier shapes in the data

Conditions:
- SDF-based frechet comparison: suppose we have two iso-lines
  (could be two representatives of our dataset) and we want
  to know, with respect to a user-selected one, where the line
  is varying the most.
  We can parametrize the selected line and, for each point in the
  line, compute its distance to the other line boundary.
  Performing pairwise comparison would be too expensive because we
  would need to wind a registration between the two lines' parametrizations.
  Nevertheless, we have the SDFs of both lines so we can use this
  information to look up distances.
"""

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt

from backend.src.datasets.simple_shapes_datasets import get_multimodal_dataset
from backend.src.utils import get_distance_transform

# General data loading

num_members = 100
num_cols = num_rows = 300

masks, mlab, olab = get_multimodal_dataset(num_cols, num_rows, num_members)
cluster_ids = np.where(np.array(mlab) == 2)[0]
sdfs = [get_distance_transform(m, tf_type="signed") for m in masks]
sdfs_abs = [np.abs(get_distance_transform(m, tf_type="signed")) for m in
            masks]  # needed to extract band around iso-line

cluster_masks = [masks[i] for i in cluster_ids]
cluster_sdfs = [sdfs[i] for i in cluster_ids]
cluster_sdfs_abs = [sdfs_abs[i] for i in cluster_ids]

# Create parametrizations (marching squares)

num_points_parametrization = 100

ref_mask_type = ["member", "mean", "median"][1]

if ref_mask_type == "member":
    ref_mask = cluster_masks[10]
    ref_sdf = cluster_sdfs[10]

elif ref_mask_type == "mean":
    mean_mask = np.array([m.flatten() for m in cluster_masks]).mean(axis=0).reshape(cluster_masks[0].shape)
    ref_mask = mean_mask.copy()
    ref_mask[mean_mask >= 0.5] = 1
    ref_mask[mean_mask < 0.5] = 0
    ref_sdf = get_distance_transform(ref_mask, tf_type="signed")

ref_contour = find_contours(ref_mask, 0.5)  # full contour, we don't necessarily want to use it all
ref_sp = []
for contour in ref_contour:
    sampling_rate = contour.shape[0] // num_points_parametrization  # TODO: too rough
    ref_sp.append(contour[::sampling_rate, :])  # sampling points

# find distances from representative (ref) to all other iso contours
# find distances from all other iso-contours to ref
from scipy.interpolate import RegularGridInterpolator

member_contours = []
member_sps = []
dists_ref_member = []
dists_member_ref = []
ref_sdf_sampler = RegularGridInterpolator((np.arange(ref_sdf.shape[0]), np.arange(ref_sdf.shape[1])), ref_sdf)
for i in range(len(cluster_masks)):
    sdf = cluster_sdfs[i]
    contours = find_contours(cluster_masks[i], 0.5)
    sps = []
    for contour in contours:
        sampling_rate = contour.shape[0] // num_points_parametrization
        sps.append(contour[::sampling_rate, :])  # sampling points
    member_contours.append(contours)
    member_sps.append(sps)

    sdf_sampler = RegularGridInterpolator((np.arange(sdf.shape[0]), np.arange(sdf.shape[1])), sdf)

    drm = []
    for sp in ref_sp:
        drm.append(sdf_sampler(sp))
    dists_ref_member.append(drm)
    dmr = []
    for sp in sps:
        dmr.append(ref_sdf_sampler(sp))
    dists_member_ref.append(dmr)
del sdf, contours, sps, contour, sampling_rate, drm, dmr

# PLOT: dataset overview: reference vs other members
plt.imshow(np.zeros_like(ref_mask), alpha=1, cmap="gray")
for c in ref_contour:
    plt.plot(c[:, 1], c[:, 0], c="turquoise", zorder=1)
for i, contours in enumerate(member_contours):
    color = plt.cm.get_cmap("rainbow")(0)
    for c in contours:
        plt.plot(c[:, 1], c[:, 0], c=color, alpha=0.3, zorder=0)
plt.show()
del c, color

# PLOT: specific cases
member_id = -1
ref_to_member_dists = [np.abs(drm) for drm in dists_ref_member[member_id]]
member_to_ref_dists = [np.abs(dmr) for dmr in dists_member_ref[member_id]]

plt.imshow(np.zeros_like(ref_mask), alpha=1, cmap="gray")

for i, c in enumerate(ref_contour):
    plt.plot(c[:, 1], c[:, 0], linewidth=2, c="turquoise", zorder=0)
    color = ["turquoise", ref_to_member_dists[i]][1]
    plt.scatter(ref_sp[i][:, 1], ref_sp[i][:, 0], s=20, c=color, linewidths=1, edgecolors="white", zorder=1)

for i, c in enumerate(member_contours[member_id]):
    plt.plot(c[:, 1], c[:, 0], linewidth=2, c="pink", zorder=0)
    color = ["pink", member_to_ref_dists[i]][1]
    plt.scatter(member_sps[member_id][i][:, 1], member_sps[member_id][i][:, 0], s=20, c=color, linewidths=1,
                edgecolors="white", zorder=1)
plt.colorbar()
plt.show()
del member_id, ref_to_member_dists, member_to_ref_dists, c, color

# PLOT: distance profiles
member_id = -1
ref_to_member_dists = np.array([np.abs(drm) for drm in dists_ref_member[member_id]]).flatten()
member_to_ref_dists = np.array([np.abs(dmr) for dmr in dists_member_ref[member_id]]).flatten()

highlight_threshold = 12
highlight_idx_rm = np.where(ref_to_member_dists > highlight_threshold)[0]
highlight_idx_mr = np.where(member_to_ref_dists > highlight_threshold)[0]

fig, axs = plt.subplots(nrows=2, figsize=(9, 7))
xrange = np.arange(ref_to_member_dists.shape[0])
axs[0].bar(x=xrange, height=ref_to_member_dists)
axs[0].bar(x=xrange[highlight_idx_rm], height=ref_to_member_dists[highlight_idx_rm], color="red")
axs[0].hlines(highlight_threshold, xrange.min(), xrange.max(), "black")
axs[0].set_title(f"Distance from reference to member {member_id}")
xrange = np.arange(member_to_ref_dists.shape[0])
axs[1].bar(x=xrange, height=member_to_ref_dists)
axs[1].bar(x=xrange[highlight_idx_mr], height=member_to_ref_dists[highlight_idx_mr], color="red")
axs[1].hlines(highlight_threshold, xrange.min(), xrange.max(), "black")
axs[1].set_title(f"Distance from member {member_id} to reference")
plt.show()
del member_id, ref_to_member_dists, member_to_ref_dists, highlight_threshold, highlight_idx_mr, highlight_idx_rm, fig, axs, xrange

# Simplified line plot
# We have one parametrization for the reference line
#  - we know the SDF value of all other members at this parametrization's points
# We have one parametrization for each member of the cluster
#  - we know the SDF value at the references SDF for each member
# The distance arrays in both directions have different shapes
# Therefore, we don't use them to compare, but only for the rendering process

# The proposed idiom has multiple components
# 1. A spine, which is a thin line plot of the representative
# 2. At each sampling point, we compute the negative and positive distances
#    we use these to get the local signed mean and spread of surrounding members
#    we use these quantities in the rendering process to plot two bands, one to each
#    side of the spine. The mean controls the band's width and the spread the opacity.
#    Note: we define a threshold to leave out elements that are further apart and
#          we only use the members within the threshold to compute the means
# 3. For each other member we also do a line plot. But we want to render the segments
#    only if they are beyond the specified mean threshold. In the rest of the plot
#    we set the segments opacity to zero. The fall off of the alpha is gradual.

from matplotlib.collections import LineCollection

band_threshold = 20

# define parametrization
ref_contour_points = ref_sp.copy()
ref_connectivities = []
for cps in ref_contour_points:
    connectivity = np.concatenate([np.arange(cps.shape[0]).reshape(-1, 1),
                                   np.concatenate([np.arange(cps.shape[0])[1:], np.zeros(1)]).reshape(-1, 1)], axis=1)
    ref_connectivities.append(connectivity)
del connectivity

# FEATURE EXTRACTION

# Per point

# - mean
p_means = []
p_inner_means = []
p_outer_means = []
for i, cps in enumerate(ref_contour_points):
    mean_dists_ref_member = np.zeros(cps.shape[0])
    mean_inner_dists_ref_member = np.zeros(cps.shape[0])
    mean_outer_dists_ref_member = np.zeros(cps.shape[0])
    counter = np.zeros(cps.shape[0])
    inner_counter = np.zeros(cps.shape[0])
    outer_counter = np.zeros(cps.shape[0])
    for member_id, dists_collection in enumerate(dists_ref_member):
        dists = dists_collection[i]
        for j, dist in enumerate(dists):
            counter[j] += 1
            mean_dists_ref_member[j] += dist
            if dist >= 0:
                outer_counter[j] += 1
                mean_outer_dists_ref_member[j] += dist
            else:
                inner_counter[j] += 1
                mean_inner_dists_ref_member[j] += dist
    p_means.append(mean_dists_ref_member / (counter + np.finfo(float).eps))
    p_inner_means.append(mean_inner_dists_ref_member / (inner_counter + np.finfo(float).eps))
    p_outer_means.append(mean_outer_dists_ref_member / (outer_counter + np.finfo(float).eps))

del i, cps, mean_dists_ref_member, mean_inner_dists_ref_member, mean_outer_dists_ref_member, counter, inner_counter, outer_counter, member_id, dists_collection, dists

# - spread
p_stds = []
p_inner_stds = []
p_outer_stds = []
for i, cps in enumerate(ref_contour_points):
    std_dists_ref_member = np.zeros(cps.shape[0])
    std_inner_dists_ref_member = np.zeros(cps.shape[0])
    std_outer_dists_ref_member = np.zeros(cps.shape[0])
    counter = np.zeros(cps.shape[0])
    inner_counter = np.zeros(cps.shape[0])
    outer_counter = np.zeros(cps.shape[0])
    for member_id, dists_collection in enumerate(dists_ref_member):
        dists = dists_collection[i]
        for j, dist in enumerate(dists):
            counter[j] += 1
            std_dists_ref_member[j] += np.square(dist - p_means[i][j])
            if dist >= 0:
                outer_counter += 1
                std_outer_dists_ref_member[j] += np.square(dist - p_outer_means[i][j])
            else:
                inner_counter += 1
                std_inner_dists_ref_member[j] += np.square(dist - p_inner_means[i][j])
    p_stds.append(np.sqrt(std_dists_ref_member / (counter - 1)))
    p_inner_stds.append(np.sqrt(std_inner_dists_ref_member / (inner_counter - 1)))
    p_outer_stds.append(np.sqrt(std_outer_dists_ref_member / (outer_counter - 1)))

del i, cps, std_dists_ref_member, std_inner_dists_ref_member, std_outer_dists_ref_member, counter, inner_counter, outer_counter, member_id, dists_collection, dists

# PLOTTING
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from numpy.linalg import norm

# plot the defined parametrization using line collections
linewidth_factor = 10
fig, ax = plt.subplots()
rots = []
for i, (cps, connects) in enumerate(zip(ref_contour_points, ref_connectivities)):
    # plt.scatter(cps[:, 1], cps[:, 0])
    segments = []
    linewidths = []
    colors = []

    rectangles_outer = []
    rectangles_inner = []
    for j in range(connects.shape[0]):
        p0 = int(connects[j, 0])
        p1 = int(connects[j, 1])

        # line collection
        segments.append(((cps[p0, 1], cps[p0, 0]), (cps[p1, 1], cps[p1, 0])))
        lw = (p_means[i][p0] + p_means[i][p1]) / 2
        color = list(plt.cm.get_cmap("rainbow")(0))
        color[-1] = (p_stds[i][p0] + p_stds[i][p1]) / 2
        linewidths.append(lw * linewidth_factor)
        colors.append(color)

        # rectangles
        seg_vec = (cps[p1] - cps[p0])
        seg_vec_len = norm(seg_vec)
        seg_vec = seg_vec / seg_vec_len
        axis_vec = np.array([1, 0])
        rotation = np.arccos(np.clip(np.dot(seg_vec, axis_vec), -1.0, 1.0))
        rotation = np.rad2deg(rotation)
        roh = (p_outer_means[i][p0] + p_outer_means[i][p1]) / 2
        rih = (p_inner_means[i][p0] + p_inner_means[i][p1]) / 2
        rectangles_outer.append(
            Rectangle(cps[p0], width=seg_vec_len, height=lw * linewidth_factor, angle=rotation, facecolor="green"))
        rectangles_inner.append(
            Rectangle(cps[p0], width=seg_vec_len, height=lw * linewidth_factor, angle=rotation + 180, facecolor="red"))
        rots.append(rotation)

    lc = LineCollection(segments=segments, linewidths=linewidths, colors=colors)
    pc_outer = PatchCollection(rectangles_outer)
    pc_inner = PatchCollection(rectangles_inner)

    ax.add_collection(lc)
    # ax.add_collection(pc_outer)
    # ax.add_collection(pc_inner)

ax.set_xlim(0, num_cols)
ax.set_ylim(0, num_rows)
ax.set_axis_off()
plt.show()

del linewidth_factor, fig, ax, i, cps, connects, segments, linewidths, j, p0, p1, lw, lc

# lc = LineCollection(segments=(((0,0), (10,10)), ((10,10), (15,5))))

from numpy.linalg import norm

fig, ax = plt.subplots()
ax.imshow(np.zeros_like(ref_mask), cmap="gray")
# ax.imshow(cluster_masks[-1], cmap="gray")

# spine
ax.plot(ref_contour[:, 1], ref_contour[:, 0], c="pink")

# outer band
points = ref_sp.copy()
normals = points[:-1] - points[1:]
normals = normals / norm(normals)
normals = normals[:, [1, 0]]
normals[:, 0] *= -1

# inner normals
inner_normals = normals.copy()
inner_normals *= -20 * mean_inner_dists_ref_member[:-1].reshape(-1, 1)
inner_normals += points[:-1]
ax.plot(inner_normals[:, 1], inner_normals[:, 0], c="purple")

# outer normals
outer_normals = normals.copy()
outer_normals *= -1
outer_normals *= 20 * mean_outer_dists_ref_member[:-1].reshape(-1, 1)
outer_normals += points[:-1]
ax.plot(outer_normals[:, 1], outer_normals[:, 0], c="purple")

# # lines of other members
# #members_lc = []
# for member_id, member_dists in enumerate(dists_member_ref):
#     idx_lg_dev = np.where(np.abs(dists_member_ref[member_id]) > 20)[0]
#     points_lg_dev = member_sps[member_id][idx_lg_dev]
#     plt.scatter(points_lg_dev[:, 1], points_lg_dev[:, 0], s=1, c="red")

plt.show()

# lines of the reference
# ref_lc = LineCollection(segments=segments, linewidths=lwidths, zorder=0)
# ax.add_collection(ref_lc)
# ax.scatter(ref_sp[:, 1], points[:, 0], c="yellow", s=5, zorder=1)
#

# trimming_margin = 20
# sdfs_trimmed = [np.clip(sdf, -20, 20) for sdf in sdfs]
#
#
# mask = masks[70]
# sdf = sdfs[70]
# sdf_trimmed = sdfs_trimmed[70]
#
# fig, axs = plt.subplots(nrows=1, ncols=3)
# axs[0].imshow(mask)
# axs[1].imshow(sdf)
# axs[2].imshow(np.abs(sdf_trimmed))
# plt.show()
#
# id_a, id_b = [0, 99]
# masks_diff = masks[id_a] - masks[id_b]
# sdfs_diff = sdfs[id_a] - sdfs[id_b]
# trimmed_sdfs_diff = np.clip(sdfs[id_a], -20, 20) - np.clip(sdfs[id_b], -20, 20)
# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8, 5))
# axs[0, 0].imshow(masks[id_a], cmap="gray")
# axs[0, 0].set_title(f"Shape a (id: {id_a})")
# axs[0, 1].imshow(masks[id_b], cmap="gray")
# axs[0, 1].set_title(f"Shape b (id: {id_b})")
# axs[1, 0].imshow(masks_diff)
# axs[1, 0].set_title("Difference of masks")
# axs[1, 1].imshow(sdfs_diff)
# axs[1, 1].set_title("Difference of SDFs")
# axs[1, 2].imshow(trimmed_sdfs_diff)
# axs[1, 2].set_title("Difference of clipped SDFs")
# for ax in axs.flatten():
#     ax.set_axis_off()
# plt.show()
#
#
# trimmed_sdfs_diff1 = np.clip(sdfs[0], -20, 20) - np.clip(sdfs[85], -20, 20)
# trimmed_sdfs_diff1 = trimmed_sdfs_diff1 / trimmed_sdfs_diff1.max()
# trimmed_sdfs_diff2 = np.clip(sdfs[0], -20, 20) - np.clip(sdfs[99], -20, 20)
# trimmed_sdfs_diff2 = trimmed_sdfs_diff2 / trimmed_sdfs_diff2.max()
# plt.imshow(trimmed_sdfs_diff1 + trimmed_sdfs_diff2)
# plt.show()
#
#
# cluster_selection = np.arange(len(mlab))#np.where(np.array(mlab) == 2)[0]
# c_masks = [masks[i] for i in cluster_selection]
# c_sdfs = [sdfs[i] for i in cluster_selection]
# c_mean = np.array([m.flatten() for m in c_masks]).mean(axis=0).reshape((num_rows, num_cols))
# c_mean_t = c_mean.copy()
# c_mean_t[c_mean >= 0.5] = 1
# c_mean_t[c_mean < 0.5] = 0
#
# repr = c_mean_t.copy()
# #repr = masks[79].copy()
# repr_conts = find_contours(repr, 0.5)[0]
# repr_conts = repr_conts[::16, :]
#
# plt.imshow(c_mean)
# plt.scatter(repr_conts[:, 1], repr_conts[:, 0], c="red", s=1)
# plt.show()
#
#
# # Build a vector for each mask of size c_mean_conts.shape[0] * neighborhood_size
# # For every dot in c_mean_conts check the neighborhood in mask_i sdf, flatten it and put it in vector
#
# from sklearn.decomposition import PCA
#
# patch_size = 1
# sdf_vec_arr = []
# for i in range(len(c_masks)):
#     neighborhood_vec_arr = []
#     for j in range(repr_conts.shape[0]):
#         p = repr_conts[j]
#         r0 = int(p[1] - patch_size/2)
#         r1 = r0 + patch_size
#         c0 = int(p[0] - patch_size/2)
#         c1 = c0 + patch_size
#         patch = c_sdfs[i][r0:r1, c0:c1].flatten()
#         patch_value = patch.mean()
#         neighborhood_vec_arr.append(patch_value)
#     sdf_vec_arr.append(np.array(neighborhood_vec_arr))
# sdf_vec_arr = np.array(sdf_vec_arr)
#
# sdf_vec_arr = sdf_vec_arr - sdf_vec_arr.mean(axis=0).reshape(1, -1)  # centering
# sdf_vec_arr = sdf_vec_arr / sdf_vec_arr.std(axis=0).reshape(1, -1)
#
# plt.matshow(sdf_vec_arr)
# plt.show()
#
# pca = PCA()
# pca.fit(sdf_vec_arr)
#
# transformed_mat = np.matmul(sdf_vec_arr, pca.components_)
# plt.scatter(transformed_mat[:,0], transformed_mat[:,1], c=mlab)
# plt.show()
#
# print(pca.components_.shape)
# print(pca.explained_variance_ratio_)
#
# #pca_component = np.abs(pca.components_[0])
# component_id = 4
# pca_component = pca.components_[component_id]
# fig, axs = plt.subplots(ncols=2, figsize=(7, 4))
# axs[0].imshow(c_mean)
# axs[0].scatter(repr_conts[:, 1], repr_conts[:, 0], c="red", s=2)
# axs[0].set_title("Parametrized shape")
#
# axs[1].imshow(c_mean)
# axs[1].scatter(repr_conts[:, 1], repr_conts[:, 0], c=pca.mean_ + 2 * np.sqrt(pca.singular_values_[component_id]) * pca_component, s=5)
# axs[1].set_title("Parametrized shape \n"
#                  f"Colors: {component_id}th principal component")
#
# for ax in axs.flatten():
#     ax.set_axis_off()
# plt.show()
#
# for i in range(pca.n_components_)[0:10]:
#     plt.plot(pca.components_[i], label=f"{i}")
# plt.legend()
# plt.show()
