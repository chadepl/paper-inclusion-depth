"""
How do vanilly boundary depths perform in the four topologies introduced in CBD?
Can we improve this by adding global information?
"""
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
from skimage.draw import disk
import matplotlib.pyplot as plt

from backend.src.contour_depths import border_depth, band_depth
from backend.src.utils import get_distance_transform
from backend.src.vis_utils import plot_contour_spaghetti

num_members = 30
num_outliers = 3
num_rows = num_cols = 300

radii = np.linspace(num_rows // 6, num_rows // 3, num_members - num_outliers)
ensemble = []
for radius in radii:
    mask = np.zeros((num_rows, num_cols))
    rr, cc = disk((num_rows // 2, num_cols // 2), radius=radius, shape=mask.shape)
    mask[rr, cc] = 1
    ensemble.append(mask)

# small one
for i in range(num_outliers):
    mask = np.zeros((num_rows, num_cols))
    d_from_center = ((num_rows // 6) + (num_rows // 3)) / 2 + 5
    angle = np.random.random() * 2 * np.pi
    center_c = d_from_center * np.cos(angle) + num_cols // 2
    center_r = d_from_center * np.sin(angle) + num_rows // 2
    radius = np.random.normal(10, 1)
    rr, cc = disk((center_r, center_c), radius=radius, shape=mask.shape)
    mask[rr, cc] = 1
    ensemble = [mask, ] + ensemble
    # ensemble.append(mask)

sdfs = [get_distance_transform(m, tf_type="signed") for m in ensemble]

bod = border_depth.compute_depths(ensemble)
bad = band_depth.compute_depths(ensemble)

fig, axs = plt.subplots(ncols=2, layout="tight", figsize=(6, 3))
plot_contour_spaghetti(ensemble, arr=bod, is_arr_categorical=False, linewidth=2, ax=axs[0], alpha=1)
plot_contour_spaghetti(ensemble, arr=bad, is_arr_categorical=False, linewidth=2, ax=axs[1], alpha=1)
plt.show()

from scipy.spatial.distance import pdist, squareform

sdf_mat = np.concatenate([sdf.flatten().reshape(1, -1) for sdf in sdfs], axis=0)
sdf_dists = squareform(pdist(sdf_mat))
plt.matshow(sdf_dists)
plt.show()

sdf_agg = sdf_mat.sum(axis=0)
plt.imshow(sdf_agg.reshape(num_rows, -1))
plt.show()

# For distance of 1 to the cloud
updated_cloud = sdf_agg.reshape(num_rows, -1) - sdfs[1]
updated_cloud = updated_cloud / (num_members - 1)
d = np.sqrt(np.sum(np.square(updated_cloud - sdfs[1])))
print(d)

updated_cloud = sdf_agg.reshape(num_rows, -1) - sdfs[-1]
updated_cloud = updated_cloud / (num_members - 1)
d = np.sqrt(np.sum(np.square(updated_cloud - sdfs[-1])))
print(d)

dists_to_cloud = []
for i in range(len(ensemble)):
    updated_cloud = sdf_agg.reshape(num_rows, -1) - sdfs[i]
    updated_cloud = updated_cloud / (num_members - 1)
    d = np.sqrt(np.sum(np.square(updated_cloud - sdfs[i])))
    dists_to_cloud.append(d)

print(dists_to_cloud)
print(dists_to_cloud[:(num_members - num_outliers)])
print(dists_to_cloud[(num_members - num_outliers):])

plt.scatter(x=np.zeros(num_members - num_outliers), y=np.array(dists_to_cloud)[:(num_members - num_outliers)], c="blue")
plt.scatter(x=np.zeros(num_outliers), y=np.array(dists_to_cloud)[(num_members - num_outliers):], c="red")
plt.show()

# Combine depth in SDF space with combinatorial function depth
# The strategy is to compute a vector that describes each curve
# Then we use a multivariate depth to compute the resulting depth
# We combine depths:
# - Functional depth based on the univariate L1 depth
# - L1 depth of the SDF

bfs = border_depth.get_border_functions(ensemble, ensemble)

# Two components for housedorf distance
print(bfs[0][9].max())  # From the perspective of 0, 19 is far
print(bfs[9][0].max())  # From the perspective of 19, 0 is close, negatively affective the depth

contour_id = 9
plot_contour_spaghetti(ensemble, arr=bod, highlight=[contour_id], is_arr_categorical=False, linewidth=2, alpha=1)
plt.show()
bf_arr = np.concatenate([bfs[contour_id][i].reshape(1, -1) for i in range(num_members)], axis=0)
l1_depths = 1 - np.abs(np.sign(bf_arr).mean(axis=0))
plt.plot(l1_depths)
plt.show()
for i in range(bf_arr.shape[0]):
    plt.plot(bf_arr[i])
plt.show()
print(bf_arr.shape)

l1_depths_left = []
l1_depths_right = []
for i in range(num_members):
    # left side (ego view, parametrized on i's boundary
    # we deal with this using functional depths
    ego_arr = np.concatenate([bfs[i][j].reshape(1, -1) for j in range(num_members)], axis=0)

    # right side (outer view, parametrized on js' boundaries)
    # we deal with this using multivariate depths
    pos_cis = (np.sign(ego_arr) > 0).astype(int).mean(axis=1)
    pos_cjs = []
    neg_cis = (np.sign(ego_arr) < 0).astype(int).mean(axis=1)
    neg_cjs = []
    for j in range(num_members):
        pos_cjs.append((np.sign(bfs[j][i]) > 0).astype(int).mean())
        neg_cjs.append((np.sign(bfs[j][i]) < 0).astype(int).mean())
    pos_cjs = np.array(pos_cjs)
    neg_cjs = np.array(neg_cjs)

    pos_cis = (pos_cis > 0).astype(int)  # strict topological version
    pos_cjs = (pos_cjs > 0).astype(int)
    neg_cis = (neg_cis > 0).astype(int)
    neg_cjs = (neg_cjs > 0).astype(int)

    # This quantifies the topological relationship between c_i and c_j "a la" hausdorff
    cj_vals = []
    for pci, nci, pcj, ncj in zip(pos_cis, neg_cis, pos_cjs, neg_cjs):
        min_p = min(pci, pcj)
        min_n = min(nci, ncj)
        relation_ij = max(min_n, min_p)

        # if relation_ij > 0.5:
        #     relation_ij = 1
        # else:
        #     relation_ij = 0

        # we get 0 if either of the contours contains the other
        # we get 1 if they are not nested
        # we get values in between for cases in between
        # because we want to capture outlyingness, we do 1 - relation
        cj_vals.append(1 - relation_ij)  # consistency of contour relationship

    signed_arr = np.sign(ego_arr)
    weights = np.array(cj_vals).reshape(-1, 1)
    if weights.sum() == 1:  # disjoint component
        l1_depths_left.append(0)
    else:
        weights[i] = 0
        weighted_arr = weights * signed_arr
        weighted_sum = weighted_arr.sum(axis=0)
        weighted_mean = weighted_sum / weights.sum()
        l1d = 1 - np.abs(weighted_mean)
        l1_depths_left.append(l1d.min())

print(bod)
print(bad)
print(l1_depths_left)

fig, axs = plt.subplots(ncols=3, layout="tight", figsize=(6, 3))
plot_contour_spaghetti(ensemble[::-1], arr=bod[::-1], is_arr_categorical=False, linewidth=2, ax=axs[0], alpha=1)
plot_contour_spaghetti(ensemble[::-1], arr=l1_depths_left[::-1], is_arr_categorical=False, linewidth=2, ax=axs[1],
                       alpha=1)
plot_contour_spaghetti(ensemble[::-1], arr=bad[::-1], is_arr_categorical=False, linewidth=2, ax=axs[2], alpha=1)
plt.show()

plt.scatter(bad[num_outliers:], l1_depths_left[num_outliers:], c="blue")
plt.scatter(bad[:num_outliers], l1_depths_left[:num_outliers], c="red")
plt.show()

# df = pd.DataFrame([bad, bod, l1_depths, dists_to_cloud]).T
# df.columns = ["bad", "bod", "l1", "dcloud"]
# df = (df-df.min())/(df.max()-df.min())
# df["name"] = ["normal" if i < num_members - num_outliers else "outlier" for i in range(num_members)]
#
# # df = df.stack().reset_index()
# # df = df.iloc[:, [1, 2]]
# # df.columns = ["dim", "val"]
#
# print(df)
#
# parallel_coordinates(df, class_column="name", color=["blue", "red"])
# plt.show()
