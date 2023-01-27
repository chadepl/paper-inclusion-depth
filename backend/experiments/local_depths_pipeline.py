
from time import time
import numpy as np
import pandas as pd
from numpy.linalg import norm
from skimage.draw import ellipse, rectangle
from skimage.measure import find_contours
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

from backend.src.vis_utils import plot_contour_spaghetti, plot_contour_boxplot, plot_grid_masks, plot_contour
from backend.src.utils import get_distance_transform
from backend.src.streamline_generation import load_ensemble_streamlines

np.random.seed(42)

# Load dataset
#  Mixture of three shapes in user defined proportions (circle, ellipse, circle with blob)
#  We define two types of outliers:
#  - Cluster outlier (close enough to fall in a cluster but low centrality)
#  - Dataset outlier (different shape in very small proportion)

def get_multimodal_dataset(num_cols,
                           num_rows,
                           num_members,
                           fraction_shapes=(1 / 3, 1 / 3, 1 / 3),
                           fraction_clust_out=(0.01, 0.005, 0.02),
                           fraction_glob_out=0.01):
    num_global_outliers = np.ceil(num_members * fraction_glob_out)
    num_available_members = num_members - num_global_outliers

    num_members_shape_a = np.floor(num_members * fraction_shapes[0])
    num_outliers_shape_a = int(np.ceil(num_members_shape_a * fraction_clust_out[0]))
    num_available_members -= num_members_shape_a
    num_members_shape_a = int(num_members_shape_a - num_outliers_shape_a)

    num_members_shape_b = np.floor(num_members * fraction_shapes[1])
    num_outliers_shape_b = int(np.ceil(num_members_shape_b * fraction_clust_out[1]))
    num_available_members -= num_members_shape_b
    num_members_shape_b = int(num_members_shape_b - num_outliers_shape_b)

    num_members_shape_c = num_available_members
    num_outliers_shape_c = int(np.ceil(num_members_shape_c * fraction_clust_out[2]))
    num_available_members -= num_members_shape_c
    num_members_shape_c = int(num_members_shape_c - num_outliers_shape_c)

    num_global_outliers += num_available_members
    num_global_outliers = int(num_global_outliers)

    num_members_calculated = (num_members_shape_a + num_members_shape_b + num_members_shape_c +
                           num_outliers_shape_a + num_outliers_shape_b + num_outliers_shape_c +
                           num_global_outliers)

    print(f" Shape a (members/outliers): {num_members_shape_a}/{num_outliers_shape_a} \n"
          f" Shape b (members/outliers): {num_members_shape_b}/{num_outliers_shape_b} \n"
          f" Shape c (members/outliers): {num_members_shape_c}/{num_outliers_shape_c} \n"
          f" Global outliers: {num_global_outliers} \n"
          f" Num members (input/calculated): {num_members}/{num_members_calculated}")

    assert num_members == num_members_calculated

    center = np.array((num_cols // 2, num_rows // 2))
    max_radii = center.min() * 0.5
    shapes_rr_cc = []
    memberships = []
    outlier = []

    # circle
    iteration_range = [0 for _ in range(num_members_shape_a)] + [1 for _ in range(num_outliers_shape_a)]
    for outlier_status in iteration_range:
        c = center.copy()
        c += (np.random.randn(2) * 5).astype(int)  # add some jitter
        r = max_radii.copy() * 0.9
        r += int(np.random.randn(1)[0] * 5)  # add some jitter
        if outlier_status == 1:
            r *= 0.7
        shapes_rr_cc.append(ellipse(c[0], c[1], r, r, shape=(num_rows, num_cols)))
        memberships.append(0)
        outlier.append(outlier_status)

    # ellipse
    iteration_range = [0 for _ in range(num_members_shape_b)] + [1 for _ in range(num_outliers_shape_b)]
    for outlier_status in iteration_range:
        c = center.copy()
        c += (np.random.randn(2) * 5).astype(int)  # add some jitter
        r_jitter_factor = (np.random.random(1) / 5) + 0.8
        r = np.array([max_radii, int(max_radii * r_jitter_factor)])
        if outlier_status == 1:
            r[1] *= 0.5
        shapes_rr_cc.append(ellipse(c[0], c[1], r[0], r[1], shape=(num_rows, num_cols)))
        memberships.append(1)
        outlier.append(outlier_status)

    # circle with blob
    iteration_range = [0 for _ in range(num_members_shape_c)] + [1 for _ in range(num_outliers_shape_c)]
    for outlier_status in iteration_range:
        c = center.copy()
        c += (np.random.randn(2) * 5).astype(int)  # add some jitter
        r = max_radii.copy()
        r += int(np.random.randn(1)[0] * 5)  # add some jitter
        angle = int(45 * ((np.random.random(1) / 10) + 0.95))  # add some jitter
        if outlier_status == 1:
            angle = int(360 * np.random.random(1))
        mc_0 = c[0] + r * np.sin(np.deg2rad(angle))
        mc_1 = c[1] + r * np.cos(np.deg2rad(angle))
        mcr = int(r * ((np.random.random(1) / 10) + 0.4))  # add some jitter
        rr_cc_bc = ellipse(c[0], c[1], r, r, shape=(num_rows, num_cols))
        rr_cc_mc = ellipse(mc_0, mc_1, mcr, mcr, shape=(num_rows, num_cols))
        rr_cc = []
        rr_cc.append(np.concatenate([rr_cc_bc[0], rr_cc_mc[0]]).flatten())
        rr_cc.append(np.concatenate([rr_cc_bc[1], rr_cc_mc[1]]).flatten())
        shapes_rr_cc.append(rr_cc)
        memberships.append(2)
        outlier.append(outlier_status)

    iteration_range = [1 for _ in range(num_global_outliers)]
    for outlier_status in iteration_range:
        c = center.copy()
        r = max_radii.copy()
        rr_cc = rectangle(c - r, c + r, shape=(num_rows, num_cols))
        shapes_rr_cc.append([rr_cc[0].astype(int), rr_cc[1].astype(int)])
        memberships.append(3)
        outlier.append(outlier_status)

    shape_masks = [np.zeros((num_rows, num_cols)) for _ in range(len(shapes_rr_cc))]
    for i, (rr, cc) in enumerate(shapes_rr_cc):
        shape_masks[i][rr, cc] = 1

    return shape_masks, memberships, outlier


num_members = 100
num_cols = num_rows = 300
masks, memberships_labels, outlier_labels = get_multimodal_dataset(num_cols, num_rows, num_members,
                                                                   fraction_shapes=(1/4, 1/4, 1/2),
                                                                   fraction_clust_out=(0.05, 0.05, 0.03))
out = load_ensemble_streamlines(num_members, num_cols, num_rows, params_set=0)
masks = [m["data"] for m in out["ensemble"]["members"]]
memberships_labels = [int(m["features"]["metadata"]["memberships"]) for m in out["ensemble"]["members"]]
outlier_labels = [1 for i in range(num_members)]

masks_arr = np.array([m.flatten() for m in masks])
masks_mean = masks_arr.mean(axis=0)
masks_std = masks_arr.std(axis=0)

plt.imshow(masks_std.reshape(300, -1))
plt.show()

fig, ax = plt.subplots(layout="tight")
plot_grid_masks(masks, ax=ax)
plt.show()

fig, ax = plt.subplots(layout="tight")
ax.imshow(np.zeros_like(masks[0]), alpha=0)
plot_contour_spaghetti(masks, memberships=memberships_labels, ax=ax)
#fig.savefig("/Users/chadepl/Downloads/temp.png")
plt.show()

masks_filter = np.where(np.array(memberships_labels) == 3)[0]
filtered_memberships = [memberships_labels[i] for i in masks_filter]
filtered_outlier_status = [outlier_labels[i] for i in masks_filter]
outlier_filter = np.where(np.array(filtered_outlier_status) == 1)[0]
fig, ax = plt.subplots(layout="tight")
ax.imshow(np.zeros_like(masks[0]), alpha=0)
plot_contour_spaghetti([masks[i] for i in masks_filter], memberships=filtered_memberships, highlight=outlier_filter, ax=ax)
plt.show()

# Compute per-member sdf (2 \times O(distance_transform_edt))
#  Exact euclidean feature transform, as described in: C. R. Maurer,
#   Jr., R. Qi, V. Raghavan, "A linear time algorithm for computing
#   exact euclidean distance transforms of binary images in arbitrary
#   dimensions. IEEE Trans." PAMI 25, 265-270, 2003


masks_sdfs = [get_distance_transform(m, tf_type="signed") for m in masks]
sdfs_arr = np.array([sdf.flatten() for sdf in masks_sdfs])

fig, ax = plt.subplots(layout="tight")
plot_grid_masks(masks_sdfs, ax=ax, cmap="viridis")
plt.show()

divnorm=colors.TwoSlopeNorm(vmin=masks_sdfs[97].min(), vcenter=0., vmax=masks_sdfs[97].max())
fig, ax = plt.subplots(layout="tight")
ax.imshow(masks_sdfs[97].reshape(300, -1), cmap="viridis")
plot_contour(find_contours(masks[97]), ax=ax, line_kwargs=dict(c="orange"))
ax.set_axis_off()
plt.show()


# Compute global per-member L1 depths

def compute_l_depth(data_matrix, members_idx=None, partition_idx=None):

    num_members = data_matrix.shape[0]
    if members_idx is None:
        members_idx = np.arange(num_members)
    else:
        members_idx = np.array(members_idx)

    if partition_idx is None:
        partition_idx = np.arange(num_members)
    else:
        partition_idx = np.array(partition_idx)

    t_start = time()

    l_depths = []
    for i in members_idx:
        vecsum = np.zeros_like(data_matrix[i, :])
        denom = 0
        for j in partition_idx:
            if i != j:
                diff = data_matrix[j, :] - data_matrix[i, :]
                uvec = diff/(norm(diff, ord=2) + np.finfo(np.float).eps)
                vecsum += uvec
                denom += 1
        vecsum = vecsum / denom
        l_depths.append((i, 1 - np.max([0, norm(vecsum, ord=2) - (1/data_matrix.shape[0])])))

    t_end = time()

    print(f"SDF-based l depths took {t_end - t_start} seconds to compute")

    # depths = [e[1] for e in sdf_l1_depths]
    # depths = np.array(depths_sdf)
    depths = np.array(l_depths)

    return depths

depths = compute_l_depth(sdfs_arr)

fig, ax = plt.subplots(layout="tight")
ax.imshow(np.zeros_like(masks[0]), alpha=0)
plot_contour_boxplot(masks, depths[:,1], epsilon_out=0.1, ax=ax)
plt.show()

fig, ax = plt.subplots(layout="tight")
ax.imshow(np.zeros_like(masks[0]), alpha=0)
plot_contour_spaghetti(masks, memberships=depths[:,1], ax=ax)#, highlight=[22,70])
plt.show()

depth_sort_idx = np.argsort(depths[:, 1])
plt.scatter(np.arange(num_members), depths[depth_sort_idx, 1], c=np.array(outlier_labels)[depth_sort_idx])
plt.xlabel("Ensemble member")
plt.ylabel("L1 depth")
plt.show()

# Compute local per-member L1 depths for a given beta



# Cluster data
#  We use the method proposed at the paper "Clustering and classification based on the L1 data depth"
#  First we initialize using PAM (k-medoids) from scikit-learn-extra https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html

kmedoids = KMedoids(n_clusters=2, random_state=0).fit(sdfs_arr)
kmedoids_medoids = kmedoids.medoid_indices_
kmedoids_labels = kmedoids.labels_
kmedoids_labels = [2 if kml == 0 else (1 if kml == 2 else 0) for kml in kmedoids.labels_]

fig, ax = plt.subplots(layout="tight")
ax.imshow(np.zeros_like(masks[0]), alpha=0)
plot_contour_spaghetti(masks, memberships=kmedoids_labels, ax=ax)#, highlight=[22,70])
plt.show()

masks_filter = np.where(np.array(kmedoids_labels) == 2)[0]
filtered_memberships = [memberships_labels[i] for i in masks_filter]
filtered_outlier_status = [outlier_labels[i] for i in masks_filter]
outlier_filter = np.where(np.array(filtered_outlier_status) == 1)[0]
fig, ax = plt.subplots(layout="tight")
ax.imshow(np.zeros_like(masks[0]), alpha=0)
plot_contour_spaghetti([masks[i] for i in masks_filter], memberships=filtered_memberships, ax=ax)
plt.show()



# Evaluate cluster consistency using within/between depth


def get_silhouettes(data_matrix, clustering, medoids_idx):
    # average distance of observation xi to other observations in the cluster
    # sil_i = (b_i - a_i) / max(a_i, b_i)
    # b_i = min d(x_i, l) with l != k
    # a_i = average d(x_i, I(k))
    clusters_idx = np.unique(clustering)
    a = []
    b = []
    sils = []
    for i in range(data_matrix.shape[0]):  # silhouette per x
        i_cluster = clustering[i]
        cluster_idx = np.where(clustering == i_cluster)[0]

        within_dists = []
        for j in cluster_idx:
            if i != j:
                within_dists.append(norm(data_matrix[i] - data_matrix[j], ord=2))
        within_dists = np.array(within_dists)

        between_dists = []
        for k in clusters_idx:
            if k != i_cluster:
                k_idx = np.where(clustering == k)[0]
                k_dists = []
                for j in k_idx:
                    k_dists.append(norm(data_matrix[i] - data_matrix[j], ord=2))
                k_dists = np.array(k_dists)
                between_dists.append(k_dists.mean())
        between_dists = np.array(between_dists)

        a.append([i, within_dists.mean()])
        b.append([i, between_dists.min()])
        sils.append([i, (b[-1][1] - a[-1][1])/np.max([b[-1][1], a[-1][1]])])

    return a, b, sils


def get_relative_depths(data_matrix, clustering, medoids_idx):
    within_depths = []
    between_depths = []
    reds = []
    for i in range(data_matrix.shape[0]):  # relative depth per x
        i_cluster = clustering[i]
        cluster_idx = np.where(clustering == i_cluster)[0]
        within_depths.append(compute_l_depth(data_matrix, (i, ), cluster_idx).tolist()[0])

        dists_to_medoids = []
        for mid in medoids_idx:
            dists_to_medoids.append(norm(data_matrix[i] - data_matrix[mid], ord=2))
        dists_to_medoids = np.array(dists_to_medoids)
        arg_sort_medoids = np.argsort(dists_to_medoids)
        for mid in arg_sort_medoids:
            if mid != i_cluster:
                competing_medoid = mid
                break

        competing_medoid_cluster_idx = np.where(clustering == competing_medoid)[0]
        between_depths.append([competing_medoid, ] + compute_l_depth(data_matrix, (i, ), competing_medoid_cluster_idx).tolist()[0])
        reds.append([i, within_depths[-1][1] - between_depths[-1][2]])

    return within_depths, between_depths, reds


def get_cost_function_info(data_matrix, clustering, medoids_idx):
    within_depths, between_depths, reds = get_relative_depths(data_matrix, clustering, medoids_idx)
    a, b, sils = get_silhouettes(data_matrix, clustering, medoids_idx)

    within_depths = np.array(within_depths); between_depths = np.array(between_depths); reds = np.array(reds)
    a = np.array(a); b = np.array(b); sils = np.array(sils)

    df = pd.DataFrame([within_depths[:, 0], kmedoids_labels, within_depths[:, 1], between_depths[:, 2], between_depths[:, 0], reds[:, 1],
                            a[:, 1], b[:, 1], sils[:, 1]]).T
    df.columns = ["x", "labels", "wd", "bd", "competing_cluster", "red", "a", "b", "sil"]

    return df


cf_df = get_cost_function_info(sdfs_arr, kmedoids_labels, kmedoids_medoids)

plot_df = cf_df.groupby("labels").mean()
plot_df.reset_index(inplace=True)
sns.barplot(data=plot_df, x="labels", y="sil")

# g = sns.FacetGrid(reds_df, row="labels")
# g.map(sns.barplot, "x", "red")
# plt.show()


def ddclust(data_matrix, initial_clustering, beta_init, lamb, cost_threshold=5, num_stale_it=3):
    beta = beta_init
    initial_clustering = np.array(initial_clustering)
    partition_labels = np.unique(initial_clustering)

    mean_sils = []
    mean_reds = []
    costs = []

    def get_mvm_idx(data_matrix, partition_labels, partition):
        #  find multivariate medians
        mvm_idx = []
        for k in partition_labels:
            partition_idx = np.where(partition == k)[0]
            partition_depths = compute_l_depth(data_matrix, partition_idx, partition_idx)
            partition_depths = np.array(partition_depths)
            mvm = np.argmax(partition_depths[:, 1])
            mvm_id = int(partition_depths[mvm, 0])
            mvm_idx.append(mvm_id)
        return mvm_idx

    current_partition = initial_clustering.copy()
    current_mvm_idx = get_mvm_idx(data_matrix, partition_labels, current_partition)
    current_cfi = get_cost_function_info(sdfs_arr, current_partition, current_mvm_idx)
    current_ci = ((1 - lamb) * current_cfi.sil + lamb * current_cfi.red).to_numpy()

    for i in range(10):
        print(f"Iteration {i}")
        outliers_idx = np.argsort(current_ci)[0:cost_threshold]  # TODO: set threshold data based?
        stale_it = num_stale_it
        while True:
            outliers_choice = np.random.choice(np.arange(outliers_idx.size), 1, replace=False)
            subset_outliers_idx = outliers_idx[outliers_choice]
            competing_cluster = current_cfi.competing_cluster[subset_outliers_idx].to_numpy()

            new_partition = current_partition.copy()
            new_partition[subset_outliers_idx] = competing_cluster
            new_mvm_idx = get_mvm_idx(data_matrix, partition_labels, new_partition)
            new_cfi = get_cost_function_info(sdfs_arr, new_partition, new_mvm_idx)
            new_ci = ((1 - lamb) * new_cfi.sil + lamb * new_cfi.red).to_numpy()

            current_c = current_ci.mean()
            new_c = new_ci.mean()

            if new_c > current_c:
                print("Change of memberships!")
                current_partition = new_partition.copy()
                current_mvm_idx = new_mvm_idx.copy()
                current_cfi = new_cfi.copy()
                current_ci = new_ci.copy()
                current_c = new_c
            else:
                stale_it -= 1

            outliers_idx = np.delete(outliers_idx, outliers_choice)

            mean_sils.append(current_cfi.sil.mean())
            mean_reds.append(current_cfi.red.mean())
            costs.append(current_c)

            if outliers_idx.size == 0 or stale_it <= 0:
                break

    return current_partition, (mean_sils, mean_reds, costs)

clustering_labels = kmedoids_labels.copy()
#clustering_labels[79] = 1
refine05, meta05 = ddclust(sdfs_arr, clustering_labels, 0, 0.5, 10)
refine08, meta08 = ddclust(sdfs_arr, clustering_labels, 0, 0.8, 10)

fig, ax = plt.subplots(layout="tight")
ax.imshow(np.zeros_like(masks[0]), alpha=0)
plot_contour_spaghetti(masks, memberships=refine08.tolist(), ax=ax)#, highlight=[22,70])
plt.show()

masks_filter = np.where(np.array(refine08) == 2)[0]
filtered_memberships = [memberships_labels[i] for i in masks_filter]
filtered_outlier_status = [outlier_labels[i] for i in masks_filter]
outlier_filter = np.where(np.array(filtered_outlier_status) == 1)[0]
fig, ax = plt.subplots(layout="tight")
ax.imshow(np.zeros_like(masks[0]), alpha=0)
plot_contour_spaghetti([masks[i] for i in masks_filter], memberships=filtered_memberships, ax=ax)
plt.show()

plt.plot(meta05[1], label="ReD (0.5)")
plt.plot(meta08[1], label="ReD (0.8)")
plt.xlabel("Iteration")
plt.ylabel("ReD")
plt.legend()
plt.show()

# multiple_k_outs = []
# ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# for k in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
#     kmedoids = KMedoids(n_clusters=k, random_state=0).fit(sdfs_arr)
#     kmedoids_medoids = kmedoids.medoid_indices_
#     kmedoids_labels = kmedoids.labels_
#     #kmedoids_labels = [2 if kml == 0 else (1 if kml == 2 else 0) for kml in kmedoids.labels_]
#
#     refined_labels, meta = ddclust(sdfs_arr, kmedoids_labels, 0, 0.8, 5)
#     multiple_k_outs.append([refined_labels, meta[1][-1]])
#
# plt.plot(ks, [mko[1] for mko in multiple_k_outs])
# plt.xlabel("k")
# plt.ylabel("ReD")
# plt.show()

labels = multiple_k_outs[1][0].tolist()
mapping = {0: 2, 2: 1, 1: 0}
labels = [mapping[kml] if kml in mapping else kml for kml in labels]

fig, ax = plt.subplots(layout="tight")
ax.imshow(np.zeros_like(masks[0]), alpha=0)
plot_contour_spaghetti(masks, memberships=labels, ax=ax)#, highlight=[22,70])
plt.show()

masks_filter = np.where(np.array(labels) == 2)[0]
filter_depths = compute_l_depth(sdfs_arr, masks_filter, masks_filter)[:,1]
filtered_memberships = [memberships_labels[i] for i in masks_filter]
filtered_outlier_status = [outlier_labels[i] for i in masks_filter]
outlier_filter = np.where(np.array(filtered_outlier_status) == 1)[0]
fig, ax = plt.subplots(layout="tight")
ax.imshow(np.zeros_like(masks[0]), alpha=0)
plot_contour_spaghetti([masks[i] for i in masks_filter], memberships=filter_depths, ax=ax)
plt.show()

fig, ax = plt.subplots(layout="tight")
ax.imshow(np.zeros_like(masks[0]), alpha=0)
plot_contour_boxplot([masks[i] for i in masks_filter], filter_depths, epsilon_out=0.1, ax=ax)
plt.show()

# Cost function of clusterings with different k

# cs = []
# random_states = [1, 2, 3, 4]
# for k in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
#     for rs in random_states:
#         kmedoids = KMedoids(n_clusters=k, random_state=rs).fit(sdfs_arr)
#         kmedoids_medoids = kmedoids.medoid_indices_
#         kmedoids_labels = kmedoids.labels_
#         df = get_cost_function_info(sdfs_arr, kmedoids_labels, kmedoids_medoids)
#         cs.append([k, rs, df.sil.mean(), df.red.mean()])
# cs = np.array(cs)
# cs_df = pd.DataFrame(cs)
# cs_df.columns = ["k", "rs", "sil", "red"]
#
# sns.lineplot(data=cs_df, x="k", y="red")
# plt.show()

#plt.plot(cs[:,0], label="sil")
# plt.plot(cs[:,1], label="red")
#plt.plot((cs * np.array([[0.1, 0.9]])).sum(axis=1), label="red+sil")
# plt.legend()
# plt.show()

# Animate transition of the trimmed mean with different clusterings


# Depth-based attention vs local DD-plots
# - Here we compute the dd plot of one area of the image vs another
# - Areas with more visual variation should result in curved dd plots
# - We want to then transform this insight into an attention map using a quadtree
# - We could then allow think of editing a shape as the process of removing outliers and weighting parts of the image differently

# depths_a = compute_l_depth(sdfs_arr)
# depths_b = compute_l_depth(sdfs_arr)
#
# depths_a = np.array(depths_a); depths_b = np.array(depths_b);
#
# # ideal dd plot
# plt.scatter(depths_a[:,1], depths_b[:,1])
# plt.show()
#
#
# # now we zoom in the data
# # - lets compare the four quadrants of the image
# from matplotlib.patches import Rectangle
# rect_selections = [
#     [0, 0, 300//2, 300//2],
#     [0, 300//2, 300//2, 300],
#     [300//2, 0, 300, 300//2],
#     [300//2, 300//2, 300, 300],
# ]
#
# # rect_selections = [
# #     [300//2, 300//2, 900//4, 900//4],
# #     [300//2, 900//4, 900//4, 300],
# #     [900//4, 300//2, 300, 900//4],
# #     [900//4, 900//4, 300, 300],
# # ]
# #
# # rect_selections = [
# #     [0, 0, 300//4, 300//4],
# #     [0, 300//4, 300//4, 300//2],
# #     [300//4, 0, 300//2, 300//4],
# #     [300//4, 300//4, 300//2, 300//2],
# # ]
#
# fig, ax = plt.subplots()
# ax.imshow(masks_arr.mean(axis=0).reshape(-1,300))
# for rect_selection in rect_selections:
#     ax.add_patch(Rectangle(rect_selection[0:2],
#                            rect_selection[2]-rect_selection[0],
#                            rect_selection[3]-rect_selection[1],
#                            edgecolor="red",
#                            fill=False))
# plt.show()
#
# selection_depths = []
# for i, rect_selection in enumerate(rect_selections):
#     zoomed_sdf_arr = [m[rect_selection[0]:rect_selection[2], rect_selection[1]:rect_selection[3]] for m in masks_sdfs]
#     zoomed_sdf_arr = np.array([m.flatten() for m in zoomed_sdf_arr])
#     sd = compute_l_depth(zoomed_sdf_arr)
#     sd = np.array(sd);
#     selection_depths.append([i, rect_selection, sd])
#
# fig, axs = plt.subplots(ncols=2, nrows=2)
# axs[0, 0].scatter(depths_a[:,1], selection_depths[0][2][:,1])
# axs[0, 0].set_title(np.argmax(selection_depths[0][2][:,1]))
# axs[0, 1].scatter(depths_a[:,1], selection_depths[1][2][:,1])
# axs[0, 1].set_title(np.argmax(selection_depths[1][2][:,1]))
# axs[1, 0].scatter(depths_a[:,1], selection_depths[2][2][:,1])
# axs[1, 0].set_title(np.argmax(selection_depths[2][2][:,1]))
# axs[1, 1].scatter(depths_a[:,1], selection_depths[3][2][:,1])
# axs[1, 1].set_title(np.argmax(selection_depths[3][2][:,1]))
# plt.suptitle(np.argmax(depths_a[:,1]))
# plt.show()
#
#
# rect_selection = np.array([0, 0, 300, 300])
# rectangles = []
# selection_depths = []
#
# for i in range(15):
#     rs = rect_selection.copy()
#     rs[0:2] += 10 * i
#     rs[2:4] -= 10 * i
#
#     rectangles.append(Rectangle(rs[0:2], rs[2] - rs[0], rs[3] - rs[1], edgecolor="red", fill=False))
#
#     zoomed_sdf_arr = [m[rs[0]:rs[2], rs[1]:rs[3]] for m in masks_sdfs]
#     zoomed_sdf_arr = np.array([m.flatten() for m in zoomed_sdf_arr])
#     sd = compute_l_depth(zoomed_sdf_arr)
#     sd = np.array(sd);
#     selection_depths.append([i, rs, sd])
#
# fig, ax = plt.subplots()
# ax.imshow(masks_arr.mean(axis=0).reshape(-1, 300))
# for rectangle in rectangles:
#     ax.add_patch(rectangle)
# plt.show()
#
# fig, axs = plt.subplots(ncols=15, figsize=(40,5))
# for i, s_depths in enumerate(selection_depths):
#     axs[i].scatter(depths_a[:,1], s_depths[2][:,1])
# plt.show()
#
#

# cluster1_idx = np.where(np.array(memberships_labels) == 0)[0]
# cluster2_idx = np.where(np.array(memberships_labels) == 2)[0]
#
# grid_data = []
# grid_data_clusters = []
#
# grid_size = 2
# grid_step = 300 // grid_size
#
# for i in range(grid_size):
#     for j in range(grid_size):
#         rs = np.array([i*grid_step, j*grid_step, i*grid_step+grid_step, j*grid_step+grid_step])
#         rectangle = Rectangle(rs[0:2], rs[2] - rs[0], rs[3] - rs[1], edgecolor="red", fill=False)
#         zoomed_sdf_arr = [m[rs[0]:rs[2], rs[1]:rs[3]] for m in masks_sdfs]
#         zoomed_sdf_arr = np.array([m.flatten() for m in zoomed_sdf_arr])
#
#         global_depth = compute_l_depth(sdfs_arr)
#         local_depth = compute_l_depth(zoomed_sdf_arr)
#
#         grid_data.append((i, j, global_depth, local_depth, rectangle))
#
#         sd_1 = compute_l_depth(zoomed_sdf_arr, cluster2_idx, cluster1_idx)
#         sd_1 = np.array(sd_1)
#
#         sd_2 = compute_l_depth(zoomed_sdf_arr, cluster2_idx, cluster2_idx)
#         sd_2 = np.array(sd_2)
#
#         grid_data_clusters.append((i, j, sd_1, sd_2, rectangle))
#
#
# fig, ax = plt.subplots()
# ax.imshow(masks_arr.mean(axis=0).reshape(-1, 300))
# for gd in grid_data:
#     ax.add_patch(gd[-1])
# plt.show()
#
# fig, axs = plt.subplots(ncols=grid_size, nrows=grid_size, figsize=(5*grid_size,5*grid_size))
# for gd in grid_data:
#     axs[gd[0], gd[1]].scatter(gd[2][:,1], gd[3][:,1])
#     axs[gd[0], gd[1]].set_title(gd[2][:,1].mean())
# plt.show

