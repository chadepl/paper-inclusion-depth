"""
Here we provide a method to cluster using depths.
The method uses as initialization kmedoids clustering.
Then uses the cluster silhouettes and depths to refine.
This file also provides tools to compute within and between
depths and silhouettes.
"""

import numpy as np
from numpy.linalg import norm
import pandas as pd
from sklearn_extra.cluster import KMedoids
from backend.src.utils import get_distance_transform
from backend.src.contour_depths import band_depth, lp_depth, sdf_depth


def cluster(ensemble_members, depth_type="band"):
    """
    The algorithm uses as initialization the result of the kmedoids
    clustering algorithm, using the matrix that the depth_method returns.
    """

    depths, depth_mat = lp_depth.compute_depths(ensemble_members,
                                                return_data_mat=True)  # seems like using sdfs is much better starting point

    kmedoids = KMedoids(n_clusters=3, random_state=0).fit(depth_mat)
    # initial_clustering = kmedoids.medoid_indices
    initial_clustering = kmedoids.labels_

    clustering = ddclust(ensemble_members, depth_mat, initial_clustering, 1, 0.5)

    return initial_clustering, clustering, depth_mat


def ddclust(ensemble_members,
            data_matrix,
            initial_clustering,
            beta_init, lamb, cost_threshold=5, num_stale_it=3):
    beta = beta_init
    initial_clustering = np.array(initial_clustering)
    partition_labels = np.unique(initial_clustering)

    mean_sils = []
    mean_reds = []
    costs = []

    def get_mvm_idx(ensemble_members, partition_labels, partition):
        #  find multivariate medians
        mvm_idx = []
        for k in partition_labels:
            partition_idx = np.where(partition == k)[0]
            partition_members = [ensemble_members[i] for i in partition_idx]
            partition_depths = lp_depth.compute_depths(partition_members)
            partition_depths = np.array(partition_depths)
            mvm = np.argmax(partition_depths)
            mvm_id = int(partition_idx[mvm])
            mvm_idx.append(mvm_id)
        return mvm_idx

    current_partition = initial_clustering.copy()
    current_mvm_idx = get_mvm_idx(ensemble_members, partition_labels, current_partition)
    current_cfi = get_cost_function_info(ensemble_members, data_matrix, current_partition, current_mvm_idx)
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
            new_cfi = get_cost_function_info(data_matrix, new_partition, new_mvm_idx)
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


def get_silhouettes(data_matrix, clustering, medoids_idx):
    # data matrix is the matrix we are using for clustering
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
        sils.append([i, (b[-1][1] - a[-1][1]) / np.max([b[-1][1], a[-1][1]])])

    return a, b, sils


def get_relative_depths(data_cloud, data_matrix, clustering, medoids_idx):
    within_depths = []
    between_depths = []
    reds = []

    for i in range(len(data_cloud)):  # relative depth per x

        i_cluster = clustering[i]
        cluster_idx = np.where(clustering == i_cluster)[0]
        cluster_cloud = [data_cloud[mid] for mid in cluster_idx]
        i_id_in_cluster = np.where(cluster_idx == i)[0][0]
        within_depths.append(lp_depth.compute_depths(cluster_cloud, points_idx=[i_id_in_cluster, ])[0])

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
        cluster_cloud = [data_cloud[mid] for mid in competing_medoid_cluster_idx]
        cluster_cloud += [data_cloud[i], ]
        i_id_in_cluster = len(cluster_cloud) - 1
        between_depths.append(lp_depth.compute_depths(cluster_cloud, points_idx=[i_id_in_cluster, ])[0])

        reds.append([i, within_depths[-1] - between_depths[-1]])

    return within_depths, between_depths, reds


def get_cost_function_info(data_cloud, data_matrix, clustering, medoids_idx):
    within_depths, between_depths, reds = get_relative_depths(data_cloud, data_matrix, clustering, medoids_idx)
    a, b, sils = get_silhouettes(data_matrix, clustering, medoids_idx)

    within_depths = np.array(within_depths);
    between_depths = np.array(between_depths);
    reds = np.array(reds)
    a = np.array(a)
    b = np.array(b)
    sils = np.array(sils)

    df = pd.DataFrame([np.arange(len(data_cloud)), within_depths, between_depths, reds, a[:, 1], b[:, 1], sils[:, 1]]).T
    df.columns = ["x", "Dw", "Db", "red", "a", "b", "sil"]

    return df


# cf_df = get_cost_function_info(sdfs_arr, kmedoids_labels, kmedoids_medoids)


if __name__ == "__main__":
    from skimage.draw import ellipse
    from skimage.measure import find_contours
    import matplotlib.pyplot as plt

    num_members = 30
    nrows = 300
    ncols = 300

    rr_cc = []
    ensemble_members = []
    memberships = []
    r_radius = c_radius = 50

    for m in range(num_members // 3):
        r = nrows // 2 + (50 * np.sin(np.deg2rad(45)))
        c = (ncols // 2) - (50 * np.cos(np.deg2rad(45)))
        r += np.random.normal(0, 1, 1)[0] * 5
        c += np.random.normal(0, 1, 1)[0] * 5
        rr, cc = ellipse(r, c, r_radius, c_radius, shape=(nrows, ncols))
        rr_cc.append((rr, cc))
        memberships.append(0)

    for m in range(num_members // 3):
        r = nrows // 2 + (50 * np.sin(np.deg2rad(45)))
        c = (ncols // 2) + (50 * np.cos(np.deg2rad(45)))
        r += np.random.normal(0, 1, 1)[0] * 5
        c += np.random.normal(0, 1, 1)[0] * 5
        rr, cc = ellipse(r, c, r_radius, c_radius, shape=(nrows, ncols))
        rr_cc.append((rr, cc))
        memberships.append(1)

    for m in range(num_members - (2 * num_members // 3)):
        r = nrows // 2 - (50 * np.sin(np.deg2rad(45)))
        c = (ncols // 2)
        r += np.random.normal(0, 1, 1)[0] * 5
        c += np.random.normal(0, 1, 1)[0] * 5
        rr, cc = ellipse(r, c, r_radius, c_radius, shape=(nrows, ncols))
        rr_cc.append((rr, cc))
        memberships.append(2)

    for rr, cc in rr_cc:
        bm = np.zeros((nrows, ncols))
        bm[rr, cc] = 1
        ensemble_members.append(bm)

    arg_shuffle = np.arange(len(ensemble_members))
    np.random.shuffle(arg_shuffle)
    ensemble_members = [ensemble_members[i] for i in arg_shuffle]
    memberships = [memberships[i] for i in arg_shuffle]

    colors = ["red", "blue", "lime"]
    plt.imshow(np.zeros_like(ensemble_members[0]), cmap="gray")
    for i, member in enumerate(ensemble_members):
        for contour in find_contours(member, 0.5):
            plt.plot(contour[:, 1], contour[:, 0], c=colors[memberships[i]])
    plt.show()

    new_memberships, depth_mat = cluster(ensemble_members)

    colors = ["red", "blue", "lime"]
    plt.imshow(np.zeros_like(ensemble_members[0]), cmap="gray")
    for i, member in enumerate(ensemble_members):
        for contour in find_contours(member, 0.5):
            plt.plot(contour[:, 1], contour[:, 0], c=colors[new_memberships[i]])
    plt.show()
