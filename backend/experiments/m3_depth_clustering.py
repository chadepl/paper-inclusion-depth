"""
In this script we explore whether the
depth matrix can be used for clustering.
"""

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from backend.src.datasets.ellipse_generation import load_ensemble_ellipses
    from backend.src.contour_band_depth import get_subsets, get_contour_band_depths, get_depth_matrix
    from backend.src.vis_utils import plot_contour_spaghetti, plot_grid_masks, plot_band_checking_procedure, plot_contour_boxplot

    #############
    # Load data #
    #############

    num_members = 30
    num_cols = num_rows = 300
    ensemble_data = load_ensemble_ellipses(num_members, num_cols, num_rows, 0)
    #ensemble_data = load_ensemble_ellipses(num_members, num_cols, num_rows, 1)
    #ensemble_data = load_ensemble_streamlines(num_members, num_cols, num_rows, 0)
    members_masks = [e["data"] for e in ensemble_data["ensemble"]["members"]]
    members_feat = [e["features"] for e in ensemble_data["ensemble"]["members"]]
    members_memberships = {i: v["metadata"]["memberships"] for i, v in enumerate(members_feat)}

    # Data overview

    # - grid
    plot_grid_masks(members_masks)
    plt.show()

    # - spaghetti plot
    plot_contour_spaghetti(members_masks, [v for v in members_memberships.values()], None, ax=None)
    plt.show()

    ##########################
    # Calculate depth matrix #
    ##########################

    # Get bands indices
    subsets = get_subsets(num_members, 2)

    # Get depth data
    depth_data = get_contour_band_depths(members_masks, subsets)

    # - plot bands sample
    subset_band_sample = [depth_data["subset_data"][i]["band_components"]["band"] for i in np.random.randint(0, len(depth_data["subset_data"]), 16)]
    plot_grid_masks(subset_band_sample)
    plt.show()

    # - inspecting particular cases (by combination of member_id + subset_id)
    plot_band_checking_procedure(members_masks, depth_data["subset_data"], member_id=0, subset_id=100)
    plt.show()

    # Get raw depth matrix
    raw_dm = get_depth_matrix(depth_data, raw_quantity="max_lc_rc")

    fig, ax = plt.subplots(layout="tight")
    sns.heatmap(raw_dm, ax=ax)
    ax.set_title(f"Raw depths matrix")
    ax.set_xlabel("Bands")
    ax.set_ylabel("Members")
    plt.show()

    # Get thresholded depth matrix (and analyze threshold)

    thresholds = np.linspace(0, raw_dm.max(), 100)
    th_depths_list = []
    for th in thresholds:
        th_dm = get_depth_matrix(depth_data, threshold=float(th))
        #d = compute_band_depths(depth_data["depth_data"], subset_data, None, th)
        #member_depths.append(d)
        th_depths_list.append(th_dm.mean(axis=1))
    th_depths = np.array(th_depths_list)

    fig, ax = plt.subplots(layout="tight")
    sns.heatmap(th_depths, ax=ax)
    ax.set_title(f"Depths per threshold")
    ax.set_xlabel("Members")
    ax.set_ylabel("Thresholds")
    ax.set_yticks([i for i, _ in enumerate(range(thresholds.size)) if i % 8 == 0], [f"{v:1f}" for i, v in enumerate(thresholds.tolist()) if i % 8 ==0])
    plt.show()

    th_dm = get_depth_matrix(depth_data, threshold=0.04)

    fig, ax = plt.subplots(layout="tight")
    sns.heatmap(th_dm, ax=ax)
    ax.set_title(f"Thresholded depths matrix (0.04)")
    ax.set_xlabel("Bands")
    ax.set_ylabel("Members")
    plt.show()

    ###################
    # Depths analysis #
    ###################

    raw_depths = (1-raw_dm.mean(axis=1)).reshape((1, -1))
    th_depths = th_dm.mean(axis=1).reshape((1, -1))
    depths_corr = np.corrcoef(raw_depths.flatten(), th_depths.flatten())

    fig, axs = plt.subplots(nrows=2, layout="tight")
    sns.heatmap(raw_depths, ax=axs[0])
    axs[0].set_title(f"Raw depths")
    sns.heatmap(th_depths, ax=axs[1])
    axs[1].set_title(f"Thresholded depths")
    axs[1].set_xlabel("Members")
    fig.suptitle(f"Raw vs th depths (corr: {depths_corr[0,1]})")
    plt.show()

    fig, ax = plt.subplots()
    plot_contour_boxplot(members_masks, th_dm, epsilon_out=0.01, ax=ax)
    ax.set_title("Contour boxplot unclustered data (th_dm)")
    plt.show()


    ################
    # Cluster data #
    ################

    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

    # Based on depths

    X_depths = raw_dm.copy()

    Z_depths = linkage(X_depths, "ward")
    dendrogram(Z_depths)
    plt.title("Dendrogram (depths)")
    plt.show()

    d_depths = [8, 2][1]  # [three circles, ellipses]
    clustering_depths = fcluster(Z_depths, d_depths, criterion="distance")

    fig, ax = plt.subplots()
    plot_contour_spaghetti(members_masks, clustering_depths.tolist(), ax=ax)
    ax.set_title(f"Clustering result (depths; {d_depths})")
    plt.show()

    # Based on distance matrix

    from backend.src.utils import get_distance_transform

    X_dists = []
    for mm in members_masks:
        dt = get_distance_transform(mm)
        X_dists.append(dt.flatten())
    X_dists = np.array(X_dists)

    Z_dists = linkage(X_dists, "ward")
    dendrogram(Z_dists)
    plt.title("Dendrogram (sdfs)")
    plt.show()

    d_dists = [2300, 2000][1]  # [three circles, ellipses]
    clustering_dists = fcluster(Z_dists, d_dists, criterion="distance")

    fig, ax = plt.subplots()
    plot_contour_spaghetti(members_masks, clustering_dists.tolist(), ax=ax)
    ax.set_title(f"Clustering result (sdf; {d_dists})")
    plt.show()

    plt.imshow(X_dists[20, :].reshape(300, -1))
    plt.show()


    ####################
    # Cluster analysis #
    ####################
    # Relative data depth to pick a cluster

    def compute_view_depths_matrix(depth_matrix, subset_data, selection_idx=None):
        """
        Returns the depth matrix filtered given a selection of members.
        """
        # filter members
        if selection_idx is None:
            selection_idx = np.range(depth_matrix.shape[0]).tolist()

        sd = dict()
        for subset_id, subset_val in subset_data.items():
            include_subset = True
            if len(selection_idx) != 0:
                for member_id in subset_val["idx"]:
                    if member_id not in selection_idx:
                        include_subset = False
                        break
            if include_subset:
                sd[subset_id] = subset_val

        subset_idx = [v["subset_id"] for k, v in sd.items()]

        i1 = [[i, ] for i in selection_idx]
        i2 = subset_idx
        filtered_dm = depth_matrix[i1, i2]

        return selection_idx, subset_idx, filtered_dm

    def linkage_validation(Z, depth_matrix, subset_data, num_steps=1000, metric="within_depths"):
        """
        Given a linkage matrix Z, this method computes a given
        clustering validation metric every num_steps.
        """
        per_level_metrics = []
        min_th = Z[:, 2].min()  # min cut off height
        max_th = Z[:, 2].max()  # max cut off height
        for dist in np.linspace(min_th, max_th, num_steps):
            # obtain clusters
            clustering = fcluster(Z, dist, criterion="distance")
            # compute per-cluster metrics
            level_metrics = []
            for i in np.unique(clustering):
                member_idx = np.where(clustering == i)[0].tolist()
                if metric == "within_depths":
                    if len(member_idx) <= 2:
                        # set too small for depths to make sense so we assign 0
                        # this also incentivizes clusters larger than size 2 because
                        # we want to maximize the average within depth
                        level_metrics.append((i, 0))
                    else:
                        _, _, dmv = compute_view_depths_matrix(depth_matrix, subset_data, member_idx)
                        ds = dmv.mean(axis=1)
                        level_metrics.append((i, ds.mean()))  # average within cluster depths
            per_level_metrics.append((dist, level_metrics))

        return per_level_metrics

    # grid of boxplots

    dm_views = []
    for i in np.unique(clustering_depths):
        member_idx = np.where(clustering_depths == i)[0].tolist()
        _, _, dm_view = compute_view_depths_matrix((1 - raw_dm), depth_data["subset_data"], member_idx)
        plot_contour_boxplot([members_masks[i] for i in member_idx], dm_view, epsilon_out=0.5)
        plt.title(f"Cluster {i}")
        plt.show()
        dm_views.append(dm_view)

    # Depth-based clustering

    depth_level_metrics = linkage_validation(Z_depths, th_dm, depth_data["subset_data"])
    dlm_x = np.array([lv[0] for lv in depth_level_metrics])
    dlm_y = [lv[1] for lv in depth_level_metrics]
    dlm_y = np.array([np.array([e[1] for e in lv]).mean() for lv in dlm_y])

    plt.plot(dlm_x, dlm_y)
    plt.xlabel("Distance dendrogram cut-off")
    plt.ylabel("Average within cluster depth")
    plt.title("Average within cluster depth for \n different levels of the dendrogram (depths)")
    plt.vlines([2.5], 0, dlm_y.max())
    plt.show()

    # sdf-based clustering

    sdf_level_metrics = linkage_validation(Z_dists, th_dm, depth_data["subset_data"])
    slm_x = np.array([lv[0] for lv in sdf_level_metrics])
    slm_y = [lv[1] for lv in sdf_level_metrics]
    slm_y = np.array([np.array([e[1] for e in lv]).mean() for lv in slm_y])

    plt.plot(slm_x, slm_y)
    plt.xlabel("Distance dendrogram cut-off")
    plt.ylabel("Average within cluster depth")
    plt.title("Average within cluster depth for \n different levels of the dendrogram (sdfs)")
    plt.vlines([2300], 0, slm_y.max())
    plt.show()

    # clustering based on threshold

    #clustering_labels = fcluster(Z_depths, 2.5, criterion="distance")
    clustering_labels = fcluster(Z_dists, 2300, criterion="distance")
    plot_contour_spaghetti(members_masks, [int(l) for l in clustering_labels.tolist()])
    plt.show()

    ##########################
    # Depth based Clustering #
    ##########################

    # - we now explore how to perform clustering using depth data
    # - to keep in mind: we want to maximize the average depth
    # - to test this, we perform aglomerative clustering on the depth_data
    # - for each cutoff point we calculate the average depth across clusters
    # - we pick the K that maximizes this metric.
    # - TODO: extend this criterion to the ReD + sil
    # - question: why agglomerative instead of k-medians
    # - can we exploit the algorithmic structure of agglomerative to speedup the process?



    # - now we see how the depths change with different distance values



