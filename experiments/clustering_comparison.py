"""
Here we compare clustering based on SDF vs depth
Probably we want a combination of both because:
- SDF provide spatial information
- Depths provide order information and are transform invariant

Nevertheless, in their construction, depths do describe the space
in a way that could permit inferring locations without the need of
metric information.
For instance, if a contour falls more times in a cluster of bands A
than in a cluster of bands B. That tells something vs contour that produces
the opposite behavior.
If this is possible to achieve, then the clustering would be more robust in
changes of the space in which the observations are embedded, only caring for
the changes in the arrangement of observations.


"""
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from skimage.measure import find_contours
import matplotlib.pyplot as plt

from backend.src.datasets.circles import circles_with_outliers, circles_multiple_radii_modes
from backend.src.datasets.simple_shapes_datasets import get_multimodal_dataset, get_shapes_dataset
from backend.src.datasets.ellipse_generation import load_ensemble_ellipses
from backend.src.utils import get_distance_transform
from backend.src.contour_depths import border_depth, band_depth

num_members = 50
num_rows = num_cols = 300

labels_gt = None
# ensemble = circles_with_outliers(num_members, num_rows, num_cols, num_outliers=int(num_members * 0.1))
# ensemble = circles_multiple_radii_modes(num_members, num_rows, num_cols, num_modes=2)
# ensemble, labels_gt, outliers_gt = get_multimodal_dataset(num_rows, num_cols, num_members)
# ensemble = get_shapes_dataset(num_rows, num_cols, num_members)
# ensemble = load_ensemble_ellipses(num_members, num_rows, num_cols, params_set=1, cache=False)
# ensemble = [m["data"].astype(float) for m in ensemble["ensemble"]["members"]]

print(labels_gt)

sdfs = [get_distance_transform(e, tf_type="signed") for e in ensemble]
sdfs = np.concatenate([sdf.flatten().reshape(1, -1) for sdf in sdfs], axis=0)
pca = PCA(n_components=10)
X_sdf = pca.fit_transform(sdfs)

border_functions = border_depth.get_border_functions(ensemble, ensemble)
X_border = border_depth.get_hist_feat_mat(border_functions, density=False, bins=10)

D_sdf = squareform(pdist(X_sdf, metric="euclidean"))
D_border = squareform(pdist(X_border, metric="correlation"))

Z_sdf = linkage(X_sdf, method="ward", metric="euclidean")
Z_border = linkage(X_border, method="average", metric="correlation")

# General clustering computation

fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(12, 6))

axs[0, 0].matshow(X_sdf.T)
axs[0, 1].matshow(D_sdf)
dendrogram(Z_sdf, ax=axs[0, 2])

axs[1, 0].matshow(X_border.T)
axs[1, 1].matshow(D_border)
dendrogram(Z_border, ax=axs[1, 2])

plt.show()

# Clustering results
num_clusters = 2
sdf_labels = fcluster(Z_sdf, criterion="maxclust", t=num_clusters) - 1
border_labels = fcluster(Z_border, criterion="maxclust", t=num_clusters) - 1

fig, axs = plt.subplots(ncols=2, nrows=1)
axs[0].set_title("SDF clustering")
axs[1].set_title("Depth clustering")
for i, clust in enumerate([sdf_labels, border_labels]):
    for j, e in enumerate(ensemble):
        for c in find_contours(e):
            axs[i].plot(c[:, 1], c[:, 0], c=plt.cm.get_cmap("tab10")(clust[j]))
for ax in axs.flatten():
    ax.set_axis_off()
    ax.set_axis_off()
plt.show()

color_by = "depths"
if labels_gt is None:
    fig, axs = plt.subplots(nrows=num_clusters, ncols=2, figsize=(3 * 2, 4 * num_clusters), layout="tight")
    titles = ["sdf_clustering", "depth_clustering"]
    clusterings = [sdf_labels, border_labels]
else:
    fig, axs = plt.subplots(nrows=num_clusters, ncols=3, figsize=(3 * 3, 4 * num_clusters), layout="tight")
    titles = ["gt_clustering", "sdf_clustering", "depth_clustering"]
    clusterings = [np.array(labels_gt), sdf_labels, border_labels]
for label in range(num_clusters):
    print(label)
    for j, clustering in enumerate(clusterings):
        if label == 0:
            axs[0, j].set_title(titles[j])
        clustering_idx = np.where(clustering == label)[0].tolist()
        sub_ensemble = [e for i, e in enumerate(ensemble) if i in clustering_idx]
        if len(sub_ensemble) > 0:
            depths = border_depth.compute_depths(sub_ensemble)
        else:
            depths = []

        if len(sub_ensemble) > 0:
            axs[label, j].imshow(np.ones_like(sub_ensemble[0]), cmap="gray_r")
        for k, e in enumerate(sub_ensemble):
            if color_by == "depths":
                color = plt.cm.get_cmap("inferno")(depths[k])
            else:
                color = plt.cm.get_cmap("tab10")(label)
            for c in find_contours(e):
                axs[label, j].plot(c[:, 1], c[:, 0], c=color)

for ax in axs.flatten():
    ax.set_axis_off()
plt.show()
