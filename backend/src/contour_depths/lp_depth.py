

import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
from backend.src.utils import get_distance_transform

def compute_depths(data_cloud: list,
                   multiplicities: list = None,
                   points_idx: list = None,
                   ord: int = 2,
                   pca_comp: int = None,
                   return_data_mat: bool = False):
    """
    Computes Lp depths for all points in a data_cloud.
    If a list of ids is specified (points_idx), then the depth
    is only computed for the points in the list.
    """

    num_points = len(data_cloud)
    if points_idx is None:
        points_idx = list(range(num_points))

    if multiplicities is None:
        multiplicities = np.ones(num_points)
    else:
        multiplicities = np.array(multiplicities).flatten()

    # Compute SDFs mat (data matrix)
    data_mat = [get_distance_transform(cm, tf_type="signed").flatten() for cm in data_cloud]
    data_mat = np.array(data_mat)

    # Reduce dimensionality of SDF array using PCA
    if pca_comp is not None:
        if pca_comp == -1:
            pca_comp = None
        pca = PCA(n_components=pca_comp)
        data_mat = pca.fit_transform(data_mat)
    num_dim_member = data_mat.shape[1]

    # Compute depth matrix (D(x_i, P) = )
    diff_mat = np.zeros((len(points_idx), num_points, num_dim_member))
    for i, point_id in enumerate(points_idx):
        for j in range(num_points):
            if point_id != j:
                diff_mat[i, j] = data_mat[point_id, :] - data_mat[j, :]

    norms_mat = norm(diff_mat, ord=ord, axis=-1)

    diff_mat_norm = diff_mat / (norms_mat[:, :, np.newaxis] + np.finfo(float).eps)

    diff_mat_norm *= multiplicities.reshape(1, -1, 1)

    depths = diff_mat_norm.sum(axis=1)  # R(y)

    depths = norm(depths, ord=ord, axis=-1)

    depths = depths - multiplicities[points_idx]

    depths = np.maximum(0, depths)/multiplicities.sum()

    depths = 1 - depths

    if return_data_mat:
        return depths, norms_mat
    else:
        return depths


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from skimage.measure import find_contours
    from skimage.draw import disk

    ensemble_members = []
    for i in range(30):
        d_arr = np.zeros((300, 300))
        rr, cc = disk((300//2, 300//2), np.random.normal(1, 0.1, 1)[0] * 100, shape=d_arr.shape)
        d_arr[rr, cc] = 1
        ensemble_members.append(d_arr)

    plt.imshow(np.zeros_like(ensemble_members[0]))
    for i, member in enumerate(ensemble_members):
        for contour in find_contours(member, 0.5):
            plt.plot(contour[:, 1], contour[:, 0])
    plt.show()

    depths = compute_depths(ensemble_members, pca_comp=2)
    depths_ref = compute_depths(ensemble_members, points_idx=[1, ], pca_comp=2)

    color_map = plt.cm.get_cmap("Purples")
    depths_cols = ((1-(depths / depths.max())) * 255).astype(int)
    plt.imshow(np.zeros_like(ensemble_members[0]), cmap="gray")
    for i, member in enumerate(ensemble_members):
        for contour in find_contours(member, 0.5):
            plt.plot(contour[:, 1], contour[:, 0], c=color_map(depths_cols[i]))
    plt.show()
