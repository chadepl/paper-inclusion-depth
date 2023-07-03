import numpy as np
from numpy.linalg import norm
from scipy.optimize import bisect
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import find_contours
from backend.src.utils import get_distance_transform


def compute_hausdorff_depths(ensemble_members, d_type="hausdorff", target_mean_depth=1 / 6):
    # Get SDFs of all ensemble_members
    # Discretize all ensemble_members (get equally spaced boundary points)
    # For ensemble member compute the symmetric hausdorff distance with respect to other members
    # - Members that are further apart are less central
    # We average the distances per member, which yields a type A depth (assume boundness)
    # We could also try a type B function (1 + E[h])^-1 which assumes unboundness

    num_ensemble_members = len(ensemble_members)

    # Compute SDFs mat (data matrix)
    sdfs = [get_distance_transform(cm, tf_type="unsigned") for cm in ensemble_members]

    # Members contours (parametrizations)
    contours = [np.concatenate(find_contours(em, 0.5), axis=0) for em in ensemble_members]
    interps = [RegularGridInterpolator((np.arange(sdf.shape[0]), np.arange(sdf.shape[1])), sdf) for sdf in sdfs]

    sdf_samples = dict()
    for i in range(num_ensemble_members):
        for j in range(i, num_ensemble_members):
            if i == j:
                ij_vals = np.zeros_like(contours[i])
                sdf_samples[(j, i)] = dict(vals=ij_vals,
                                           min=ij_vals.min(),
                                           max=ij_vals.max(),
                                           mean=ij_vals.mean(),
                                           std=ij_vals.std(),
                                           num_el=ij_vals.size)
            else:
                ij_vals = interps[j](contours[i])
                sdf_samples[(i, j)] = dict(vals=ij_vals,
                                           min=ij_vals.min(),
                                           max=ij_vals.max(),
                                           mean=ij_vals.mean(),
                                           std=ij_vals.std(),
                                           num_el=ij_vals.size)

                ji_vals = interps[i](contours[j])
                sdf_samples[(j, i)] = dict(vals=ji_vals,
                                           min=ji_vals.min(),
                                           max=ji_vals.max(),
                                           mean=ji_vals.mean(),
                                           std=ji_vals.std(),
                                           num_el=ji_vals.size)

    if d_type == "band":
        t_band = np.array([d["mean"] for d in sdf_samples.values()]).mean()
        print(f"[sdfd] Threshold for bands: {t_band}")

    dists = np.zeros((num_ensemble_members, num_ensemble_members))
    for i in range(num_ensemble_members):
        for j in range(i, num_ensemble_members):
            if i != j:
                if d_type == "hausdorff":
                    dists[i, j] = np.maximum(sdf_samples[(i, j)]["max"], sdf_samples[(j, i)]["max"])
                    dists[j, i] = dists[i, j]
                elif d_type == "l2":
                    dists[i, j] = np.maximum(norm(sdf_samples[(i, j)]["vals"], ord=2),
                                             norm(sdf_samples[(j, i)]["vals"], ord=2))
                    dists[j, i] = dists[i, j]
                elif d_type == "band":
                    dij = np.where(sdf_samples[(i, j)]["vals"] > t_band)[0].size / sdf_samples[(i, j)]["num_el"]
                    dji = np.where(sdf_samples[(j, i)]["vals"] > t_band)[0].size / sdf_samples[(j, i)]["num_el"]
                    dists[i, j] = np.maximum(dij, dji)
                    dists[j, i] = dists[i, j]

    if d_type == "hausdorff":
        depth_type = "B"  # unbounded
    elif d_type == "l2":
        depth_type = "B"  # unbounded
    elif d_type == "band":
        # larger dists, mean more is falling outside
        # therefore, less is more central
        # furthermore is bounded between (0 and 1)
        depth_type = "A"  # bounded

    if depth_type == "A":
        # TODO: should we find a way to threshold the dist matrix like in CBD?
        depths = 1 - dists.mean(axis=1)

    elif depth_type == "B":
        depths = dists.mean(axis=1)
        depths = 1 / (1 + depths)  # Type B depth function

    return depths


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from skimage.measure import find_contours
    from skimage.draw import disk

    ensemble_members = []
    for i in range(100):
        d_arr = np.zeros((300, 300))
        rr, cc = disk((300 // 2, 300 // 2), np.random.normal(1, 0.1, 1)[0] * 100, shape=d_arr.shape)
        d_arr[rr, cc] = 1
        ensemble_members.append(d_arr)

    plt.imshow(np.zeros_like(ensemble_members[0]))
    for i, member in enumerate(ensemble_members):
        for contour in find_contours(member, 0.5):
            plt.plot(contour[:, 1], contour[:, 0])
    plt.show()

    depths1, dists1 = compute_hausdorff_depths(ensemble_members, d_type="l2")
    depths2, dists2 = compute_hausdorff_depths(ensemble_members, d_type="band")

    color_map = plt.cm.get_cmap("Purples")
    depths_cols = ((1 - (depths1 / depths1.max())) * 255).astype(int)
    plt.imshow(np.zeros_like(ensemble_members[0]), cmap="gray")
    for i, member in enumerate(ensemble_members):
        for contour in find_contours(member, 0.5):
            plt.plot(contour[:, 1], contour[:, 0], c=color_map(depths_cols[i]))
    plt.show()

    color_map = plt.cm.get_cmap("Purples")
    depths_cols = ((1 - (depths2 / depths2.max())) * 255).astype(int)
    plt.imshow(np.zeros_like(ensemble_members[0]), cmap="gray")
    for i, member in enumerate(ensemble_members):
        for contour in find_contours(member, 0.5):
            plt.plot(contour[:, 1], contour[:, 0], c=color_map(depths_cols[i]))
    plt.show()

    d_argsort = np.argsort(depths1)
    plt.bar(np.arange(depths1.size), depths1[d_argsort])
    plt.show()

    d_argsort = np.argsort(depths2)
    plt.bar(np.arange(depths2.size), depths2[d_argsort])
    plt.show()

    plt.scatter(depths1, depths2)
    plt.show()
