import numpy as np
from backend.src.contour_depths import band_depth, lp_depth, sdf_depth


def compute_relative_depths(ensemble_members: list, clustering: list, subset: list = None, depth_fn="l1"):
    # What this method basically does is to call other depths multiple times
    # For every ensemble member (or the subset, if specified), it computes the within depth, between depth and
    # relative depths (Dw - Db).

    num_ensemble_members = len(ensemble_members)
    if subset is None:
        subset = np.arange(len(ensemble_members))

    clustering = np.array(clustering)

    clusters_idx = np.unique(clustering)

    within_depths = np.zeros(num_ensemble_members)
    for clusters_id in clusters_idx:
        within_members = np.where(clustering == clusters_id)[0]
        d = band_depth.compute_band_depths([ensemble_members[i] for i in within_members])
        for i, j in enumerate(within_members):
            within_depths[j] = d[i]

    return within_depths


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from skimage.measure import find_contours
    from skimage.draw import disk

    ensemble_members = []
    for i in range(50):
        d_arr = np.zeros((300, 300))
        rr, cc = disk((300//2, 300//2 - 30), np.random.normal(1, 0.02, 1)[0] * 50, shape=d_arr.shape)
        d_arr[rr, cc] = 1
        ensemble_members.append(d_arr)

    for i in range(50):
        d_arr = np.zeros((300, 300))
        rr, cc = disk((300//2, 300//2 + 30), np.random.normal(1, 0.02, 1)[0] * 50, shape=d_arr.shape)
        d_arr[rr, cc] = 1
        ensemble_members.append(d_arr)

    plt.imshow(np.zeros_like(ensemble_members[0]))
    for i, member in enumerate(ensemble_members):
        for contour in find_contours(member, 0.5):
            plt.plot(contour[:, 1], contour[:, 0])
    plt.show()

    clustering = [0 for _ in range(50)] + [1 for _ in range(50)]

    depths = band_depth.compute_band_depths(ensemble_members)
    red = compute_relative_depths(ensemble_members, clustering)

    d_sort = np.argsort(depths)
    red_sort = np.argsort(red)

    colors = [(1,0,0,1), (0,0,1,1)]
    plt.bar(x=np.arange(depths.size), height=red[d_sort], color=[colors[i] for i in np.array(clustering)[red_sort]]);
    plt.show()

    plt.bar(x=np.arange(depths.size), height=depths[d_sort], color=[colors[i] for i in np.array(clustering)[d_sort]]);
    plt.show()
