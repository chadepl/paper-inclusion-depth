
if __name__ == "__main__":
    from time import time
    import numpy as np
    from statsmodels.graphics.functional import banddepth
    from scipy.ndimage import distance_transform_edt
    import matplotlib.pyplot as plt

    from src.contour_depths.depths import band_depth
    from src.contour_depths.datasets.bd_paper import get_contaminated_contour_ensemble_center, \
        get_contaminated_contour_ensemble_magnitude, \
        get_contaminated_contour_ensemble_shape, \
        get_contaminated_contour_ensemble_topological
    from src.contour_depths.vis_utils import plot_contour_spaghetti, plot_contour_boxplot

    ensemble = get_contaminated_contour_ensemble_topological(30, 512, 512)
    sdfs = [distance_transform_edt(e) + distance_transform_edt(1 - e) * -1 for e in ensemble]
    sdfs = [sdf.flatten().reshape(1, -1) for sdf in sdfs]
    sdfs = np.concatenate(sdfs, axis=0)

    t_tick = time()
    depths = band_depth.compute_depths(ensemble, modified=False, target_mean_depth=None)
    print(f"normal version: {time() - t_tick}")
    t_tick = time()
    depths_fast = banddepth(sdfs, method="BD2")
    print(f"fast version: {time() - t_tick}")

    print()
    print(depths)
    print()
    print(depths_fast)

    plot_contour_spaghetti(ensemble)
    plt.show()

    plot_contour_boxplot(ensemble, depths, epsilon_out=5)
    plt.show()

    plot_contour_boxplot(ensemble, depths_fast, epsilon_out=5)
    plt.show()