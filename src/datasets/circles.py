"""
 Different datasets with circles to demonstrate
 basic capabilities of our methods
"""

import numpy as np
from skimage.draw import ellipse, rectangle


def random_points_within_bounds(fn, args, kwargs, lb, ub):
    p = fn(*args, **kwargs)
    while np.where(np.logical_or(p < lb, p > ub))[0].size > 0:
        p = fn(*args, **kwargs)
    p = np.sort(p).tolist()
    p = [lb, ] + p + [ub, ]
    return p


def circles_different_radii_spreads(num_members, num_rows, num_cols, high_spread=True):
    mean_radius = num_rows // 3
    free_space = (num_rows // 2) - mean_radius
    upper_limit = mean_radius + free_space * 0.5
    lower_limit = mean_radius - free_space * 0.5

    if high_spread:
        params = random_points_within_bounds(np.random.normal,
                                             (mean_radius, free_space // 3, num_members - 2),
                                             dict(),
                                             lower_limit, upper_limit)
    elif not high_spread:
        params = random_points_within_bounds(np.random.normal,
                                             (mean_radius, free_space // 10, num_members - 2),
                                             dict(),
                                             lower_limit, upper_limit)

    return generate_circle_ensemble(params, num_rows, num_cols)


def circles_different_radii_distribution(num_members, num_rows, num_cols, dist="normal"):
    mean_radius = num_rows // 3
    free_space = (num_rows // 2) - mean_radius
    upper_limit = mean_radius + free_space * 0.5
    lower_limit = mean_radius - free_space * 0.5

    if dist == "normal":
        params = random_points_within_bounds(np.random.normal,
                                             (mean_radius, free_space // 3, num_members - 2),
                                             dict(),
                                             lower_limit, upper_limit)
    elif dist == "uniform":
        params = np.linspace(lower_limit, upper_limit, num_members).tolist()

    return generate_circle_ensemble(params, num_rows, num_cols)


def circles_multiple_radii_modes(num_members, num_rows, num_cols, num_modes=2):
    mean_radius = num_rows // 3
    free_space = (num_rows // 2) - mean_radius
    upper_limit = mean_radius + free_space * 0.5
    lower_limit = mean_radius - free_space * 0.5

    if num_modes == 1:
        params = random_points_within_bounds(np.random.normal,
                                             (mean_radius, free_space // 3, num_members - 2),
                                             dict(),
                                             lower_limit, upper_limit)
    elif num_modes == 2:
        params1 = random_points_within_bounds(np.random.normal,
                                              (mean_radius + free_space // 3, free_space // 6, (num_members // 2) - 1),
                                              dict(),
                                              lower_limit, upper_limit)
        params2 = random_points_within_bounds(np.random.normal,
                                              (mean_radius - free_space // 3, free_space // 6, (num_members // 2) - 1),
                                              dict(),
                                              lower_limit, upper_limit)
        params = params1[:-1] + params2[1:]

    return generate_circle_ensemble(params, num_rows, num_cols)


def circles_with_outliers(num_members, num_rows, num_cols, num_outliers=2):
    circles = circles_different_radii_distribution(num_members - num_outliers, num_rows, num_cols, dist="normal")

    outliers = []
    for i in range(num_outliers):
        r = np.random.normal(num_rows // 3.2, 10, 1)[0]
        start = (num_rows // 2 - r, num_cols // 2 - r)
        end = (num_rows // 2 + r, num_cols // 2 + r)
        rr, cc = rectangle(start, end, shape=(num_rows, num_cols))
        rr, cc = [rr.astype(int), cc.astype(int)]
        outliers.append(np.zeros((num_rows, num_cols)))
        outliers[-1][rr, cc] = 1

    return circles + outliers


def generate_circle_ensemble(ensemble_params, num_rows, num_cols):
    members = []
    for radius in ensemble_params:
        rr, cc = ellipse(num_rows / 2, num_cols / 2, radius, radius, shape=(num_rows, num_cols))
        em = np.zeros((num_rows, num_cols))
        em[rr, cc] = 1
        members.append(em)
    return members


if __name__ == "__main__":
    from skimage.measure import find_contours
    import matplotlib.pyplot as plt

    ensemble = circles_with_outliers(20, 300, 300, num_outliers=5)

    for m in ensemble:
        contour = find_contours(m, 0.5)
        for c in contour:
            plt.plot(c[:, 1], c[:, 0], c="red", alpha=0.2)
    plt.show()
