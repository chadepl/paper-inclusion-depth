"""
In this script we show the limitations of the CBD methodology.
There are two components to this methodology: the depth (order) computation
 and the visual encoding.
- Poor scaling.
- Misses global distribution characteristics like spread and skewness.
  - Does not leverage notion of left/right, inside/outside
- Insensitive to local changes in distribution of shapes.
- Misses changes in topology.
- Not sensitive to all shape deviations.
"""

import numpy as np
from skimage.draw import ellipse
from skimage.measure import find_contours
import matplotlib.pyplot as plt

from backend.src.utils import get_distance_transform
from backend.src.contour_depths import band_depth, lp_depth
from backend.src.vis_utils import plot_contour_boxplot

experiment_id = ["global", "local"][1]

################
# Poor scaling #
################

# We show this in the depth_comparison.py script where
# we compare CBD against other types of depths.


##############################################
# Misses global distribution characteristics #
##############################################

if experiment_id == "global":
    def generate_ensembles_varying_global_statistics(num_members, num_rows, num_cols, case=0):
        # Unimodal
        # - Case 0: Tight vs sparse
        # - Case 1: Uniform vs gaussian
        # Bimodal
        # - One mode towards the center vs two modes

        # Generate N points using the distributions
        # Order the curves based on the order of the parameter
        mean_radius = num_rows // 3
        free_space = (num_rows // 2) - mean_radius
        upper_limit = mean_radius + free_space * 0.5
        lower_limit = mean_radius - free_space * 0.5

        ensembles_params = []

        def random_points_within_bounds(fn, args, kwargs, lb, ub):
            p = fn(*args, **kwargs)
            while np.where(np.logical_or(p < lb, p > ub))[0].size > 0:
                p = fn(*args, **kwargs)
            p = np.sort(p).tolist()
            p = [lower_limit, ] + p + [upper_limit, ]
            return p

        if case == 0:  # tight vs sparse (std in gaussian)
            ensembles_labels = ["sparse", "tight"]

            params = random_points_within_bounds(np.random.normal,
                                                 (mean_radius, free_space//3, num_members - 2),
                                                 dict(),
                                                 lower_limit, upper_limit)
            ensembles_params.append(params)

            params = random_points_within_bounds(np.random.normal,
                                                 (mean_radius, free_space // 10, num_members - 2),
                                                 dict(),
                                                 lower_limit, upper_limit)
            ensembles_params.append(params)

        elif case == 1:  # normal vs uniform
            ensembles_labels = ["normal", "uniform"]

            params = random_points_within_bounds(np.random.normal,
                                                 (mean_radius, free_space // 3, num_members - 2),
                                                 dict(),
                                                 lower_limit, upper_limit)
            ensembles_params.append(params)

            params = np.linspace(lower_limit, upper_limit, num_members).tolist()
            ensembles_params.append(params)

        elif case == 2:  # unimodal vs multimodal
            ensembles_labels = ["unimodal", "bimodal"]

            params = random_points_within_bounds(np.random.normal,
                                                 (mean_radius, free_space // 3, num_members - 2),
                                                 dict(),
                                                 lower_limit, upper_limit)
            ensembles_params.append(params)

            params1 = random_points_within_bounds(np.random.normal,
                                                 (mean_radius + free_space // 3, free_space // 6, (num_members // 2) - 1),
                                                 dict(),
                                                 lower_limit, upper_limit)
            params2 = random_points_within_bounds(np.random.normal,
                                                 (mean_radius - free_space // 3, free_space // 6, (num_members // 2) - 1),
                                                 dict(),
                                                 lower_limit, upper_limit)
            params = params1[:-1] + params2[1:]
            ensembles_params.append(params)

        ensembles = []
        for ensemble_param_values in ensembles_params:
            ensemble_param_values = np.sort(ensemble_param_values)
            members = []
            for radius in ensemble_param_values:
                rr, cc = ellipse(num_rows / 2, num_cols / 2, radius, radius, shape=(num_rows, num_cols))
                em = np.zeros((num_rows, num_cols))
                em[rr, cc] = 1
                members.append(em)
            ensembles.append(members)

        return zip(ensembles_labels, ensembles, ensembles_params)




    num_members = 100
    num_rows = num_cols = 300

    ensembles = generate_ensembles_varying_global_statistics(num_members,
                                                             num_rows,
                                                             num_cols,
                                                             case=2)
    ensembles_labels, ensembles_members, ensembles_values = list(zip(*ensembles))

    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    axs[0].imshow(np.zeros((num_rows, num_cols)), cmap="Greys")
    axs[1].imshow(np.zeros((num_rows, num_cols)), cmap="Greys")

    for i, ensemble_members in enumerate(ensembles_members):
        for member in ensemble_members:
            contours = find_contours(member, 0.5)
            for c in contours:
                axs[i].plot(c[:, 0], c[:, 1], alpha=0.1, c="purple")
            axs[i].set_title(ensembles_labels[i])
            axs[i].set_axis_off()
    plt.show()

    depths_bd = []
    depths_l1 = []
    for ensemble_members in ensembles_members:
        depths_bd.append(band_depth.compute_depths(ensemble_members))
        depths_l1.append(lp_depth.compute_depths(ensemble_members))

    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    for i, ensemble_members in enumerate(ensembles_members):
        axs[i].set_title(ensembles_labels[i])
        plot_contour_boxplot(ensemble_members, depths_bd[i], epsilon_out=0.01, ax=axs[i])
    plt.show()


    fig, ax = plt.subplots()
    for i, d in enumerate(depths_bd):
        ax.scatter(depths_bd[i], np.zeros(num_members) + i, c = ["blue", "orange"][i], label = ensembles_labels[i])
    plt.legend()
    plt.show()



#############################################
# Misses local distribution characteristics #
#############################################


if experiment_id == "local":

    num_members = 50
    num_rows = num_cols = 300

    radius = 100
    pos = 150 + np.linspace(-25, 25, num_members-2)
    radii = np.ones_like(pos) * 100
    pos = [150, ] + pos.tolist() + [150, ]
    radii = [70, ] + radii.tolist() + [130, ]

    #pos = [150, 140, 145, 150, 155, 160, 150]
    #radii = [70, 100, 100, 100, 100, 100, 130]

    ensemble_members = []
    for p, r in zip(pos, radii):
        rr, cc = ellipse(300//2, p, r, r)
        ensemble_members.append(np.zeros((300, 300)))
        ensemble_members[-1][rr, cc] = 1

    full_depths = band_depth.compute_depths(ensemble_members)
    left_depths = band_depth.compute_depths([em[:, :num_cols//2] for em in ensemble_members])
    right_depths = band_depth.compute_depths([em[:, (num_cols // 2):] for em in ensemble_members])

    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    axs[0].imshow(np.zeros((num_rows, num_cols)), cmap="Greys")
    axs[1].imshow(np.zeros((num_rows, num_cols)), cmap="Greys")

    for i, member in enumerate(ensemble_members):
        contours = find_contours(member, 0.5)
        for c in contours:
            axs[0].plot(c[:, 1], c[:, 0], alpha=full_depths[i], c="purple")
        axs[0].set_axis_off()
    plt.show()


