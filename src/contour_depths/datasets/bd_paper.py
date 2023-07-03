"""
Synthetic datasets we used in the Boundary Depths paper.
"""

import math
import numpy as np
from scipy.spatial.distance import cdist
from skimage.draw import disk


# Gaussian process to generate shapes

def get_base_gp(num_members, domain_points, scale=0.01, sigma=1.0):
    """
    Gaussian process with exponentiated quadratic kernel (https://peterroelants.github.io/posts/gaussian-process-tutorial/).
    :param num_members:
    :param domain_points:
    :param scale: Scale of the noise (the smaller the closest the shape remains to a circle)
    :param sigma: Decrease to obtain higher frequency variations (1 is closer to circle)
    :return:
    """
    num_vertices = domain_points.size

    gp_mean = np.zeros(num_vertices)
    # gp_cov = np.diag(np.ones(num_vertices))
    # gp_cov = gp_smoother * np.exp(-(1 / (2 * gp_sigma)) * cdist(thetas.T, thetas.T, "sqeuclidean"))
    gp_cov = scale * np.exp(-(1 / (2 * sigma)) * cdist(domain_points, domain_points, "sqeuclidean"))

    # - sample from gaussian process
    gp_sample = np.random.multivariate_normal(gp_mean, gp_cov, num_members)
    return gp_sample


def get_xy_coords(angles, radii, scale_x=1, scale_y=1):
    num_members = radii.shape[0]
    x = radii * np.cos(angles.repeat(num_members, axis=0))
    y = radii * np.sin(angles.repeat(num_members, axis=0))

    # Compute bounding box to ensure it fits in frame
    xlim = np.maximum(np.abs(x.min()), np.abs(x.max()))
    ylim = np.maximum(np.abs(y.min()), np.abs(y.max()))
    lim = np.maximum(xlim, ylim) + 0.2  # add some padding

    x = ((x / lim) + 1) * 0.5  # polygon is in [0, 1] box
    y = ((y / lim) + 1) * 0.5  # polygon is in [0, 1] box

    x *= scale_x
    y *= scale_y

    return x, y


def rasterize_coords(coords, num_rows, num_cols):
    from skimage.draw import polygon2mask
    contours = []
    for coords_arr in coords:
        mask = polygon2mask((num_rows, num_cols), coords_arr[:, [1, 0]]).astype(float)
        contours.append(mask)
    return contours


# Dataset center contaminatino

def get_contaminated_contour_ensemble_center(num_members, num_rows, num_cols, num_vertices=100):
    center_x, center_y = [num_cols // 2, num_rows // 2]
    centers_x = [center_x + perturbation for perturbation in np.random.normal(0, 10, num_members)]
    centers_y = [center_y + perturbation for perturbation in np.random.normal(0, 10, num_members)]
    contours = []
    for i in range(num_members):
        contour = np.zeros((num_rows, num_cols))
        rr, cc = disk((centers_y[i], centers_x[i]), (np.minimum(num_rows, num_cols) / 2) * 0.5, shape=contour.shape)
        contour[rr, cc] = 1
        contours.append(contour)
    return contours


# Dataset Amplitude (magnitude contamination)

def get_contaminated_contour_ensemble_magnitude(num_members, num_rows, num_cols, case=0, num_vertices=100,
                                                p_contamination=0.2, return_labels=False,
                                                use_ellipse=False, ellipse_kwargs=None):
    thetas = np.linspace(0, 2 * np.pi, num_vertices).reshape(1, -1)

    if use_ellipse:
        rotation_rad = np.deg2rad(0)
        ellipticity = 0.6
        translation_x = 0
        translation_y = 0

        a = 1
        b = -((ellipticity ** 2) * (a ** 2) - (a ** 2))

        # Find radius
        x_term = np.square(np.cos(thetas) / (a ** 2))
        y_term = np.square(np.sin(thetas) / (b ** 2))
        r = np.sqrt(1 / (x_term + y_term))

        # - Create domain coordinates X \in [-1,1] and Y \in [-1, 1]
        X = 1 * np.cos(thetas)
        Y = 1 * np.sin(thetas)
        # X = np.repeat(np.linspace(-1, 1, num_cols).reshape(1, -1), num_rows, axis=0)
        # Y = np.repeat(np.linspace(1, -1, num_rows).reshape(-1, 1), num_cols, axis=1)

        # - Rotate domain by rotation_deg degrees
        X1 = X * math.cos(rotation_rad) - Y * math.sin(rotation_rad)
        Y1 = X * math.sin(rotation_rad) + Y * math.cos(rotation_rad)

        # - Translate the domain using the translation vector
        X1 = X1 - translation_x  # Subtracting the translation vector has the opposite visual effect
        Y1 = Y1 - translation_y  # Subtracting the translation vector has the opposite visual effect

        # - Scale domain (from circle to ellipse)
        X1 = X1 / a
        Y1 = Y1 / b

        base_function = r  # np.sqrt(np.square(X1) + np.square(Y1))
        gp_sample = get_base_gp(num_members, base_function.T, scale=0.01, sigma=0.001)  # np.cos(thetas.T))
    else:
        base_function = np.ones_like(thetas) * 1  # if we want constant radius (for a circle)
        gp_sample = get_base_gp(num_members, np.cos(thetas.T))

    # Sampling radii functions from a Gaussian Process

    # ys = np.sin(thetas) / 0.8
    # rot_xs = xs * np.cos(np.pi/2) - ys * np.sin(np.pi / 2)
    # rot_ys = xs * np.sin(np.pi/2) + ys * np.cos(np.pi / 2)
    # base_function = np.sqrt(np.square(rot_xs) + np.square(rot_ys))

    radii = base_function + gp_sample

    labels = np.zeros(num_members)  # keep track of outliers

    # Other models

    if case == 1:  # symmetric contamination
        should_contaminate = (np.random.random(num_members) > (1 - p_contamination)).astype(float)
        contaminate_side = np.random.choice([-1, 1], num_members)
        radii_delta = 0.3 * should_contaminate * contaminate_side
        radii_delta = radii_delta.reshape(-1, 1).repeat(num_vertices, axis=1)
        radii += radii_delta
        labels[np.where(should_contaminate > 0)[0]] = 1

    if case == 2:  # partially/ peak contaminated contaminated
        should_contaminate = (np.random.random(num_members) > (1 - p_contamination)).astype(float)
        contaminate_side = np.random.choice([-1, 1], num_members)
        inf_bound = np.random.random(num_members) * np.pi * 2
        excess = 2 * np.pi - inf_bound
        sup_bound = np.random.random(num_members) * excess + inf_bound
        for i in range(num_members):
            idx = np.where(np.logical_and(thetas.flatten() > inf_bound[i], thetas.flatten() < sup_bound[i]))[0]
            radii_delta = 0.3 * should_contaminate[i] * contaminate_side[i]
            radii[i, idx] += radii_delta
        labels[np.where(should_contaminate > 0)[0]] = 1

    xs, ys = get_xy_coords(thetas, radii, num_cols, num_rows)

    coords = [np.array([xs[i], ys[i]]).squeeze().T for i in range(num_members)]
    contours = rasterize_coords(coords, num_rows, num_cols)

    if return_labels:
        return contours, labels
    else:
        return contours


# Dataset Shape

def get_contaminated_contour_ensemble_shape(num_members, num_rows, num_cols, num_vertices=100, p_contamination=0.2,
                                            scale=0.01, freq=0.01, return_labels=False):
    thetas = np.linspace(0, 2 * np.pi, num_vertices).reshape(1, -1)

    # Sampling radii functions from a Gaussian Process
    base_function = 1
    gp_sample_sample = get_base_gp(num_members, np.cos(thetas.T))
    gp_sample_outliers = get_base_gp(num_members, np.cos(thetas.T), scale, freq)

    should_contaminate = (np.random.random(num_members) > (1 - p_contamination)).astype(float).reshape(num_members,
                                                                                                       -1).repeat(
        num_vertices, axis=1)
    radii = base_function + (gp_sample_sample * (1 - should_contaminate)) + (gp_sample_outliers * should_contaminate)

    xs, ys = get_xy_coords(thetas, radii, num_cols, num_rows)

    coords = [np.array([xs[i], ys[i]]).squeeze().T for i in range(num_members)]
    contours = rasterize_coords(coords, num_rows, num_cols)

    labels = should_contaminate[:, 0].astype(int)

    if return_labels:
        return contours, labels
    else:
        return contours


def get_contaminated_contour_ensemble_topological(num_members, num_rows, num_cols, num_vertices=100,
                                                  p_contamination=0.2, return_labels=False):
    thetas = np.linspace(0, 2 * np.pi, num_vertices).reshape(1, -1)
    base_function = 1
    gp_sample_sample = get_base_gp(num_members, np.cos(thetas.T), scale=0.06, sigma=10)
    gp_sample_outliers = get_base_gp(num_members, np.cos(thetas.T), 0.1, 0.1)

    radii = base_function + gp_sample_sample
    radii_out = base_function + gp_sample_outliers

    xs, ys = get_xy_coords(thetas, radii, num_cols, num_rows)

    # Generation of disconnected components and holes
    random_scaling = np.random.randint(35, 70, num_members).reshape(-1, 1)
    scaled_radii_out = radii_out / random_scaling
    xs_out, ys_out = get_xy_coords(thetas, scaled_radii_out, num_cols, num_rows)

    random_pos = np.random.choice(np.arange(xs.shape[1]), xs.shape[0],
                                  replace=True)  # we get a point along the boundary per curve
    random_xs = np.array([xs[i, rp] for i, rp in enumerate(random_pos)]).reshape(-1, 1) - num_cols // 2
    random_ys = np.array([ys[i, rp] for i, rp in enumerate(random_pos)]).reshape(-1, 1) - num_rows // 2

    magnitude = np.sqrt(np.square(random_xs) + np.square(random_ys))
    angle = np.array([thetas[0, rp] for i, rp in enumerate(random_pos)]).reshape(-1,
                                                                                 1)  # np.arctan2(random_xs, random_ys)

    topo_feat_sign = np.random.random(num_members)
    topo_feat_sign[topo_feat_sign > 0.5] = 1
    topo_feat_sign[topo_feat_sign <= 0.5] = -1
    topo_feat_sign = topo_feat_sign.reshape(-1, 1)
    xs_out = xs_out - num_cols // 2
    ys_out = ys_out - num_cols // 2
    max_xs_out = np.abs(xs_out).max(axis=1).reshape(-1, 1)
    max_ys_out = np.abs(ys_out).max(axis=1).reshape(-1, 1)
    random_margin = np.random.randint(5, 20, num_members).reshape(-1, 1)
    xs_out = xs_out + (magnitude + topo_feat_sign * (max_xs_out + random_margin)) * np.cos(
        angle)  # + scaled_radii_out.max(axis=1).reshape(-1, 1)*(num_cols)) * np.cos(angle)#(scaled_radii_out * num_cols//2).max(axis=1).reshape(-1, 1)
    ys_out = ys_out + (magnitude + topo_feat_sign * (max_ys_out + random_margin)) * np.sin(
        angle)  # + scaled_radii_out.max(axis=1).reshape(-1, 1)*(num_rows)) * np.sin(angle) #(scaled_radii_out * num_rows//2).max(axis=1).reshape(-1, 1)
    xs_out = xs_out + num_cols // 2
    ys_out = ys_out + num_cols // 2
    # if its a hole then radius + my radius + buffer ... opposite if itsa disconnected component

    coords = [np.array([xs[i], ys[i]]).squeeze().T for i in range(num_members)]
    contours = rasterize_coords(coords, num_rows, num_cols)

    coords_out = [np.array([xs_out[i], ys_out[i]]).squeeze().T for i in range(num_members)]
    contours_out = rasterize_coords(coords_out, num_rows, num_cols)

    should_contaminate = (np.random.random(num_members) > (1 - p_contamination)).astype(float).reshape(num_members,
                                                                                                       -1).repeat(
        num_vertices, axis=1)
    contamination_type = np.random.randint(0, 2, num_members)

    contours_final = []
    for i, (bool_cont, cont_type) in enumerate(zip(should_contaminate[:, 0], contamination_type)):
        if bool_cont == 1:
            if cont_type == 0:  # only one
                cf = contours_out[i]
            elif cont_type == 1:  # both
                cf = (contours[i] + contours_out[i]) == 1
        else:
            cf = contours[i]

        contours_final.append(cf.astype(float))

    labels = should_contaminate[:, 0].astype(int)

    if return_labels:
        return contours_final, labels
    else:
        return contours_final

#from skimage.draw imp
def get_problematic_case(num_rows, num_cols):
    ensemble = []
    inner_circle_r = num_rows//10
    for i in range(10): # inside circles, no overlap
        mask = np.zeros((num_rows, num_cols))
        x = num_rows//2 + 2 * inner_circle_r * np.cos(i * 2 * np.pi * (1 / 10))
        y = num_cols//2 + 2 * inner_circle_r * np.sin(i * 2 * np.pi * (1 / 10))
        rr, cc = disk((x, y), inner_circle_r * 0.5, shape=(num_rows, num_cols))
        mask[rr, cc] = 1
        ensemble.append(mask)

    for i in range(10):  # outside circles, no overlap
        mask = np.zeros((num_rows, num_cols))
        rr, cc = disk((num_rows // 2, num_cols // 2), (num_rows // 3)+5*i, shape=(num_rows, num_cols))
        mask[rr, cc] = 1
        ensemble.append(mask)

    return ensemble

def get_han_dataset_ParotidR(num_rows, num_cols):
    from backend.src.datasets.han_ensembles import get_han_slice_ensemble
    img, gt, ensemble_masks = get_han_slice_ensemble(num_rows, num_cols, patient_id=0, structure_name="Parotid_R",
                                                     slice_num=41)
    return img, gt, ensemble_masks


def get_han_dataset_BrainStem(num_rows, num_cols):
    from backend.src.datasets.han_ensembles import get_han_slice_ensemble
    img, gt, ensemble_masks = get_han_slice_ensemble(num_rows, num_cols, patient_id=0, structure_name="BrainStem",
                                                     slice_num=31)
    return img, gt, ensemble_masks


if __name__ == "__main__":
    import numpy as np
    from skimage.draw import ellipse
    from skimage.measure import find_contours
    import matplotlib.pyplot as plt
    from backend.src.vis_utils import plot_contour_spaghetti

    # Figure of synthetic datasets
    num_members = 10
    num_rows = num_cols = 540
    alpha = 0.3

    titles = [
        # "Base Model",
        # "Magnitude",
        # "Peaks",
        # "Shape_in",
        # "Shape_out",
        "Topo"]
    datasets = [
        # get_contaminated_contour_ensemble_magnitude(num_members, num_rows, num_cols, num_vertices=1000, case=0,
        #                                                              p_contamination=0.2, return_labels=True, use_ellipse=False),
        # get_contaminated_contour_ensemble_magnitude(num_members, num_rows, num_cols, case=1,
        #                                                          p_contamination=0.2, return_labels=True),
        # get_contaminated_contour_ensemble_magnitude(num_members, num_rows, num_cols, case=2, p_contamination=0.2,
        #                                             return_labels=True),
        # get_contaminated_contour_ensemble_shape(num_members, num_rows, num_cols, scale=0.01, freq=0.01,
        #                                         p_contamination=0.2, return_labels=True),
        # get_contaminated_contour_ensemble_shape(num_members, num_rows, num_cols, scale=0.05, freq=0.05,
        #                                         p_contamination=0.2, return_labels=True),
        get_contaminated_contour_ensemble_topological(num_members, num_rows, num_cols, p_contamination=0.1,
                                                      return_labels=True)
    ]

    for i, t in enumerate(titles):
        fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
        ensemble, labs = datasets[i]
        plot_contour_spaghetti(ensemble, arr=labs, is_arr_categorical=True, linewidth=2, alpha=alpha, ax=ax)
        plt.show()
        # fig.savefig(f"/Users/chadepl/Downloads/{t}.png")

    # Figure of real dataset

    # #img, gt, ensemble = get_han_dataset_ParotidR(num_rows, num_cols)
    # img, gt, ensemble = get_han_dataset_BrainStem(num_rows, num_cols)
    #
    # fig, ax = plt.subplots()
    #
    # #plt.imshow(img)
    # plot_contour_spaghetti(ensemble, under_mask=img, ax=ax)
    #
    # plt.show()
