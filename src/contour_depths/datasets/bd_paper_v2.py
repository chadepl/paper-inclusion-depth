"""
We fix a problem with the datasets in which we adjusted
the bounding box dynamically.
"""

# 1. Create GP with radius set to R with R < 1
# 2. Get GP x, y coordinates
# 3. Rasterize area -1 < x < 1 and -1 < y < 1

import numpy as np
from scipy.spatial.distance import cdist

def get_base_gp(num_members, domain_points, scale=0.01, sigma=1.0):
    thetas = domain_points.flatten().reshape(-1, 1)
    num_vertices = thetas.size
    gp_mean = np.zeros(num_vertices)

    gp_cov_sin = scale * np.exp(-(1 / (2 * sigma)) * cdist(np.sin(thetas), np.sin(thetas), "sqeuclidean"))
    gp_sample_sin = np.random.multivariate_normal(gp_mean, gp_cov_sin, num_members)
    gp_cov_cos = scale * np.exp(-(1 / (2 * sigma)) * cdist(np.cos(thetas), np.cos(thetas), "sqeuclidean"))
    gp_sample_cos = np.random.multivariate_normal(gp_mean, gp_cov_cos, num_members)

    return gp_sample_sin + gp_sample_cos


def get_xy_coords(angles, radii):
    num_members = radii.shape[0]
    angles = angles.flatten().reshape(1,- 1).repeat(num_members, axis=0)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return x, y


def rasterize_coords(x_coords, y_coords, grid_size):
    from skimage.draw import polygon2mask
    masks = []
    for xc, yc in zip(x_coords, y_coords):
        coords_arr = np.concatenate([xc.reshape(-1,1), yc.reshape(-1,1)], axis=1)
        coords_arr *= grid_size//2
        coords_arr += grid_size//2
        mask = polygon2mask((grid_size, grid_size), coords_arr).astype(float)
        masks.append(mask)
    return masks


def get_population_mean(radius, grid_size, num_vertices=100):
    thetas = np.linspace(0, 2 * np.pi, num_vertices).reshape(1, -1)
    radius = np.ones_like(thetas) * radius
    xs, ys = get_xy_coords(thetas, radius)
    mask = rasterize_coords(xs, ys, grid_size=grid_size)
    return mask[0]

def dataset_magnitude_outliers(num_members,
                               grid_size,
                               population_radius=0.5,
                               contamination_offset=0.2,
                               case=0,
                               num_vertices=100,
                               normal_scale=0.003,
                               normal_freq=0.9,
                               p_contamination=0.2,
                               return_labels=False):
    thetas = np.linspace(0, 2 * np.pi, num_vertices)
    population_radius = np.ones_like(thetas) * population_radius  # if we want constant radius (for a circle)
    gp_sample = get_base_gp(num_members, thetas, scale=normal_scale, sigma=normal_freq)

    radii = population_radius + gp_sample
    labels = np.zeros(num_members)  # keep track of outliers

    # Other models
    if case == 1:  # symmetric contamination
        should_contaminate = (np.random.random(num_members) > (1 - p_contamination)).astype(float)
        contaminate_side = np.random.choice([-1, 1], num_members)
        radii_delta = contamination_offset * should_contaminate * contaminate_side
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
            radii_delta = contamination_offset * should_contaminate[i] * contaminate_side[i]
            radii[i, idx] += radii_delta
        labels[np.where(should_contaminate > 0)[0]] = 1

    xs, ys = get_xy_coords(thetas, radii)

    contours = rasterize_coords(xs, ys, grid_size)

    if return_labels:
        return contours, labels
    else:
        return contours


def dataset_no_outliers(num_members,
                        grid_size,
                        num_vertices=100,
                        return_labels=False):
    return dataset_magnitude_outliers(num_members,
                                      grid_size,
                                      population_radius=0.5,
                                      contamination_offset=0.2,
                                      case=0,
                                      num_vertices=num_vertices,
                                      p_contamination=0.1,
                                      return_labels=return_labels)


def dataset_sym_mag_outliers(num_members,
                             grid_size,
                             num_vertices=100,
                             return_labels=False):
    return dataset_magnitude_outliers(num_members,
                                      grid_size,
                                      population_radius=0.5,
                                      contamination_offset=0.2,
                                      case=1,
                                      num_vertices=num_vertices,
                                      p_contamination=0.1,
                                      return_labels=return_labels)

def dataset_peaks_mag_outliers(num_members,
                             grid_size,
                             num_vertices=100,
                             return_labels=False):
    return dataset_magnitude_outliers(num_members,
                                      grid_size,
                                      population_radius=0.5,
                                      contamination_offset=0.2,
                                      case=2,
                                      num_vertices=num_vertices,
                                      p_contamination=0.1,
                                      return_labels=return_labels)

def dataset_shape_outliers(num_members,
                           grid_size,
                           num_vertices=100,
                           population_radius=0.5,
                           p_contamination=0.2,
                           normal_scale=0.003,
                           normal_freq=0.9,
                           outlier_scale=0.01,
                           outlier_freq=0.01,
                           return_labels=False):
    thetas = np.linspace(0, 2 * np.pi, num_vertices)
    population_radius = np.ones_like(thetas) * population_radius  # if we want constant radius (for a circle)
    gp_sample_normal = get_base_gp(num_members, thetas, scale=normal_scale, sigma=normal_freq)
    gp_sample_outliers = get_base_gp(num_members, thetas, scale=outlier_scale, sigma=outlier_freq)

    should_contaminate = (np.random.random(num_members) > (1 - p_contamination)).astype(float)
    should_contaminate = should_contaminate.reshape(num_members,-1).repeat(num_vertices, axis=1)

    radii = population_radius + (gp_sample_normal * (1 - should_contaminate)) + (gp_sample_outliers * should_contaminate)

    xs, ys = get_xy_coords(thetas, radii)
    contours = rasterize_coords(xs, ys, grid_size)

    labels = should_contaminate[:, 0].astype(int)

    if return_labels:
        return contours, labels
    else:
        return contours


def dataset_in_shape_outliers(num_members,
                              grid_size,
                              num_vertices=100,
                              return_labels=False):
    return dataset_shape_outliers(num_members,
                                  grid_size,
                                  num_vertices,
                                  0.5,
                                  p_contamination=0.1,
                                  normal_scale=0.003,
                                  normal_freq=0.9,
                                  outlier_scale=0.003,
                                  outlier_freq=0.01,
                                  return_labels=return_labels)


def dataset_out_shape_outliers(num_members,
                              grid_size,
                              num_vertices=100,
                              return_labels=False):
    return dataset_shape_outliers(num_members,
                                  grid_size,
                                  num_vertices,
                                  0.5,
                                  p_contamination=0.1,
                                  normal_scale=0.003,
                                  normal_freq=0.9,
                                  outlier_scale=0.009,
                                  outlier_freq=0.04,
                                  return_labels=return_labels)


def dataset_topological_outliers(num_members,
                           grid_size,
                           num_vertices=100,
                           population_radius=0.5,
                           p_contamination=0.1,
                           normal_scale=0.003,
                           normal_freq=0.9,
                           return_labels=False):
    thetas = np.linspace(0, 2 * np.pi, num_vertices)
    population_radius = np.ones_like(thetas) * population_radius  # if we want constant radius (for a circle)
    gp_sample = get_base_gp(num_members, thetas, scale=normal_scale, sigma=normal_freq)
    gp_sample_components = get_base_gp(num_members, thetas, scale=normal_scale, sigma=normal_freq)

    radii = population_radius + gp_sample
    radii_components = population_radius + gp_sample_components

    xs, ys = get_xy_coords(thetas, radii)

    # Generation of disconnected components and holes
    random_scaling = np.random.randint(5, 10, num_members).reshape(-1, 1)
    scaled_radii_out = radii_components / random_scaling
    xs_out, ys_out = get_xy_coords(thetas, scaled_radii_out)

    random_pos = np.random.choice(np.arange(xs.shape[1]), xs.shape[0],
                                  replace=True)  # we get a point along the boundary per curve
    random_xs = np.array([xs[i, rp] for i, rp in enumerate(random_pos)]).reshape(-1, 1)
    random_ys = np.array([ys[i, rp] for i, rp in enumerate(random_pos)]).reshape(-1, 1)

    magnitude = np.sqrt(np.square(random_xs) + np.square(random_ys))
    angle = np.array([thetas[rp] for i, rp in enumerate(random_pos)]).reshape(-1, 1)

    topo_feat_sign = np.random.random(num_members)
    topo_feat_sign[topo_feat_sign > 0.5] = 1
    topo_feat_sign[topo_feat_sign <= 0.5] = -1
    topo_feat_sign = topo_feat_sign.reshape(-1, 1)
    # xs_out = xs_out - grid_size // 2
    # ys_out = ys_out - grid_size // 2
    max_xs_out = np.abs(xs_out).max(axis=1).reshape(-1, 1)
    max_ys_out = np.abs(ys_out).max(axis=1).reshape(-1, 1)
    random_margin = np.random.randint(5, 10, num_members).reshape(-1, 1)/100 # between 0.05 and 0.1
    xs_out = xs_out + (magnitude + topo_feat_sign * (max_xs_out + random_margin)) * np.cos(
        angle)
    ys_out = ys_out + (magnitude + topo_feat_sign * (max_ys_out + random_margin)) * np.sin(
        angle)
    # if its a hole then radius + my radius + buffer ... opposite if itsa disconnected component

    contours = rasterize_coords(xs, ys, grid_size)
    contours_out = rasterize_coords(xs_out, ys_out, grid_size)

    should_contaminate = (np.random.random(num_members) > (1 - p_contamination)).astype(float)
    should_contaminate = should_contaminate.reshape(num_members, -1).repeat(num_vertices, axis=1)
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    num_members = 100

    # masks, labels = dataset_magnitude_outliers(num_members, 500,
    #                                            population_radius=0.5,
    #                                            case=1, return_labels=True)

    # masks, labels = dataset_no_outliers(num_members, 500, return_labels=True)
    # masks, labels = dataset_sym_mag_outliers(num_members, 500, return_labels=True)
    masks, labels = dataset_peaks_mag_outliers(num_members, 500, return_labels=True)
    # masks, labels = dataset_out_shape_outliers(num_members, 500, return_labels=True)
    # masks, labels = dataset_topological_outliers(num_members, 500,
    #                        p_contamination=0.1,
    #                        return_labels=True)
    population_mean = get_population_mean(0.5, 500)

    from skimage.measure import find_contours

    for lab, mask in zip(labels, masks):
        contours = find_contours(mask)
        for c in contours:
            if lab == 0:
                plt.plot(c[:, 1], c[:, 0], c="blue", alpha=0.1)
            else:
                plt.plot(c[:, 1], c[:, 0], c="orange", alpha=1)
    contours = find_contours(population_mean)
    for c in contours:
        plt.plot(c[:, 1], c[:, 0], c="red", linewidth=3)
    plt.xlim(0, 500)
    plt.ylim(0, 500)
    plt.show()
