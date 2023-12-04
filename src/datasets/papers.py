"""
Synthetic datasets used in contour visualization paper.
"""

import numpy as np
from backend.src.datasets.ellipse_generation import generate_ellipse_sdf, get_ensemble_params


def ensemble_ellipses_cbp(num_members: int, num_rows: int, num_cols: int):
    """
    Generates an ensemble of ellipses with the properties in the cbp paper.
    This entails varying the ellipses' size, ellipticity and rotation and adding
    noise.
    """
    memberships = [0 for i in range(num_members)]
    ellipticities = np.random.normal(0.5, 0.1, num_members).astype(float)
    iso_values = np.random.normal(0.7, 0.02, num_members).astype(float)
    translations_x = np.zeros(num_members)
    translations_y = np.zeros(num_members)
    rotations_deg = np.random.normal(45, 5, num_members).astype(float)
    noise_params = {
        "l0.freq": np.random.randint(2, 12, num_members).astype(int),
        "l0.weight": np.ones(num_members) * 0.05
    }

    ensemble_params = get_ensemble_params(num_members,
                                          ellipticities,
                                          iso_values,
                                          translations_x,
                                          translations_y,
                                          rotations_deg,
                                          noise_params,
                                          metadata=dict(memberships=memberships))

    ensemble = []
    for m in range(num_members):
        params_dict = ensemble_params[m]
        data = generate_ellipse_sdf(num_cols, num_rows, **params_dict)
        data[data >= 0] = 1  # binarize
        data[data < 0] = 0  # binarize
        ensemble.append(data)

    return ensemble


def ensemble_venn_diagram(num_members: int,
                          num_rows: int,
                          num_cols: int,
                          offset: float = 0.3,
                          translation_x_std: tuple = (0.03, 0.03, 0.03),
                          translation_y_std: tuple = (0.03, 0.03, 0.03),
                          radii: tuple = (0.5, 0.5, 0.5),
                          radii_std: tuple = (0.02, 0.04, 0.02),
                          sampling_rates: tuple = (1 / 3, 1 / 3, 1 / 3)):
    """
    Generates an ensemble of circles of varying sizes clustered in 3 regions
    in space (like a venn diagram with three circles)
    For each group of circles it is possible to specify:
      - Their offset from the center
      - Their radii
      - The std of the radii
      - Their sampling rates
    """
    t1 = (0.0, offset)  # middle
    t2 = (offset * np.cos(np.deg2rad(330)), offset * np.sin(np.deg2rad(330)))  # right
    t3 = (offset * np.cos(np.deg2rad(210)), offset * np.sin(np.deg2rad(210)))  # left
    ts = [t1, t2, t3]

    iv_dist_params = [(radii[0], radii_std[0]), (radii[1], radii_std[1]), (radii[2], radii_std[2])]

    probs = np.array(sampling_rates)
    probs = probs / probs.sum()
    memberships = [i for i in np.random.choice(np.arange(3), num_members, replace=True, p=probs)]

    ellipticities = np.zeros(num_members).astype(float)
    iso_values = [np.random.normal(*iv_dist_params[i]) for i in memberships]
    translations = [(ts[i], translation_x_std[i], translation_y_std[i]) for i in memberships]
    translations_x = [a[0][0] + np.random.normal(0.0, a[1], 1)[0] for a in translations]
    translations_y = [a[0][1] + np.random.normal(0.0, a[2], 1)[0] for a in translations]
    rotations_deg = np.zeros(num_members).astype(float)

    noise_params = {
        "l0.freq": np.zeros(num_members),
        "l0.weight": np.zeros(num_members)
    }

    ensemble_params = get_ensemble_params(num_members,
                                          ellipticities,
                                          iso_values,
                                          translations_x,
                                          translations_y,
                                          rotations_deg,
                                          noise_params,
                                          metadata=dict(memberships=memberships))

    ensemble = []
    for m in range(num_members):
        params_dict = ensemble_params[m]
        data = generate_ellipse_sdf(num_cols, num_rows, **params_dict)
        data[data >= 0] = 1  # binarize
        data[data < 0] = 0  # binarize
        ensemble.append(data)

    return ensemble


if __name__ == "__main__":
    print("Test run")
    import matplotlib.pyplot as plt

    from backend.src.vis_utils import plot_contour_spaghetti, plot_grid_masks

    colors = ["#1b9e77", "#d95f02", "#7570b3"]

    num_members = 100
    num_cols = num_rows = 300

    ensemble_ellipses = ensemble_venn_diagram(num_members, num_cols, num_rows)

    plot_contour_spaghetti(ensemble_ellipses)
    plot_grid_masks(ensemble_ellipses)
