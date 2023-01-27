"""
Utilities for generating ensembles of ellipses.
"""

import math
import numpy as np
from skimage.measure import find_contours

import matplotlib.pyplot as plt


def generate_ellipse_sdf(num_cols: int, num_rows: int,
                         ellipticity: float = 0.0,
                         isovalue: float = 0.5,
                         translation_x: float = 0.0,
                         translation_y: float = 0.0,
                         rotation_deg: int = 0,
                         noise_params: dict = None, **kwargs) -> np.ndarray:
    """
    Generates the signed distance field of an ellipse with the specified parameters.
    We assume that the x-axis goes from left (negative) to right (positive).
    We assume that the y-axis foes from bottom (negative) to top (positive).
    We use array like conventions:
      - SDF[0, 0] corresponds to the top left (x: -1, y: 1) corner of the field.
      - SDF[num_cols-1, num_rows-1] corresponds to the bottom right (x: 1, y: -1) corner of the field.
    :param num_cols:
    :param num_rows:
    :param ellipticity:
    :param isovalue:
    :param translation:
    :param rotation_deg:
    :param noise_params:
    :return:
    """

    rotation_rad = np.deg2rad(rotation_deg)

    a = 1
    b = -((ellipticity ** 2) * (a ** 2) - (a ** 2))

    # Mesh creation

    # - Create domain coordinates X \in [-1,1] and Y \in [-1, 1]
    X = np.repeat(np.linspace(-1, 1, num_cols).reshape(1, -1), num_rows, axis=0)
    Y = np.repeat(np.linspace(1, -1, num_rows).reshape(-1, 1), num_cols, axis=1)

    # - Rotate domain by rotation_deg degrees
    X1 = X*math.cos(rotation_rad) - Y*math.sin(rotation_rad)
    Y1 = X*math.sin(rotation_rad) + Y*math.cos(rotation_rad)

    # - Translate the domain using the translation vector
    X1 = X1 - translation_x  # Subtracting the translation vector has the opposite visual effect
    Y1 = Y1 - translation_y  # Subtracting the translation vector has the opposite visual effect

    # - Scale domain (from circle to ellipse)
    X1 = X1 / a
    Y1 = Y1 / b

    # Distance field calculation

    dist = np.sqrt(np.square(X1) + np.square(Y1))
    signed_dist = isovalue - dist

    # TODO: theta-correlated noise operations
    theta = np.arctan2(Y1, X1)
    if noise_params is None:
        noise_params = {"l0.freq": 1, "l0.weight": 1}
    noisy_dist = signed_dist + noise_params["l0.weight"]*np.sin(noise_params["l0.freq"]*theta)

    return noisy_dist


def get_ensemble_params(num_members, ellipticities=None, isovalues=None, translations_x=None, translations_y=None, rotations_deg=None, noise_params=None, metadata=None):
    """
    Transforms individual lists of params into a list of dicts of params that can be passed
    to `generate_ellipse_sdf` as **kwargs.
    If the value of the parameter is None, a default is used.
    If the value of the parameter is an element, the element is tiled to form an array of size num_members.
    """
    def check_numeric_param(param):
        if param is None:
            param = 0.0
        if type(param) is not list and type(param) is not np.ndarray:
            param_type = type(param)
            param = np.repeat(param, num_members)
        return param

    ellipticities = check_numeric_param(ellipticities)
    isovalues = check_numeric_param(isovalues)
    translations_x = check_numeric_param(translations_x)
    translations_y = check_numeric_param(translations_y)
    rotations_deg = check_numeric_param(rotations_deg)

    if noise_params is None:
        noise_params = {"l0.freq": 1, "l0.weight": 0}
    if type(noise_params) is dict:
        noise_params_list = [dict() for i in range(num_members)]
        for k in noise_params.keys():
            for i, v in enumerate(noise_params[k]):
                noise_params_list[i][k] = v
        noise_params = noise_params_list

    if metadata is None:
        metadata = dict()
    if type(metadata) is dict:
        metadata_list = [dict() for i in range(num_members)]
        for k in metadata.keys():
            for i, v in enumerate(metadata[k]):
                metadata_list[i][k] = v
        metadata = metadata_list

    params = []
    for member_id in range(num_members):
        params.append(dict(
            ellipticity=ellipticities[member_id],
            isovalue=isovalues[member_id],
            translation_x=translations_x[member_id],
            translation_y=translations_y[member_id],
            rotation_deg=rotations_deg[member_id],
            noise_params=noise_params[member_id],
            metadata=metadata[member_id]
        ))
    return params


def ensemble_ellipses_cbp(num_members: int):
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

    return get_ensemble_params(num_members, ellipticities, iso_values, translations_x, translations_y, rotations_deg,
                               noise_params, metadata=dict(memberships=memberships))


def ensemble_venn_diagram(num_members: int,
                          offset: float = 0.3,
                          translation_x_std: tuple = (0.0, 0.0, 0.0),
                          translation_y_std: tuple = (0.0, 0.0, 0.0),
                          radii: tuple = (0.5, 0.5, 0.5),
                          radii_std: tuple = (0.02, 0.02, 0.02),
                          sampling_rates: tuple = (1/3, 1/3, 1/3)):
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

    return get_ensemble_params(num_members, ellipticities, iso_values, translations_x, translations_y, rotations_deg, noise_params, metadata=dict(memberships=memberships))


def load_ensemble_ellipses(num_members, num_cols, num_rows, params_set=0, cache=True, random_state=42):
    """
    Generates or loads (if cached) an ensemble of ellipses associated with a specific parameter set.
    Available ids for the params_set are 0, 1 and 2.
    """
    from pathlib import Path
    import pickle
    path = Path("/Users/chadepl/git/multimodal-contour-vis/backend")  # absolute path in server
    path = path.joinpath("data", "ellipses", f"{num_members}-{num_cols}-{num_rows}-{params_set}-{random_state}.pkl")
    print(Path(__name__).absolute())

    if path.exists() and cache:
        with open(path, "rb") as f:
            ensemble_data_object = pickle.load(f)
    else:
        # TODO: set random state
        ensemble_data_object = {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "fields": None,
            "ensemble": {
                "type": "grid",  # we could also send contours
                "members": [
                    #  list with all ensemble members as list of objects {data: , ...}
                ]
            }
        }

        if params_set == 0:
            desc = "CBP ellipses. Ellipses with different size, rotation and ellipticity."
            ensemble_params = ensemble_ellipses_cbp(num_members=num_members)

        if params_set == 1:
            desc = "EnConVis circles. Circles with different radii located in three locations with small positional perturbations."
            ensemble_params = ensemble_venn_diagram(num_members=num_members,
                                                    translation_x_std=(0.03, 0.03, 0.03),
                                                    translation_y_std=(0.03, 0.03, 0.03),
                                                    radii_std=(0.02, 0.04, 0.02),
                                                    sampling_rates=(10, 2, 1))
        else:
            pass


        ensemble = []
        for m in range(num_members):
            params_dict = ensemble_params[m]
            member = {
                "data": generate_ellipse_sdf(num_cols, num_rows, **params_dict),
                "features": params_dict,
            }
            member["data"][member["data"] >= 0] = 1  # binarize
            member["data"][member["data"] < 0] = 0  # binarize
            member["data"] = member["data"].astype(int)
            ensemble.append(member)

        ensemble_data_object["ensemble"]["desc"] = desc
        ensemble_data_object["ensemble"]["members"] = ensemble

        if cache:
            with open(path, "wb") as f:
                pickle.dump(ensemble_data_object, f)

    return ensemble_data_object


def plot_ensemble_ellipses_overlay(num_members, contours, features, alpha=0.1):
    colors = ["#1b9e77", "#d95f02", "#7570b3"]
    fig, axs = plt.subplots(figsize=(6, 6), layout="tight")
    for m in range(num_members):
        if "memberships" in features[0]["metadata"]:
            color = colors[features[m]["metadata"]["memberships"]]
        else:
            color = colors[0]
        axs.plot(contours[m][:, 1], contours[m][:, 0], c=color, alpha=alpha)
    axs.set_axis_off()
    plt.show()


if __name__ == "__main__":
    print("Test run")
    import matplotlib.pyplot as plt

    colors = ["#1b9e77", "#d95f02", "#7570b3"]

    num_members = 100
    num_cols = num_rows = 1020

    ensemble_ellipses = load_ensemble_ellipses(num_members, num_cols, num_rows, params_set=0, cache=False)

    members_data = [m["data"] for m in ensemble_ellipses["ensemble"]["members"]]
    members_feat = [m["features"] for m in ensemble_ellipses["ensemble"]["members"]]
    members_cont = [find_contours(md, 0.5)[0] for md in members_data]



    fig, axs = plt.subplots(ncols=5, nrows=5, figsize=(6, 6), layout="tight")
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        ax.imshow(members_data[i], cmap="gray")
        ax.set_axis_off()
    plt.show()


