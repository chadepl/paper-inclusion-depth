"""
Utilities for generating ensembles of streamlines.
"""

import numpy as np


def generate_streamline_random_walker(num_cols: int, num_rows: int,
                                      start_pos: tuple=(-1.0, -1.0),
                                      step_size: float=0.01,
                                      probs_x: tuple=(0.0, 0.5, 0.5),
                                      probs_y: tuple=(0.0, 0.5, 0.5),
                                      return_raster: bool=False,
                                      **kwargs):
    """
    Generates single sources streamlines using random walks.
    We first generate the lines and then rasterize them.
    """
    positions = [start_pos, ]
    step = ((-step_size, 0, step_size), (-step_size, 0, step_size))
    probs = (probs_x, probs_y)
    contained = True
    while contained:
        step_x = np.random.choice(step[0], 1, p=probs[0])[0]
        step_y = np.random.choice(step[1], 1, p=probs[1])[0]
        pos_x = positions[-1][0] + step_x
        pos_y = positions[-1][1] + step_y
        positions.append((pos_x, pos_y))
        contained = -1 <= pos_x <= 1 and -1 <= pos_y <= 1
    positions = np.array(positions)
    positions /= 2
    positions += 0.5
    positions[:, 0] *= num_cols - 1
    positions[:, 1] *= num_rows - 1

    if return_raster:
        return get_raster_from_path_coords(num_cols, num_rows, positions)

    return positions


def get_raster_from_path_coords(num_cols: int, num_rows: int, coords: np.ndarray):
    from skimage.draw import line_aa
    raster = np.zeros((num_rows, num_cols))
    for row_num in np.arange(coords.shape[0] - 1):
        start_pos = np.floor(coords[row_num, :]).astype(int)
        end_pos = np.floor(coords[row_num + 1, :]).astype(int)
        rr, cc, _ = line_aa(start_pos[1], start_pos[0], end_pos[1], end_pos[0])
        raster[rr, cc] = 1.0
    return raster


def ensemble_streamlines_rw(num_members: int):
    params = []
    param_sets = [
        dict(probs_x=(0.0, 0.8, 0.2), probs_y=(0.05, 0.95, 0.0), start_pos=(-1, 0.1)),
        dict(probs_x=(0.0, 0.9, 0.1), probs_y=(0.05, 0.9, 0.05), start_pos=(-1, -0.1))
    ]
    for i in range(num_members):
        ps_id = np.random.choice(np.arange(len(param_sets)), 1, p=(0.5, 0.5))[0]
        ps = param_sets[ps_id]
        ps["metadata"] = dict(memberships=ps_id)
        params.append(ps)
    return params


def load_ensemble_streamlines(num_members, num_cols, num_rows, params_set=0, cache=True, random_state=42):

    from pathlib import Path
    import pickle
    path = Path("/Users/chadepl/git/multimodal-contour-vis/backend")  # absolute path in server
    path = path.joinpath("data", "streamlines", f"{num_members}-{num_cols}-{num_rows}-{params_set}-{random_state}.pkl")
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
        ensemble_params = ensemble_streamlines_rw(num_members=num_members)

        ensemble = []
        for m in range(num_members):
            params_dict = ensemble_params[m]
            member = {
                "data_path": generate_streamline_random_walker(num_cols, num_rows, **params_dict),
                "features": params_dict,
            }
            member["data"] = get_raster_from_path_coords(num_cols, num_rows, member["data_path"])
            ensemble.append(member)
    else:
        pass

    ensemble_data_object["ensemble"]["desc"] = desc
    ensemble_data_object["ensemble"]["members"] = ensemble

    if cache:
        with open(path, "wb") as f:
            pickle.dump(ensemble_data_object, f)

    return ensemble_data_object

# TODO: Generate vector field
# https://stackoverflow.com/questions/35437010/including-brownian-motion-into-particle-trajectory-integration-using-scipy-integ
# https://aip.scitation.org/doi/10.1063/1.4968528

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    colors = ["#1b9e77", "#d95f02", "#7570b3"]

    num_members = 100
    num_cols = num_rows = 300
    ensemble_streamlines = load_ensemble_streamlines(num_members, num_cols, num_rows, cache=False)
    streamlines_paths = [m["data_path"] for m in ensemble_streamlines["ensemble"]["members"]]
    streamlines_feat = [m["features"] for m in ensemble_streamlines["ensemble"]["members"]]
    streamlines_rasters = [m["data"] for m in ensemble_streamlines["ensemble"]["members"]]

    fig, ax = plt.subplots(figsize=(6, 6), layout="tight")
    for member_id, streamline in enumerate(streamlines_paths):
        ax.plot(streamline[:, 0], streamline[:, 1], c=colors[streamlines_feat[member_id]["metadata"]["memberships"]], alpha=0.1)
    ax.set_xlim([0, num_cols])
    ax.set_ylim([0, num_rows])
    ax.set_axis_off()
    plt.show()

    fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(6, 6), layout="tight")
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        ax.imshow(streamlines_rasters[i], cmap="gray")
        ax.set_axis_off()
    plt.show()