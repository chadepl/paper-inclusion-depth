
import math
import numpy as np
from skimage.measure import find_contours


def generate_ellipse_sdf(w: int, h: int,
                         ellipticity: float = 0.0,
                         isovalue: float = 0.5,
                         rotation_deg: int = 0,
                         noise_params: dict = None) -> np.ndarray:
    

    num_cols = w
    num_rows = h
    rotation_rad = np.deg2rad(rotation_deg)

    a = 1
    b = -((ellipticity ** 2) * (a ** 2) - (a ** 2))

    # Create domain coordinates X \in [-1,1] and Y \in [-1, 1]
    X = np.repeat(np.linspace(-1, 1, num_cols).reshape(1, -1), num_rows, axis=0)
    Y = np.repeat(np.linspace(1, -1, num_rows).reshape(-1, 1), num_cols, axis=1)

    # Rotate domain by rotation_deg degrees
    X1 = X*math.cos(rotation_rad) - Y*math.sin(rotation_rad)
    Y1 = X*math.sin(rotation_rad) + Y*math.cos(rotation_rad)

    # Scale domain (from circle to ellipse)
    X1 = X1 / a
    Y1 = Y1 / b

    dist = np.sqrt(np.square(X1) + np.square(Y1))
    signed_dist = isovalue - dist

    # TODO: theta-correlated noise operations
    theta = np.arctan2(Y1, X1)
    if noise_params is None:
        noise_params = {"l0.freq": 1, "l0.weight": 1}
    noisy_dist = signed_dist + noise_params["l0.weight"]*np.sin(noise_params["l0.freq"]*theta)

    return noisy_dist


def generate_ensemble_ellipses(num_members, w, h):

    ensemble_data_object = {
        "num_rows": h,
        "num_cols": w,
        "fields": None,
        "ensemble": {
            "type": "grid",  # we could also send contours
            "members": [
                #  list with all ensemble members as list of objects {data: , ...}
            ]
        }
    }

    params = []
    for m in range(num_members):
        params.append({
            "ellipticity": float(np.random.normal(0.5, 0.1, 1)[0]),
            "isovalue": float(np.random.normal(0.7, 0.05, 1)[0]),
            "rotation_deg": int(np.random.randint(40, 50, 1)[0]),
            "noise_params": {
                "l0.freq": int(np.random.randint(2, 12, 1)[0]),
                "l0.weight": float(0.1)
            }
        })

    ensemble = []
    for m in range(num_members):
        params_dict = params[m]
        member = {
            "data": generate_ellipse_sdf(w, h, **params_dict),
            "features": params_dict,
        }
        member["data"][member["data"] >= 0] = 1  # binarize
        member["data"][member["data"] < 0] = 0  # binarize
        member["data"] = member["data"].astype(int)
        ensemble.append(member)

    ensemble_data_object["ensemble"]["members"] = ensemble

    return ensemble_data_object


if __name__ == "__main__":
    print("Test run")
    import matplotlib.pyplot as plt

    num_members = 100
    w = h = 300
    ensemble_ellipses = generate_ensemble_ellipses(num_members, w, h)

    img_arr = np.zeros_like(ensemble_ellipses["ensemble"]["members"][0]["data"], dtype=float)
    contours = []
    for m in range(num_members):
        ellipse_sdf = ensemble_ellipses["ensemble"]["members"][m]["data"]
        img_arr += (ellipse_sdf * (1/num_members))
        contours.append(find_contours(ellipse_sdf, 0.5)[0])
        #ellipse_contours = find_contours(ellipse_sdf, 0.5)[0]

        #plt.plot(ellipse_contours[:, 1], ellipse_contours[:, 0], c="yellow")

    plt.imshow(img_arr, cmap="viridis")
    plt.show()

    for m in range(num_members):
        plt.plot(contours[m][:, 1], contours[m][:, 0], c="red")
    plt.show()

