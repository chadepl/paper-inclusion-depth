from time import time
import numpy as np

from flask import Flask
import flask
from flask import request
import json
from flask_cors import CORS

from src import contour_boxplot
from backend.src.datasets import ellipse_generation

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello():
    return "Hello. world"


@app.route("/users", methods=["GET"])
def users():
    print("users endpoint reached")
    with open("data/users.json", "r") as f:
        data = json.load(f)
        data.append({
            "username": "user4",
            "pets": ["hamster"]
        })
        return flask.jsonify(data)


@app.route("/available_datasets", methods=["GET"])
def get_available_datasets():
    """
    Provides endpoints of ensemble datasets.
    :return:
    """
    response = {
        "available_datasets": [
            dict(name="circles_1", type="synthetic", endpoint="/ellipses_dataset", kwargs={"params_set": 0}),
            dict(name="circles_2", type="synthetic", endpoint="/ellipses_dataset", kwargs={"params_set": 1}),
            dict(name="ellipses", type="synthetic", endpoint="/ellipses_dataset", kwargs={"params_set": 2}),
            dict(name="parotids", type="real", endpoint="/miccai_han_slices_dataset", kwargs={}),
            dict(name="weather", type="real", endpoint="/ellipses_dataset", kwargs={})
        ]
    }
    return flask.jsonify(response)


@app.route("/ellipses_dataset", methods=["POST"])
def get_ellipses_dataset():
    print("[backend] hit endpoint: ellipses_dataset")

    content = request.json
    content = content if content is not None else dict()

    num_cols = int(content.get("num_cols", 300))
    num_rows = int(content.get("num_rows", 300))
    num_members = int(content.get("num_members", 20))
    kwargs = content.get("kwargs", {"params_set": 0})
    if "params_set" in kwargs:
        kwargs["params_set"] = int(kwargs["params_set"])
    else:
        kwargs["params_set"] = 0
    kwargs["random_state"] = 42

    t_start = time()
    print("[backend] Generating ensemble data")
    print(f"[backend] - Grid size: ({num_cols}, {num_rows})")
    print(f"[backend] - Num_members: {num_members}")

    ensemble = ellipse_generation.load_ensemble_ellipses(num_members, num_cols, num_rows, **kwargs)

    t_stop = time()
    print(f"[backend] - Computation time: {t_stop - t_start} seconds")

    for m in ensemble["ensemble"]["members"]:
        m["data"] = m["data"].flatten().tolist()

    return flask.jsonify(ensemble)


@app.route("/ensemble_depths", methods=["POST"])
def get_ensemble_depths():
    print("[backend] hit endpoint: ensemble_depths")

    content = request.json
    content = content if content is not None else dict()

    w = int(content.get("num_cols", 300))
    h = int(content.get("num_rows", 300))
    ensemble = content.get("ensemble", [])
    subset_combinations = content.get("subset_combinations", [2, ])

    ensemble = [np.array(m).reshape((h, w)) for m in ensemble]

    depth_data = contour_boxplot.get_depth_data(ensemble, subset_combinations, True)

    return flask.jsonify(dict(depth_data=depth_data))


@app.route("/member_subset_band", methods=["POST"])
def get_member_subset_band():
    print("[backend] hit endpoint: member_subset_band")

    content = request.json
    content = content if content is not None else dict()


@app.route("/member_path_representation", methods=["POST"])
def get_member_path_representation():
    print("[backend] hit endpoint: member_path_representation")

    content = request.json
    content = content if content is not None else dict()

    from skimage.measure import find_contours

    w = int(content.get("num_cols", 300))
    h = int(content.get("num_rows", 300))
    isovalue = float(content.get("isovalue", 0.5))
    arr = content.get("array", [])
    arr = np.array(arr).reshape((h, w))
    path = find_contours(arr, isovalue)[0]

    return flask.jsonify(dict(path=path.tolist()))


@app.route("/ensemble_data", methods=["GET"])
def ensemble_data():
    w = h = int(request.args.get("size", 300))
    num_members = int(request.args.get("num_members", 20))

    t_start = time()
    print("[backend] Generating ensemble data")
    print(f"[backend] - Grid size: ({w}, {h})")
    print(f"[backend] - Num_members: {num_members}")

    ensemble = ellipse_generation.generate_ensemble_ellipses(num_members, w, h)

    depth_data = contour_boxplot.get_depth_data([m["data"] for m in ensemble["ensemble"]["members"]], (2,), True)
    ensemble["contour_boxplot"] = depth_data

    t_stop = time()
    print(f"[backend] - Computation time: {t_stop - t_start} seconds")

    for m in ensemble["ensemble"]["members"]:
        m["data"] = m["data"].flatten().tolist()

    return flask.jsonify(ensemble)


@app.route("/contour_band", methods=["GET", "POST"])
def get_contour_band():
    print("Get band")
    return flask.jsonify({"a": 1});


if __name__ == "__main__":
    app.run("localhost", 6969)
