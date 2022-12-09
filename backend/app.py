from time import time

from flask import Flask
import flask
from flask import request
import json
from flask_cors import CORS

from src import ellipse_generation, contour_boxplot

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