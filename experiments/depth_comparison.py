"""
Depths can be obtained in different ways
For instance, there are metric and functional depths
In this experiment we compare the performance of three depths,
for estimating the centrality of members of an ensemble

We focus on the following depths:
 - l1 (metric, sdf-based)
 - band depth (functional, based on set operations on binary masks)
 - sdf sample depth (functional)

We consider different arrangements of synthetic datasets aimed to
stress each type of depths. The goal of this sub-experiment is to
determine in which datasets one depth might be better than others.

In addition to visual comparisons, we also compare the centrality scores
and computation times for different sample sizes.

Some ideas as to what to vary in the different ensembles of shapes:
- Shape globally
- Shape locally
- Reflections
- Rotation
- Scale
- Position
"""

from time import time

import numpy as np
import pandas as pd
from numpy.linalg import norm
from skimage.draw import disk, ellipse
from skimage.measure import find_contours

import matplotlib.pyplot as plt
import seaborn as sns

from backend.src.utils import get_distance_transform
from backend.src.contour_band_depth import get_depth_matrix, get_contour_band_depths
from backend.src.vis_utils import plot_contour_spaghetti

from backend.src.datasets.simple_shapes_datasets import affine_transforms_dataset
from backend.src.contour_depths import band_depth, lp_depth, sdf_depth


def plt_ensemble_depths(ensemble_members, depths, title=None, ax=None):
    color_map = plt.cm.get_cmap("Purples")
    depths_cols = ((1 - (depths / depths.max())) * 255).astype(int)
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(np.zeros_like(ensemble_members[0]), cmap="gray")
    for i, member in enumerate(ensemble_members):
        for contour in find_contours(member, 0.5):
            ax.plot(contour[:, 1], contour[:, 0], c=color_map(depths_cols[i]))
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax


transform_type = ["scale", "position", "rotation"][2]
num_members = 50
num_cols = num_rows = 300

ensemble_members = affine_transforms_dataset(num_cols, num_rows, num_members, transform_type=transform_type)

plt.imshow(np.zeros_like(ensemble_members[0]), cmap="gray")
colors = plt.cm.rainbow(np.linspace(0, 1, num_members))
for i, (member, color) in enumerate(zip(ensemble_members, colors)):
    for contour in find_contours(member, 0.5):
        plt.plot(contour[:, 1], contour[:, 0])  # , c=color)
plt.title(f"Dataset ({transform_type})")
plt.axis("off")
plt.show()

# Depth comparison

depth_fn = [
    dict(name="Band depth", fn=band_depth.compute_depths, kwargs=dict(target_mean_depth=1 / 6)),
    dict(name="L1 depth", fn=lp_depth.compute_depths, kwargs=dict(pca_comp=2)),
    dict(name="SDF samples \n (Hausdorff)", fn=sdf_depth.compute_hausdorff_depths, kwargs=dict(d_type="hausdorff")),
    dict(name="SDF samples \n (L2-norm)", fn=sdf_depth.compute_hausdorff_depths, kwargs=dict(d_type="l2")),
    dict(name="SDF samples \n (Band)", fn=sdf_depth.compute_hausdorff_depths, kwargs=dict(d_type="band"))
]

# depths = dict()
# for dfn in depth_fn:
#     t_start = time()
#     d = dfn["fn"](ensemble_members, **dfn["kwargs"])
#     t_end = time()
#     depths[dfn["name"]] = d
#     print(f"{dfn['name']} took {t_end - t_start} seconds to compute")
#
# ncols = np.minimum(3, len(depth_fn))
# nrows = np.ceil(len(depth_fn) / ncols).astype(int)
# fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols*4, nrows*4))
# axs = axs.flatten()
# for i, dfn in enumerate(depth_fn):
#     plt_ensemble_depths(ensemble_members, depths[dfn["name"]], ax=axs[i], title=dfn["name"])
# plt.show()


# Timings
sizes = np.linspace(10, 100, 10)
timings_data = dict()
bd_times = []
ld_times = []
bds = []
lds = []
for n in sizes:
    print(f"Processing size {int(n)}")
    ensemble_members = affine_transforms_dataset(num_cols, num_rows, int(n), transform_type=transform_type)
    timings_data[n] = dict()
    for dfn in depth_fn:
        t_start = time()
        d = dfn["fn"](ensemble_members, **dfn["kwargs"])
        t_end = time()
        timings_data[n][dfn["name"]] = dict(depths=d, time=t_end - t_start)

ncols = np.minimum(3, len(depth_fn))
nrows = np.ceil(len(depth_fn) / ncols).astype(int)
fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 4, nrows * 4))
axs = axs.flatten()
for i, dfn in enumerate(depth_fn):
    plt_ensemble_depths(ensemble_members, timings_data[sizes[-1]][dfn["name"]]["depths"], ax=axs[i], title=dfn["name"])
plt.show()

sample_size = sizes[-1]
tmp = pd.DataFrame({k: v["depths"] for k, v in timings_data[sample_size].items()})
tmp = tmp.iloc[:, [0, 1, 2]]
sns.pairplot(tmp)
plt.show()

df_data = []
for ss in sizes:
    for k, v in timings_data[ss].items():
        if k in [dfn["name"] for dfn in depth_fn[:3]]:
            df_data.append(dict(size=ss, depth_fn=k, time=v["time"]))
tmp = pd.DataFrame(df_data)

fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
sns.lineplot(tmp, x="size", y="time", hue="depth_fn")
ax.set_xlabel("Ensemble size")
ax.set_ylabel("Elapsed time (seconds)")
plt.show()

# fig, ax = plt.subplots()
# ax.plot(sizes, bd_times, label="band depths")
# ax.plot(sizes, ld_times, label="l1 depths")
# ax.set_xlabel("Ensemble size")
# ax.set_ylabel("Time (seconds)")
# ax.set_title("Sample size vs time for different depth functions")
# ax.legend()
# plt.show()
