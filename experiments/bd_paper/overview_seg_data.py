"""
Code used to generate teaser image of paper
"""
from time import time
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from backend.src.vis_utils import plot_contour_spaghetti, plot_contour_boxplot
from backend.src.datasets import han_ensembles
from backend.src.datasets.bd_paper import get_han_dataset_ParotidR, get_han_dataset_BrainStem
from backend.src.contour_depths import border_depth, band_depth

structure_name = ["Parotid_R", "BrainStem"][0]

# img, gt, ensemble = han_ensembles.get_han_slice_ensemble(540, 540)
if structure_name == "Parotid_R":
    img, gt, ensemble = get_han_dataset_ParotidR(540, 540)
elif structure_name == "BrainStem":
    img, gt, ensemble = get_han_dataset_BrainStem(540, 540)

# gt = gt
# plot_contour_spaghetti([gt, ], under_mask=img)
# plt.show()

# spaghetti plot
labs = np.arange(len(ensemble))
np.random.shuffle(labs)
fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
plot_contour_spaghetti(ensemble, under_mask=img, resolution=(540, 540), ax=ax)
plt.show()
fig.savefig(f"/Users/chadepl/Downloads/han-spag-{structure_name}.png")

# heatmap
ensemble_std = np.concatenate([np.expand_dims(e, axis=0) for e in ensemble], axis=0)
ensemble_std = np.std(ensemble_std, axis=0)
fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
ax.imshow(img, cmap="gray")
ax.imshow(ensemble_std, cmap="magma", alpha=(ensemble_std > 0).astype(float))
ax.set_axis_off()
plt.show()
fig.savefig(f"/Users/chadepl/Downloads/han-std-{structure_name}.png")

# contour depths plot
res_path = Path("results_data/han-depths")
if not res_path.exists():
    res_path.mkdir()
res_path = res_path.joinpath(f"{structure_name}")

if res_path.exists():
    with open(res_path.joinpath("depths_mtbad.pkl"), "rb") as f:
        depths_bad = pickle.load(f)
    with open(res_path.joinpath("depths_bod.pkl"), "rb") as f:
        depths_bod = pickle.load(f)
    with open(res_path.joinpath("depths_wbod.pkl"), "rb") as f:
        depths_wbod = pickle.load(f)
    with open(res_path.joinpath("times.pkl"), "rb") as f:
        times = pickle.load(f)

else:
    print("Results did not exist. Computing ...")

    print("CBD ...")
    times = dict()
    t_start = time()
    depths_bad = band_depth.compute_depths(ensemble, modified=True, target_mean_depth=1 / 6)
    t_end = time()
    times["mtbad"] = t_end - t_start

    print("BoD ...")
    t_start = time()
    depths_bod, p_depths, border_coords = border_depth.compute_depths(ensemble, use_fast=True, global_criteria=None,
                                                                      modified=False, return_point_depths=True,
                                                                      return_border_coords=True)
    t_end = time()
    times["bod"] = t_end - t_start

    print("wBoD ...")
    t_start = time()
    depths_wbod, wbod_p_depths, wbod_border_coords = border_depth.compute_depths(ensemble, global_criteria="nestedness",
                                                                                 modified=True,
                                                                                 return_point_depths=True,
                                                                                 return_border_coords=True)
    t_end = time()
    times["wbod"] = t_end - t_start

    res_path.mkdir()

    with open(res_path.joinpath("depths_mtbad.pkl"), "wb") as f:
        pickle.dump(depths_bad, f)

    with open(res_path.joinpath("depths_bod.pkl"), "wb") as f:
        pickle.dump(depths_bod, f)

    with open(res_path.joinpath("depths_wbod.pkl"), "wb") as f:
        pickle.dump(depths_wbod, f)

    with open(res_path.joinpath("times.pkl"), "wb") as f:
        pickle.dump(times, f)

print(f"CBD: {times['mtbad']}")
print(f"BoD (fast): {times['bod']}")
print(f"BoD  (weighted): {times['wbod']}")

print("Outliers")
print(np.argsort(depths_bad)[:10])
print(np.argsort(depths_bod)[:10])
print(np.argsort(depths_wbod)[:10])

print("Inliers")
print(np.argsort(depths_bad)[::-1][:10])
print(np.argsort(depths_bod)[::-1][:10])
print(np.argsort(depths_wbod)[::-1][:10])

depths_arr = [depths_bad, depths_bod, depths_wbod]
df = pd.DataFrame(depths_arr).T
df.columns = ["CBD", "BoD", "wBoD"]
sns.pairplot(df)
plt.show()
print(df.shape)

overlap_in = np.zeros((3, 3))
overlap_out = np.zeros((3, 3))
for i, ds1 in enumerate(depths_arr):
    for j, ds2 in enumerate(depths_arr):
        in_i = np.argsort(ds1)[::-1][:100].astype(int)
        in_j = np.argsort(ds2)[::-1][:100].astype(int)
        out_i = np.argsort(ds1)[:5].astype(int)
        out_j = np.argsort(ds2)[:5].astype(int)

        overlap_in[i, j] = np.intersect1d(in_i, in_j).size / 100
        overlap_out[i, j] = np.intersect1d(out_i, out_j).size / 5

print(overlap_in)
print()
print(overlap_out)

# plot_contour_spaghetti(ensemble, under_mask=img, arr=depths_bod, is_arr_categorical=False, vmin=0, vmax=1)#, ax=ax)
# plt.show()

# plot_contour_spaghetti(ensemble, under_mask=img, arr=depths_bad, is_arr_categorical=False, vmin=0, vmax=1)
# plt.show()

fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
plot_contour_boxplot(ensemble, depths=depths_bad, under_mask=img, epsilon_out=10, ax=ax)
plt.show()
fig.savefig(f"/Users/chadepl/Downloads/han-cbd-{structure_name}.png")

fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
plot_contour_boxplot(ensemble, depths=depths_bod, under_mask=img, epsilon_out=10, ax=ax)
plt.show()
fig.savefig(f"/Users/chadepl/Downloads/han-bod-{structure_name}.png")

fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
plot_contour_boxplot(ensemble, depths=depths_wbod, under_mask=img, epsilon_out=10, ax=ax)
plt.show()
fig.savefig(f"/Users/chadepl/Downloads/han-wbod-{structure_name}.png")

# Plot showing local computation
from skimage.measure import find_contours

fig, ax = plt.subplots(layout="tight")
ax.imshow(img, cmap="gray")
for i, e in enumerate(ensemble):
    for c in find_contours(e):
        pass
        # ax.plot(c[:, 1], c[:, 0])

# Focus on one
ax.scatter(wbod_border_coords[0][:, 1], wbod_border_coords[0][:, 0], c=wbod_p_depths[0])
ax.set_axis_off()
plt.show()
