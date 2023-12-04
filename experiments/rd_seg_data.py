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
from src.vis_utils import plot_contour_spaghetti, plot_contour_boxplot
from src.contour_depths.datasets import han_ensembles
from src.contour_depths.datasets.bd_paper import get_han_dataset_ParotidR, get_han_dataset_BrainStem
from src.contour_depths.depths import band_depth, boundary_depth

structure_name = ["Parotid_R", "BrainStem"][0]

# img, gt, ensemble = han_ensembles.get_han_slice_ensemble(540, 540)
if structure_name == "Parotid_R":
    img, gt, ensemble = get_han_dataset_ParotidR(540, 540)
elif structure_name == "BrainStem":
    img, gt, ensemble = get_han_dataset_BrainStem(540, 540)

# spaghetti plot
labs = np.arange(len(ensemble))
np.random.shuffle(labs)
fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
plot_contour_spaghetti(ensemble, under_mask=img, resolution=(540, 540), ax=ax)
plt.show()
fig.savefig(f"/Users/chadepl/Downloads/han-spag-{structure_name}.svg")

# heatmap
ensemble_std = np.concatenate([np.expand_dims(e, axis=0) for e in ensemble], axis=0)
ensemble_std = np.std(ensemble_std, axis=0)
fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
ax.imshow(img, cmap="gray")
ax.imshow(ensemble_std, cmap="magma", alpha=(ensemble_std > 0).astype(float))
ax.set_axis_off()
plt.show()
fig.savefig(f"/Users/chadepl/Downloads/han-std-{structure_name}.svg")

# contour depths plot
res_path = Path("results_data/han-depths")
res_path = res_path.joinpath(f"{structure_name}")
if not res_path.exists():
    res_path.mkdir(parents=True)

#####################
# DEPTH COMPUTATION #
#####################

times = dict()

print("CBD ...")
cbd_depths_path = res_path.joinpath("depths_cbd.pkl")
cbd_times_path = res_path.joinpath("times_cbd.pkl")
if cbd_depths_path.exists():
    with open(cbd_depths_path, "rb") as f:
        depths_cbd = pickle.load(f)
    with open(cbd_times_path, "rb") as f:
        times["cbd"] = pickle.load(f)
else:
    print("- Results did not exist. Computing ...")
    t_start = time()
    depths_cbd = band_depth.compute_depths(ensemble, modified=False)
    t_end = time()
    times["cbd"] = t_end - t_start

    with open(cbd_depths_path, "wb") as f:
        pickle.dump(depths_cbd, f)
    with open(cbd_times_path, "wb") as f:
        pickle.dump(times["cbd"], f)

print("mCBD ...")
mcbd_depths_path = res_path.joinpath("depths_mcbd.pkl")
mcbd_times_path = res_path.joinpath("times_mcbd.pkl")
if mcbd_depths_path.exists():
    with open(mcbd_depths_path, "rb") as f:
        depths_mcbd = pickle.load(f)
    with open(mcbd_times_path, "rb") as f:
        times["mcbd"] = pickle.load(f)
else:
    print("- Results did not exist. Computing ...")
    t_start = time()
    depths_mcbd = band_depth.compute_depths(ensemble, modified=True, target_mean_depth=None)
    t_end = time()
    times["mcbd"] = t_end - t_start

    with open(mcbd_depths_path, "wb") as f:
        pickle.dump(depths_mcbd, f)
    with open(mcbd_times_path, "wb") as f:
        pickle.dump(times["mcbd"], f)

print("BoD ...")
bod_depths_path = res_path.joinpath("depths_bod.pkl")
bod_times_path = res_path.joinpath("times_bod.pkl")
if bod_depths_path.exists():
    with open(bod_depths_path, "rb") as f:
        depths_bod = pickle.load(f)
    with open(bod_times_path, "rb") as f:
        times["bod"] = pickle.load(f)
else:
    print("- Results did not exist. Computing ...")
    t_start = time()
    depths_bod = boundary_depth.compute_depths(ensemble, modified=False)
    t_end = time()
    times["bod"] = t_end - t_start

    with open(bod_depths_path, "wb") as f:
        pickle.dump(depths_bod, f)
    with open(bod_times_path, "wb") as f:
        pickle.dump(times["bod"], f)

print("mBoD ...")
mbod_depths_path = res_path.joinpath("depths_mbod.pkl")
mbod_times_path = res_path.joinpath("times_mbod.pkl")
if mbod_depths_path.exists():
    with open(mbod_depths_path, "rb") as f:
        depths_mbod = pickle.load(f)
    with open(mbod_times_path, "rb") as f:
        times["mbod"] = pickle.load(f)
else:
    print("- Results did not exist. Computing ...")
    t_start = time()
    depths_mbod = boundary_depth.compute_depths(ensemble, modified=True)
    t_end = time()
    times["mbod"] = t_end - t_start

    with open(mbod_depths_path, "wb") as f:
        pickle.dump(depths_mbod, f)
    with open(mbod_times_path, "wb") as f:
        pickle.dump(times["mbod"], f)


print(f"CBD: {times['cbd']}")
print(f"mCBD: {times['mcbd']}")
print(f"BoD: {times['bod']}")
print(f"mBoD: {times['mbod']}")

###########
# RESULTS #
###########

print("Outliers")
print(np.argsort(depths_cbd)[:10])
print(np.argsort(depths_mcbd)[:10])
print(np.argsort(depths_bod)[:10])
print(np.argsort(depths_mbod)[:10])

print("Inliers")
print(np.argsort(depths_cbd)[::-1][:10])
print(np.argsort(depths_mcbd)[::-1][:10])
print(np.argsort(depths_bod)[::-1][:10])
print(np.argsort(depths_mbod)[::-1][:10])

depths_arr = [depths_cbd, depths_mcbd, depths_bod, depths_mbod]
df = pd.DataFrame(depths_arr).T
df.columns = ["CBD", "mCBD", "BoD", "mBoD"]
sns.pairplot(df)
plt.show()
print(df.shape)

overlap_in = np.zeros((4, 4))
overlap_out = np.zeros((4, 4))
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
plot_contour_boxplot(ensemble, depths=depths_cbd, under_mask=img, epsilon_out=10, ax=ax)
plt.show()
fig.savefig(f"/Users/chadepl/Downloads/han-cbd-{structure_name}.png")

fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
plot_contour_boxplot(ensemble, depths=depths_mcbd, under_mask=img, epsilon_out=10, ax=ax)
plt.show()
fig.savefig(f"/Users/chadepl/Downloads/han-mcbd-{structure_name}.png")

fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
plot_contour_boxplot(ensemble, depths=depths_bod, under_mask=img, epsilon_out=10, ax=ax)
plt.show()
fig.savefig(f"/Users/chadepl/Downloads/han-bod-{structure_name}.png")

fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
plot_contour_boxplot(ensemble, depths=depths_mbod, under_mask=img, epsilon_out=10, ax=ax)
plt.show()
fig.savefig(f"/Users/chadepl/Downloads/han-mbod-{structure_name}.png")