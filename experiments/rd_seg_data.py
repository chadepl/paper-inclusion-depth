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

import sys
sys.path.insert(0, "..")
from src.vis_utils import plot_contour_spaghetti, plot_contour_boxplot
from src.datasets import han_ensembles
from src.datasets.id_paper import get_han_dataset_ParotidR, get_han_dataset_BrainStem
from src.depths import band_depth, inclusion_depth

structure_name = ["Parotid_R", "BrainStem"][1]

res_path = Path(f"results/rd-seg/{structure_name}")

if res_path.exists():
    print(f"Dir {res_path} already exists")
else:
    res_path.mkdir(exist_ok=True, parents=True)

    # img, gt, ensemble = han_ensembles.get_han_slice_ensemble(540, 540)
    if structure_name == "Parotid_R":
        img, gt, ensemble = get_han_dataset_ParotidR(540, 540, "../data/han_ensembles")
    elif structure_name == "BrainStem":
        img, gt, ensemble = get_han_dataset_BrainStem(540, 540, "../data/han_ensembles")

    # spaghetti plot
    labs = np.arange(len(ensemble))
    np.random.shuffle(labs)
    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    plot_contour_spaghetti(ensemble, under_mask=img, resolution=(540, 540), ax=ax)
    # plt.show()
    fig.savefig(res_path.joinpath(f"han-spag-{structure_name}.svg"))

    # heatmap
    ensemble_std = np.concatenate([np.expand_dims(e, axis=0) for e in ensemble], axis=0)
    ensemble_std = np.std(ensemble_std, axis=0)
    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    ax.imshow(img, cmap="gray")
    ax.imshow(ensemble_std, cmap="magma", alpha=(ensemble_std > 0).astype(float))
    ax.set_axis_off()
    # plt.show()
    fig.savefig(res_path.joinpath(f"han-std-{structure_name}.svg"))

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

    print("ID ...")
    id_depths_path = res_path.joinpath("depths_id.pkl")
    id_times_path = res_path.joinpath("times_id.pkl")
    if id_depths_path.exists():
        with open(id_depths_path, "rb") as f:
            depths_id = pickle.load(f)
        with open(id_times_path, "rb") as f:
            times["id"] = pickle.load(f)
    else:
        print("- Results did not exist. Computing ...")
        t_start = time()
        depths_id = inclusion_depth.compute_depths(ensemble, modified=False)
        t_end = time()
        times["id"] = t_end - t_start

        with open(id_depths_path, "wb") as f:
            pickle.dump(depths_id, f)
        with open(id_times_path, "wb") as f:
            pickle.dump(times["id"], f)

    print("mID ...")
    mid_depths_path = res_path.joinpath("depths_mid.pkl")
    mid_times_path = res_path.joinpath("times_mid.pkl")
    if mid_depths_path.exists():
        with open(mid_depths_path, "rb") as f:
            depths_mid = pickle.load(f)
        with open(mid_times_path, "rb") as f:
            times["mid"] = pickle.load(f)
    else:
        print("- Results did not exist. Computing ...")
        t_start = time()
        depths_mid = inclusion_depth.compute_depths(ensemble, modified=True)
        t_end = time()
        times["mid"] = t_end - t_start

        with open(mid_depths_path, "wb") as f:
            pickle.dump(depths_mid, f)
        with open(mid_times_path, "wb") as f:
            pickle.dump(times["mid"], f)

    print(f"CBD: {times['cbd']}")
    print(f"mCBD: {times['mcbd']}")
    print(f"ID: {times['id']}")
    print(f"eID: {times['mid']}")

    ###########
    # RESULTS #
    ###########

    print("Outliers")
    out_mcbd = np.argsort(depths_mcbd)[:12]
    out_mid = np.argsort(depths_mid)[:12]
    print(f"{np.intersect1d(out_mid, out_mcbd).size}/12")
    print(out_mcbd)
    print(out_mid)

    print("Inliers")
    in_mcbd = np.argsort(depths_mcbd)[::-1][:100]
    in_mid = np.argsort(depths_mid)[::-1][:100]
    print(f"{np.intersect1d(in_mid, in_mcbd).size}/100")
    print(in_mcbd)
    print(in_mid)

    print("Score correlation")
    print(np.corrcoef(depths_mcbd, depths_mid))

    depths_arr = [depths_cbd, depths_mcbd, depths_id, depths_mid]
    df = pd.DataFrame(depths_arr).T
    df.columns = ["CBD", "mCBD", "ID", "eID"]
    # sns.pairplot(df)
    # plt.show()
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

    print("Masks comparison")
    med_cbd = ensemble[in_mcbd[0]]
    med_id = ensemble[in_mid[0]]
    mean_cbd = (np.array([ensemble[e] for e in in_mcbd]).mean(axis=0)>0.5).astype(float)
    mean_id = (np.array([ensemble[e] for e in in_mid]).mean(axis=0)>0.5).astype(float)

    print(f"MSE Med: {np.square(med_cbd - med_id).mean()}")
    print(f"MSE Mean: {np.square(mean_cbd - mean_id).mean()}")

    from skimage.filters import gaussian
    c1 = med_cbd
    c2 = med_id
    for i in range(1):
        c1 = gaussian(c1, 3)
        c2 = gaussian(c2, 3)
    fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
    ax.imshow(img, cmap="gray")
    ax.contour(c1, levels=[0.5, ], colors=["yellow"], linestyles=["solid",])
    ax.contour(c2, levels=[0.5, ], colors=["yellow"], linestyles=["dotted",], linewidths=[3,])
    ax.set_axis_off()
    fig.savefig(res_path.joinpath(f"han-cbd-vs-id-med-{structure_name}.png"))
    # plt.show()

    c1 = mean_cbd
    c2 = mean_id
    for i in range(1):
        c1 = gaussian(c1, 3)
        c2 = gaussian(c2, 3)
    fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
    ax.imshow(img, cmap="gray")
    ax.contour(c1, levels=[0.5, ], colors=["dodgerblue"], linestyles=["solid",])
    ax.contour(c2, levels=[0.5, ], colors=["dodgerblue"], linestyles=["dotted",], linewidths=[3,])
    ax.set_axis_off()
    fig.savefig(res_path.joinpath(f"han-cbd-vs-id-mean-{structure_name}.png"))
    # plt.show()

    # plot_contour_spaghetti(ensemble, under_mask=img, arr=depths_id, is_arr_categorical=False, vmin=0, vmax=1)#, ax=ax)
    # plt.show()

    # plot_contour_spaghetti(ensemble, under_mask=img, arr=depths_bad, is_arr_categorical=False, vmin=0, vmax=1)
    # plt.show()

    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    plot_contour_boxplot(ensemble, depths=depths_cbd, under_mask=img, epsilon_out=10, ax=ax)
    # plt.show()
    fig.savefig(res_path.joinpath(f"han-cbd-{structure_name}.png"))

    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    plot_contour_boxplot(ensemble, depths=depths_mcbd, under_mask=img, epsilon_out=10, ax=ax)
    # plt.show()
    fig.savefig(res_path.joinpath(f"han-mcbd-{structure_name}.png"))

    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    plot_contour_boxplot(ensemble, depths=depths_id, under_mask=img, epsilon_out=10, ax=ax)
    # plt.show()
    fig.savefig(res_path.joinpath(f"han-id-{structure_name}.png"))

    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    plot_contour_boxplot(ensemble, depths=depths_mid, under_mask=img, epsilon_out=10, ax=ax)
    # plt.show()
    fig.savefig(res_path.joinpath(f"han-mid-{structure_name}.png"))