from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "..")
from src.vis_utils import plot_contour_spaghetti, plot_contour_boxplot

#############
# LOAD DATA #
#############
exp_dir = Path("results_data/exp_speed").resolve()
res_dir = exp_dir.joinpath("pickles")
ens_path = exp_dir.joinpath("ensembles")

# Retrieve GT outliers
entries = []
for f in ens_path.rglob("*pkl"):
    dataset_name, size, rep = f.stem.split("-")
    entry = dict(dataset_name=dataset_name, size=int(size), replication_id=int(rep), dataset_path=f)
    entries.append(entry)
df_datasets = pd.DataFrame(entries)

# Retrieve experimental results
entries = []
for f in res_dir.rglob("*pkl"):
    with open(f, "rb") as f1:
        entries += pickle.load(f1)

df_exp = pd.DataFrame(entries)
df_exp = df_exp.drop_duplicates(subset=["dataset_name", "size", "replication_id",
                                        "method"])  # this is due to an error in the experiments, there should be no duplicates

print(df_exp.head())

#############
# FILTERING #
#############

selected_datasets = [
    "no_cont",
    # "cont_mag_sym",
    # "cont_mag_peaks",
    # "cont_shape_in",
    # "cont_shape_out",
    # "cont_topo",
]

selected_methods = [
    "cbd",
    "mcbd",
    "id",
    "mid",
]

selected_methods_families = {
    "cbd": "red",
    "id": "blue"
}

selected_replica = dict(
    no_cont=0,
    cont_mag_sym=0,
    cont_mag_peaks=0,
    cont_shape_in=2,
    cont_shape_out=0,
    cont_topo=0
)[selected_datasets[0]]

# Filtering
df_exp = df_exp.loc[df_exp["dataset_name"].apply(lambda v: v in selected_datasets)]
df_exp = df_exp.loc[df_exp["method"].apply(lambda v: v in selected_methods)]
df_exp = df_exp.loc[df_exp["method_family"].apply(lambda v: v in list(selected_methods_families.keys()))]
df_exp = df_exp.loc[df_exp["replication_id"] == selected_replica]  # 0, 2, 8, 9
df_exp = df_exp.loc[df_exp["size"] == 100]

###############
# DF ASSEMBLY #
###############

index_cols = ["dataset_name", "size", "replication_id"]

# dataset size replication m1_med m2_med m3_med m4_med  m1_out m2_out m3_out m4_out
df_outs = df_exp.copy()
df_outs = df_outs.loc[:, index_cols + ["method", "depths", ]]
df_outs = df_outs.pivot(index=index_cols, columns="method", values=["depths", ])
df_outs = df_outs.reset_index()
df_outs.columns = [' '.join(col).strip() for col in df_outs.columns.values]
df_outs = df_outs.merge(df_datasets, left_on=index_cols, right_on=index_cols, how="left")

# print(df_outs.columns)
# print(df_outs.head())


# Open a replication of each dataset with sample size 100

print(df_exp.head())

for index, row in df_outs.iterrows():
    print(row["dataset_path"])
    with open(row["dataset_path"], "rb") as f:
        ensemble, labels = pickle.load(f)
    outliers_idx = np.where(labels == 1)[0]

    # Spaghetti plot with outliers highlighted

    fig, ax = plt.subplots(ncols=1, layout="tight", figsize=(10, 10))
    lab_sort = np.argsort(np.array(labels))
    plot_contour_spaghetti([ensemble[i] for i in lab_sort], arr=[labels[i] for i in lab_sort], is_arr_categorical=True,
                           linewidth=4, alpha=0.5, smooth_line=False, ax=ax)
    # plt.show()
    #fig.savefig(f"results/sd_res_overview/spa_outs-{row['dataset_name']}-{row['replication_id']}.svg")
    fig.savefig(f"results/sd_res_overview/spa_outs-{row['dataset_name']}-{row['replication_id']}.png", dpi=300)


    # num_out = np.where(np.isclose(row["depths mtbad"], 0))[0].size
    # print(row["depths mtbad"])
    # print(f" - {num_out}")
    # get GT outliers
    for ri in row.index:
        if "depths" in ri:
            print(ri)
            depths_ri = row[ri]
            fig, ax = plt.subplots(ncols=1, layout="tight", figsize=(10, 10))
            # thresholds: topo (0.11)
            plot_contour_boxplot(ensemble, depths_ri, outlier_type="tail", epsilon_out=int(depths_ri.size * 0.2), smooth_line=False, ax=ax)
            # ax.set_title(ri)
            # plt.show()
            fig.savefig(f"results/sd_res_overview/cbp-{row['dataset_name']}-{ri}-{row['replication_id']}.svg")

            # Spaghetti plot with depths
            vmin = vmax = 0
            if "cbd" in ri:
                vmax = 1
            if "id" in ri:
                vmax = 0.5

            fig, ax = plt.subplots(ncols=1, layout="tight", figsize=(10, 10))
            lab_sort = np.argsort(np.array(labels))
            plot_contour_spaghetti(ensemble, arr=depths_ri,
                                   is_arr_categorical=False,
                                   linewidth=1, alpha=0.5, smooth_line=False,
                                   # vmin=vmin, vmax=vmax,
                                   ax=ax)
            # plt.show()
            fig.savefig(f"results/sd_res_overview/spa_depths-{row['dataset_name']}-{ri}-{row['replication_id']}.svg")
    break


# color bar

x = np.linspace(0, 1, 100, ).reshape((1, -1))
x = np.repeat(x, 50, axis=0)
fig, ax = plt.subplots(figsize=(10, 3), layout="tight")
ax.matshow(x, cmap="magma")
ax.set_axis_off()
#plt.show()
fig.savefig("results/sd_res_overview/colorbar.png")