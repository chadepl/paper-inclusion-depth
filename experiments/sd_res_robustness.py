# We compare the multivariate median and trimmed mean in terms of distance
# Given that we are dealing with masks we will use dice coefficient. But maybe not ideal


# df_exp["median_idx"] = df_exp["depths"].apply(lambda v: np.argsort(v)[-3:])

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.io import imsave

import sys
sys.path.insert(0, "..")
from src.datasets.bd_paper_v2 import get_population_mean

#############
# LOAD DATA #
#############
exp_dir = Path("results_data/exp_speed").resolve()
res_dir = exp_dir.joinpath("pickles")
ens_path = exp_dir.joinpath("ensembles")

# Retrieve GT outliers
entries = []
for f in ens_path.rglob("*pkl"):
    with open(f, "rb") as f1:
        ens, labs = pickle.load(f1)
    dataset_name, size, rep = f.stem.split("-")
    outliers_idx = np.where(labs == 1)[0]
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


def get_num_outs(depths, alpha=0.2):
    num_outs = int(np.ceil(len(depths) * alpha))  # fractional
    # num_outs = 10  # constant
    return num_outs

alpha_outs = 0.3

df_exp["ins_idx"] = df_exp["depths"].apply(lambda v: np.argsort(v)[get_num_outs(
    v, alpha_outs):])  # we get the ids of the contours with the N - N \times alpha largest depths
df_exp["outs_idx"] = df_exp["depths"].apply(lambda v: np.argsort(v)[:get_num_outs(v, alpha_outs)])

print(df_exp.head())

#############
# FILTERING #
#############

selected_datasets = [
    "no_cont",
    "cont_mag_sym",
    "cont_mag_peaks",
    "cont_shape_in",
    "cont_shape_out",
    "cont_topo"
]

selected_methods = [
    "cbd",
    "mcbd",
    "bod",
    "mbod",
]

selected_methods_families = {
    "cbd": "red",
    "bod": "blue"
}

# Filtering
df_exp = df_exp.loc[df_exp["dataset_name"].apply(lambda v: v in selected_datasets)]
df_exp = df_exp.loc[df_exp["method"].apply(lambda v: v in selected_methods)]
df_exp = df_exp.loc[df_exp["method_family"].apply(lambda v: v in list(selected_methods_families.keys()))]
df_exp = df_exp.loc[df_exp["replication_id"] < 10]
df_exp = df_exp.loc[df_exp["size"] == 100]  # so that we compare the same for CBD and BoD

###############
# DF ASSEMBLY #
###############

index_cols = ["dataset_name", "size", "replication_id"]

# dataset size replication m1_med m2_med m3_med m4_med  m1_out m2_out m3_out m4_out
df_ins = df_exp.copy()
df_ins = df_ins.loc[:, index_cols + ["method", "ins_idx", "outs_idx"]]
df_ins = df_ins.pivot(index=index_cols, columns="method", values=["ins_idx", "outs_idx"])
df_ins = df_ins.reset_index()
df_ins.columns = [' '.join(col).strip() for col in df_ins.columns.values]
df_ins = df_ins.merge(df_datasets, left_on=index_cols, right_on=index_cols, how="left")


# Per row (dataset, size, replication), compute the estimators and compare them to f()
# Estimators we consider: sample mean (WC), MVM, robust mean CBD, mCBD, BOD, mBOD
stats_df = []
statistic = ["median", "trimmed_mean"][1]
for index, row in df_ins.iterrows():

    # Open dataset
    with open(row["dataset_path"], "rb") as f:
        ensemble, labs = pickle.load(f)
        mask_size = ensemble[0].shape[0]

    entry = dict(dataset_name=row["dataset_name"], size=row["size"], replication_id=row["replication_id"])

    # Obtain reference f()
    f_mask = get_population_mean(radius=0.5, grid_size=mask_size)

    # Estimators computation

    # ref_est = ["ins_idx cdb", ensemble[row["ins_idx cdb"][0]]]  # reference estimator could be another method

    # - Sample mean
    # -- We compute the sample mean
    # -- We compute the difference between the sample mean and the population mean
    masks_arr = np.concatenate([mask.reshape(1, -1) for mask in ensemble], axis=0)
    sample_mean = masks_arr.mean(axis=0)
    sample_mean = sample_mean.reshape((mask_size, mask_size))
    sample_mean_error = np.square(sample_mean - f_mask).sum()/(mask_size * mask_size)
    entry[f"pop_error-sample_mean"] = sample_mean_error

    # - Multivariate median
    mv_median = np.sort(masks_arr, axis=0)
    mv_median = mv_median[masks_arr.shape[0]//2].reshape((mask_size, mask_size))
    mv_median_error = np.square(mv_median - f_mask).sum()/(mask_size * mask_size)
    entry[f"pop_error-mvm"] = mv_median_error

    # - Depth methods' trimmed means
    other_est = []
    for v in row.index.values:
        if "ins_idx" in v:
            depth_method = v.split()[1]
            ensemble_in_idx = row[v]
            alpha_ensemble = [ensemble[in_id].reshape(1, -1) for in_id in ensemble_in_idx]
            alpha_ensemble = np.concatenate(alpha_ensemble, axis=0)
            trimmed_mean = alpha_ensemble.mean(axis=0).reshape((mask_size, mask_size))
            entry[f"pop_error-trimmed_mean_{depth_method}"] = np.square(trimmed_mean - f_mask).sum()/(mask_size * mask_size)

    name = f"{row['dataset_name']}_{row['replication_id']}"
    imsave(f"results/sd_res_robustness/{name}-sample_mean.png", sample_mean)
    imsave(f"results/sd_res_robustness/{name}-mvm.png", mv_median)
    imsave(f"results/sd_res_robustness/{name}-trimmed_mean.png", sample_mean)
    # # - Depth methods' trimmed means
    # other_est = [[i, ensemble[row[i][0]]] for i in ["ins_idx bod_fast", "ins_idx mbod_nest"]]
    #
    # diffs = []
    # for om_n, om_id in other_est:
    #     entry[f"med-{om_n.split(' ')[1]}"] = (np.square(ref_est[1] - om_id)).mean()
    #
    # # - worse case median
    # entry["med-worse"] = (np.square(ref_est[1] - ensemble[row["outs_idx mtbad"][0]])).mean()
    #
    # # - Depth methods' medians
    # mean_arr_ref_est = np.concatenate([ensemble[i][:, :, np.newaxis] for i in row["ins_idx mtbad"]], axis=-1).mean(
    #     axis=-1).squeeze()
    # iso_mean_arr_ref_est = (mean_arr_ref_est >= 0.5).astype(float)
    # ref_est = ["ins_idx mtbad", mean_arr_ref_est]
    # other_est = []
    # for i in ["ins_idx bod_fast", "ins_idx mbod_nest"]:
    #     mean_arr = np.concatenate([ensemble[j][:, :, np.newaxis] for j in row[i]], axis=-1).mean(axis=-1).squeeze()
    #     iso_mean = (mean_arr >= 0.5).astype(float)
    #     other_est.append([i, iso_mean])
    #
    # diffs = []
    # for om_n, om_id in other_est:
    #     entry[f"mean-{om_n.split(' ')[1]}"] = (np.square(ref_est[1] - om_id)).mean()
    #
    # # - worse case mean
    # random_mean = [ensemble[i][:, :, np.newaxis] for i in row["outs_idx mtbad"]]
    # random_mean = np.concatenate(random_mean, axis=-1).mean(axis=-1).squeeze()
    # random_mean = (random_mean >= 0.5).astype(float)
    # entry["mean-worse"] = (np.square(ref_est[1] - random_mean)).mean()

    stats_df.append(entry)

stats_df = pd.DataFrame(stats_df)
stats_df = stats_df.set_index(index_cols).stack().reset_index()
stats_df["statistic"] = stats_df["level_3"].apply(lambda v: v.split("-")[0])
stats_df["method"] = stats_df["level_3"].apply(lambda v: v.split("-")[1])
stats_df = stats_df.drop(["level_3", ], axis=1)
stats_df = stats_df.rename({0: "MSE"}, axis=1)
stats_df = stats_df[index_cols + ["statistic", "method", "MSE"]]

stats_df = stats_df.astype(dict(dataset_name="category",
                                size="category",
                                replication_id="category",
                                statistic="category",
                                method="category",
                                MSE=float))

print(stats_df.head())

###################
# FORMATTED TABLE #
###################
# we focus on size 100
latex_df = stats_df.loc[np.logical_or(stats_df["size"] == 100, stats_df["size"] == 100), :]
latex_df = latex_df.loc[latex_df["statistic"] == "pop_error", :]  # Focus on the GT as reference
latex_df = latex_df.groupby(by=["dataset_name", "size", "method", "statistic"]).aggregate([np.mean, np.std])
# latex_df = latex_df.reset_index()
# latex_df = latex_df.pivot(index=["dataset_name", "size"], columns="method", values="percentage")
latex_df = latex_df.dropna()  # For size values without entries
latex_df = latex_df.unstack("method").unstack("statistic")
latex_df.index = latex_df.index.set_levels(["D3", "D2", "D4", "D5", "D6", "D1"], level=0)
latex_df.index = latex_df.index.swaplevel(1, 0)

latex_df = latex_df.sort_values(["size", "dataset_name"], axis=0)
latex_df = latex_df.droplevel(0, axis=0)  # size level (drop if considering only one size)

latex_df = latex_df.droplevel(0, axis=1)
latex_df = latex_df.sort_values(["statistic", "method"], axis=1, ascending=False)

latex_df = latex_df.swaplevel(0, 2, axis=1)
# latex_df = latex_df.droplevel(2, axis=1)  # mean/std level

formatted_latex_table = latex_df.copy()
formatted_latex_table = formatted_latex_table * 100
formatted_latex_table = formatted_latex_table.dropna(axis=1)
formatted_latex_table = formatted_latex_table.T
formatted_latex_table = formatted_latex_table.groupby(["statistic", "method"]).agg(
    lambda r: f"{r[0]:2.2f} pm {r[1]:2.2f}")
formatted_latex_table = formatted_latex_table.T
formatted_latex_table = formatted_latex_table.droplevel(0, axis=1)
#formatted_latex_table = formatted_latex_table[["sample_mean", "mvm", "trimmed_mean_cbd", "trimmed_mean_mcbd", "trimmed_mean_bod", "trimmed_mean_mbod"]]
formatted_latex_table = formatted_latex_table[["sample_mean", "trimmed_mean_cbd", "trimmed_mean_mcbd", "trimmed_mean_bod", "trimmed_mean_mbod"]]
#formatted_latex_table.columns = ["SM", "MVM", "mu_CBD", "mu_mCBD", "mu_BOD", "mu_mBOD"]
formatted_latex_table.columns = ["SM", "mu_CBD", "mu_mCBD", "mu_BOD", "mu_mBOD"]
print(formatted_latex_table.to_latex())

# latex_df = latex_df[["cont_mag_sym", "cont_mag_peaks", "cont_shape_in", "cont_shape_out"]]
# latex_df.columns = latex_df.columns.set_levels(["D1", "D2", "D3", "D4"], level=1)
# latex_df.columns.set_levels = ["D1", "D2", "D3", "D4"]
# latex_df.columns = [e[1] for e in latex_df.columns.to_flat_index()]
# latex_df = latex_df.reindex(["mtbad", "bod_base", "mbod_nest"], axis=1)
# latex_df = latex_df.rename(dict(mtbad="CBD", bod_base="BoD", mbod_nest="wBoD"), axis=1)
# print(latex_df.to_latex(float_format="%.4f"))

# g = sns.FacetGrid(stats_df, col="dataset_name", col_wrap=2)
# g.map_dataframe(sns.lineplot, x="size", y="MSE", hue="method")
#
# g.fig.subplots_adjust(top=0.9)
# g.figure.suptitle(f"{statistic}")
#
# g.add_legend()
# leg = g.legend
# for line in leg.get_lines():
#     line.set_linewidth(3)
#
# plt.show()

# plt.imshow(ensemble[0])
# plt.show()
