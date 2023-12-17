"""
In this file we compare the performance of CBD and ID
methods in terms of their capability to identify outliers.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    entry = dict(dataset_name=dataset_name, size=int(size), replication_id=int(rep), outs_gt_idx=outliers_idx)
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

df_exp["outs_idx"] = df_exp["depths"].apply(lambda v: np.argsort(v)[:get_num_outs(
    v, alpha=alpha_outs)])  # we get the ids of the contours with the 10 contours with the smallest depth

#############
# FILTERING #
#############

selected_datasets = [
    # "no_cont",
    "cont_mag_sym",
    "cont_mag_peaks",
    "cont_shape_in",
    "cont_shape_out",
    "cont_topo",
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

# Filtering
df_exp = df_exp.loc[df_exp["dataset_name"].apply(lambda v: v in selected_datasets)]
df_exp = df_exp.loc[df_exp["method"].apply(lambda v: v in selected_methods)]
df_exp = df_exp.loc[df_exp["method_family"].apply(lambda v: v in list(selected_methods_families.keys()))]
df_exp = df_exp.loc[df_exp["replication_id"] < 10]
df_exp = df_exp.loc[df_exp["size"] == 100]  # so that we compare the same for CBD and id

###############
# DF ASSEMBLY #
###############

index_cols = ["dataset_name", "size", "replication_id"]

# dataset size replication m1_med m2_med m3_med m4_med  m1_out m2_out m3_out m4_out
df_outs = df_exp.copy()
df_outs = df_outs.loc[:, index_cols + ["method", "outs_idx", ]]
df_outs = df_outs.pivot(index=index_cols, columns="method", values=["outs_idx", ])
df_outs = df_outs.reset_index()
df_outs.columns = [' '.join(col).strip() for col in df_outs.columns.values]
df_outs = df_outs.merge(df_datasets, left_on=index_cols, right_on=index_cols, how="left")

print(df_outs.columns)
print(df_outs.head())


def perc_outliers_identified(out_method, out_ref, adjust_size=False):
    """
    Given the outliers of a method (m_out) and reference ones (gt_out)
    computes the fraction of the ones in gt_out that are in m_out
    """
    if out_ref.size > 0:
        if adjust_size:
            return np.intersect1d(out_ref, out_method[:out_ref.size]).size / out_ref.size
        else:
            return np.intersect1d(out_ref, out_method).size / out_ref.size
    else:
        return pd.NA


cols = [c for c in df_outs.columns if
        "outs_idx" in c]  # ["outs_idx bad", "outs_idx mbad", "outs_idx mtbad"]#"outs_idx id", "outs_idx mid"]  # columns in pivoted table
# cols = df_outs
new_cols = []  # columns we are interested in for the analysis

# Using gt as ref
for col in cols:
    new_cols.append(f"perc gt_m {col}")
    df_outs[new_cols[-1]] = df_outs.apply(
        lambda x: perc_outliers_identified(x[col], x["outs_gt_idx"], adjust_size=False), axis=1)

# Using bad as ref
for col in cols:
    new_cols.append(f"perc cbd_m {col}")
    df_outs[new_cols[-1]] = df_outs.apply(
        lambda x: perc_outliers_identified(x[col], x["outs_idx mcbd"], adjust_size=False), axis=1)

df_outs = df_outs.loc[:, index_cols + new_cols]
df_outs = df_outs.set_index(index_cols).stack().reset_index()
df_outs["reference"] = df_outs["level_3"].apply(lambda v: v.split(" ")[1])
df_outs["method"] = df_outs["level_3"].apply(lambda v: v.split(" ")[-1])
df_outs = df_outs.drop(["level_3", ], axis=1)
df_outs = df_outs.rename({0: "percentage"}, axis=1)
df_outs = df_outs[index_cols + ["reference", "method", "percentage"]]
# df_outs.columns = index_cols + ["method", "percentage"]
# df_outs["method"] = df_outs["method"].apply(lambda x: x.split(" ")[-1])

df_outs = df_outs.astype(dict(dataset_name="category",
                              size="category",
                              replication_id="category",
                              reference="category",
                              method="category",
                              percentage=float))

###################
# FORMATTED TABLE #
###################

# Focus on size 100. To focus on other sizes add more ORs. Comment the line to consider the whole data
latex_df = df_outs.loc[np.logical_or(df_outs["size"] == 100, df_outs["size"] == 100), :]  # .drop(["size"], axis=1)

# Focus on the GT as reference
latex_df = latex_df.loc[latex_df["reference"] == "gt_m", :]

# Create groups and perform aggregations
latex_df = latex_df.groupby(by=["dataset_name", "size", "reference", "method"]).aggregate([np.mean, np.std])

# Shape table properly
latex_df = latex_df.unstack("method").unstack("reference")
latex_df = latex_df.dropna(how="all")  # For size values without entries

# Fix levels (axis=0)
latex_df.index = latex_df.index.set_levels(["D3", "D2", "D4", "D5", "D6", "D1"], level=0)
latex_df.index = latex_df.index.swaplevel(1, 0)
latex_df = latex_df.droplevel(0, axis=0)  # size level (drop if considering only one size)
latex_df = latex_df.sort_values(["dataset_name",], axis=0)  # add size to the list if considering more than one size

# Fix levels (axis=1)
latex_df = latex_df.droplevel(0, axis=1)  # percentage level
latex_df = latex_df.sort_values(["reference", "method"], axis=1, ascending=False)  # grouping mean+std per method
latex_df = latex_df.droplevel(0, axis=1)  # mean/std level
latex_df = latex_df.droplevel(1, axis=1)  # reference level
#latex_df.columns = latex_df.columns.swaplevel(1, 0)

# latex_df = latex_df.swaplevel(0, 2, axis=1)
# latex_df = latex_df.swaplevel(1, 2, axis=1)
# latex_df = latex_df.droplevel(2, axis=1)  # mean/std level

# From proportions to percentages
latex_df = latex_df * 100

# Final formatting
formatted_latex_table = latex_df.copy()
formatted_latex_table = formatted_latex_table.dropna(axis=1)
formatted_latex_table = formatted_latex_table.T
formatted_latex_table = formatted_latex_table.groupby(["method", ]).agg(
    lambda r: f"{r[0]:.2f} pm {r[1]:.2f}")
formatted_latex_table = formatted_latex_table.T
formatted_latex_table = formatted_latex_table[["cbd", "mcbd", "id", "mid"]]
formatted_latex_table.columns = ["CBD", "mCBD", "ID", "eID"]
print(formatted_latex_table.to_latex())

# latex_df = latex_df.T

# latex_df = latex_df[["cont_mag_sym", "cont_mag_peaks", "cont_shape_in", "cont_shape_out"]]
# latex_df.columns = latex_df.columns.set_levels(["D1", "D2", "D3", "D4"], level=1)
# latex_df.columns.set_levels = ["D1", "D2", "D3", "D4"]
# latex_df.columns = [e[1] for e in latex_df.columns.to_flat_index()]
# latex_df = latex_df.reindex(["mtbad", "id_base", "mid_nest"], axis=1)
# latex_df = latex_df.rename(dict(mtbad="CBD", id_base="ID", mid_nest="eID"), axis=1)
# print(latex_df.to_latex(float_format="%.2f"))

############
# PLOTTING #
############

if False:
    g = sns.FacetGrid(df_outs, col="dataset_name", col_wrap=2)
    g.map_dataframe(sns.lineplot, x="size", y="percentage", hue="method")

    g.set_xlabels("Size")
    g.set_ylabels("Percentage")

    g.fig.subplots_adjust(top=0.9)
    g.figure.suptitle("Percentages of Outlier Overlap vs Dataset Size for Contaminated Datasets")

    for kax, ax in g.axes_dict.items():
        names_dict = {"cont_mag_peaks": "Magnitude - Peaks",
                      "cont_mag_sym": "Magnitude - Symmetric",
                      "cont_shape_in": "Shape - Inside",
                      "cont_shape_out": "Shape - Outside"}
        if kax in names_dict:
            ax.set_title(names_dict[kax])

    g.add_legend()
    leg = g.legend
    for line in leg.get_lines():
        line.set_linewidth(3)

    plt.show()
    g.fig.savefig("results/sd_res_outliers/outlier_overlap.png")
