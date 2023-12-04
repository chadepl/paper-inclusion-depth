"""
In this experiment we compare the speed of 2 methods:
 - band depth (simplicial)
 - thresholded band depth (has an extra binary optimization process)
 - boundary depth (half space)


"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

exp_dir = Path("results_data/exp_speed").resolve()
res_dir = exp_dir.joinpath("pickles")
ens_path = exp_dir.joinpath("ensembles")

entries = []
for f in res_dir.rglob("*pkl"):
    with open(f, "rb") as f1:
        entries += pickle.load(f1)

time_entries = []
for entry in entries:
    new_entry = dict()
    for k, v in entry.items():
        if "depths" not in k:
            new_entry[k] = v

    time_entries.append(new_entry)

df = pd.DataFrame(time_entries)
print(df.head())
print(df.columns)
print(df["dataset_name"].unique())

cb_colors = dict(
    red="#e41a1c",
    blue="#377eb8",
    green="#4daf4a",
    purple="#984ea3",
    orange="#ff7f00",
    yellow="#ffff33",
    brown="#a65628",
    pink="#f781bf")

selected_datasets = [
    "no_cont",
    "cont_mag_sym",
    "cont_mag_peaks",
    "cont_shape_in",
    "cont_shape_out",
    "cont_topo",
]

selected_methods = dict(
    cbd = cb_colors["purple"],
    mcbd = cb_colors["orange"],
    bod = cb_colors["blue"],
    mbod=cb_colors["green"],
)

selected_methods_families = {
    "cbd": cb_colors["blue"],
    "bod": cb_colors["red"]
}

# New columns
df["time_full"] = df["time_preproc"] + df["time_core"]

# Filtering
df = df.loc[df["dataset_name"].apply(lambda v: v in selected_datasets)]
df = df.loc[df["method"].apply(lambda v: v in list(selected_methods.keys()))]
df = df.loc[df["method_family"].apply(lambda v: v in list(selected_methods_families.keys()))]
df = df.loc[df["replication_id"] < 10]

# Renaming
# df["method_family"] = df["method_family"].apply(lambda v: "Contour Band Depths" if v == "bad" else "Boundary Depths")
# df["method_type"] =df["method"].apply(lambda v: "Modified" if "m" in v else "Strict")

##########
# FIGURE #
##########

sns.set_palette("colorblind")
fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
sns_lp = sns.lineplot(df, x="size", y="time_full", hue="method", palette=selected_methods, linewidth=2, ax=ax)

# sns.regplot(x="size", y="time_core", data=df.loc[df["method_family"] == "Contour Band Depths"],
#            order=3, ci=None, scatter_kws={"s": 80}, color="red", ax=ax);
# sns.regplot(x="size", y="time_core", data=filtered_df.loc[filtered_df["method_family"] == "Boundary Depths"],
#            order=2, ci=None, scatter_kws={"s": 80}, color="green", ax=ax);

sns_lp.set(yscale='log', xscale="log")
ax.set_title("Runtimes vs Ensemble Size for \n Contour Band Depth and Inclusion Depth ")
ax.set_ylabel("Time (seconds)")
ax.set_xlabel("Size")

handles, labels = ax.get_legend_handles_labels()
order = [2, 0, 1, 3]
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

leg = ax.get_legend()
leg.set_title("Method")  # works if legend is not nested
for line in leg.get_lines():
    line.set_linewidth(3)
print(leg.texts)
for t in leg.texts:
    if t.get_text() == "bod":
        t.set_text("ID")
    if t.get_text() == "mbod":
        t.set_text("eID")
    if t.get_text() == "cbd":
        t.set_text("CBD")
    if t.get_text() == "mcbd":
        t.set_text("mCBD")



plt.show()

fig.savefig("results/sd_res_speed/speed_gains.svg")

#########
# TABLE #
#########

dataset_names_dict = dict(no_cont="D1", cont_mag_sym="D2", cont_mag_peaks="D3", cont_shape_in="D4", cont_shape_out="D5", cont_topo="D6")
time_type_dict = dict(time_preproc="t1", time_core="t2", time_full="t3")

latex_df = df.copy()
latex_df = latex_df[latex_df["dataset_name"] == "no_cont"]
latex_df = latex_df[latex_df["size"] == 100]


latex_df["dataset_name"] = latex_df["dataset_name"].apply(lambda v: dataset_names_dict[v])
latex_df = latex_df.loc[latex_df["size"] == 100, ["dataset_name", "method", "time_preproc", "time_core", "time_full"]].melt(id_vars=["dataset_name", "method"], value_vars=["time_preproc", "time_core", "time_full"], var_name="time_type", value_name="time_value")
latex_df["time_type"] = latex_df["time_type"].apply(lambda v: time_type_dict[v])
latex_df = latex_df.groupby(by=["dataset_name", "method", "time_type"]).aggregate([np.mean, np.std]).unstack(level=1)

latex_df = latex_df.droplevel(0, axis=1)
latex_df.columns.names = ["statistic", "method"]
latex_df = latex_df.sort_values(["method", "statistic"], axis=1, ascending=True)
latex_df = latex_df.swaplevel(0, 1, axis=1)

formatted_latex_table = latex_df.copy()
formatted_latex_table = formatted_latex_table.dropna(axis=1)
formatted_latex_table = formatted_latex_table.T
formatted_latex_table = formatted_latex_table.reset_index()
formatted_latex_table.columns = ["method", "statistic", "t1", "t2", "t3"]
print()
def apply_fun(*args):
    print(args)
    df_sub = args[0].reset_index()
    outs = [df_sub.loc[0, "method"]]
    outs.append(f"{df_sub.loc[0, 't1']:.2f} pm {df_sub.loc[1, 't1']:.2f}")
    outs.append(f"{df_sub.loc[0, 't2']:.2f} pm {df_sub.loc[1, 't2']:.2f}")
    outs.append(f"{df_sub.loc[0, 't3']:.2f} pm {df_sub.loc[1, 't3']:.2f}")
    return pd.Series(outs, index=["method", "t1", "t2", "t3"])


# formatted_latex_table = formatted_latex_table.groupby("method").agg(
#     lambda r: f"{r[0]:.2f} pm {r[1]:.2f}")
formatted_latex_table = formatted_latex_table.groupby("method").apply(
    apply_fun)
#formatted_latex_table = formatted_latex_table.T
#formatted_latex_table = formatted_latex_table.droplevel(0, axis=1)
formatted_latex_table = formatted_latex_table.loc[:, ["t1", "t2", "t3"]]
formatted_latex_table = formatted_latex_table.loc[["cbd", "mcbd", "bod", "mbod"],:]
formatted_latex_table = formatted_latex_table.reset_index()

def change_method_lab(m):
    return dict(cbd="CBD", mcbd="mCBD", bod="BoD", mbod="mBoD")[m]
formatted_latex_table["method"] = formatted_latex_table["method"].apply(change_method_lab)

#formatted_latex_table.rows = ["WC (x10^-3)", "BoD (x10^-5)", "wBoD (x10^-5)"]
print(formatted_latex_table.to_latex())

#latex_df.index = latex_df.index.set_levels([dataset_names_dict[n] for n in latex_df.index.get_level_values(level=0)], level=0)

print(latex_df.to_latex())