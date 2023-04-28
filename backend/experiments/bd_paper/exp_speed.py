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
    # "cont_mag_sym",
    # "cont_mag_peaks",
    # "cont_shape_in",
    # "cont_shape_out",
]

selected_methods = dict(
    # bad = cb_colors["purple"],
    # mbad = cb_colors["orange"],
    mtbad = cb_colors["green"],
    #bod_base = cb_colors["blue"],
    bod_fast = cb_colors["blue"],
    # bod_nest = cb_colors["yellow"],
    mbod_nest = cb_colors["red"],
    # mbod_l2 = cb_colors["brown"]
)

selected_methods_families = {
    "bad": cb_colors["blue"],
    "bod": cb_colors["red"]
}

# Filtering
df = df.loc[df["dataset_name"].apply(lambda v: v in selected_datasets)]
df = df.loc[df["method"].apply(lambda v: v in list(selected_methods.keys()))]
df = df.loc[df["method_family"].apply(lambda v: v in list(selected_methods_families.keys()))]
df = df.loc[df["replication_id"] < 5]

# Renaming
#df["method_family"] = df["method_family"].apply(lambda v: "Contour Band Depths" if v == "bad" else "Boundary Depths")
#df["method_type"] =df["method"].apply(lambda v: "Modified" if "m" in v else "Strict")

sns.set_palette("colorblind")
fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
sns_lp = sns.lineplot(df, x="size", y="time_core", hue="method", palette=selected_methods,  ax=ax)

# sns.regplot(x="size", y="time_core", data=df.loc[df["method_family"] == "Contour Band Depths"],
#            order=3, ci=None, scatter_kws={"s": 80}, color="red", ax=ax);
# sns.regplot(x="size", y="time_core", data=filtered_df.loc[filtered_df["method_family"] == "Boundary Depths"],
#            order=2, ci=None, scatter_kws={"s": 80}, color="green", ax=ax);

sns_lp.set(yscale='log', xscale="log")
ax.set_title("Runtimes vs Ensemble Size for \n Contour Band Depths and Boundary Depths ")
ax.set_ylabel("Time (seconds)")
ax.set_xlabel("Size")

leg = sns_lp.get_legend()
leg.set_title("Method")  # works if legend is not nested
for line in leg.get_lines():
    line.set_linewidth(3)
print(leg.texts)
for t in leg.texts:
    if t.get_text() == "bod_fast":
        t.set_text("BoD")
    if t.get_text() == "mbod_nest":
        t.set_text("wBoD")
    if t.get_text() == "mtbad":
        t.set_text("CBD")

plt.show()

fig.savefig("/Users/chadepl/Downloads/speed_gains.png")

