import pickle
from pathlib import Path
from time import time
import numpy as np

import sys
sys.path.insert(0, "..")
from src.depths import band_depth, inclusion_depth
# from src.contour_depths.datasets.id_paper import get_contaminated_contour_ensemble_center, \
#     get_contaminated_contour_ensemble_magnitude, get_contaminated_contour_ensemble_shape, \
#     get_contaminated_contour_ensemble_topological
from src.datasets.id_paper_v2 import dataset_no_outliers, \
    dataset_sym_mag_outliers, \
    dataset_peaks_mag_outliers, \
    dataset_in_shape_outliers, \
    dataset_out_shape_outliers, \
    dataset_topological_outliers

num_replications = 10
num_rows = 300
num_cols = 300
max_size = 500
size_limit_cbd = 100
samples_spacing = 10

exp_dir = Path("results_data/exp_speed").resolve()
res_dir = exp_dir.joinpath("pickles")
ens_dir = exp_dir.joinpath("ensembles")

# The loop below redo's stuff that has not been done before

if not exp_dir.exists():
    print(f"Creating: {exp_dir}")
    exp_dir.mkdir()

if not res_dir.exists():
    print(f"Creating: {res_dir}")
    res_dir.mkdir()

if not ens_dir.exists():
    print(f"Creating: {ens_dir}")
    ens_dir.mkdir()

sizes = np.concatenate([np.arange(10, size_limit_cbd + 1, samples_spacing).astype(int),
                        np.arange(size_limit_cbd, max_size + 1, samples_spacing * 5).astype(int)[1:]])

# v1 of the paper
# datasets = dict(
#     no_cont=lambda nm,nr,nc: get_contaminated_contour_ensemble_magnitude(nm, nr, nc, case=0, return_labels=True, p_contamination=0.1),
#     cont_mag_sym=lambda nm,nr,nc: get_contaminated_contour_ensemble_magnitude(nm, nr, nc, case=1, return_labels=True, p_contamination=0.1),
#     cont_mag_peaks=lambda nm,nr,nc: get_contaminated_contour_ensemble_magnitude(nm, nr, nc, case=2, return_labels=True, p_contamination=0.1),
#     cont_shape_in=lambda nm,nr,nc: get_contaminated_contour_ensemble_shape(nm, nr, nc, return_labels=True, scale=0.01, freq=0.01, p_contamination=0.1),
#     cont_shape_out=lambda nm,nr,nc: get_contaminated_contour_ensemble_shape(nm, nr, nc, return_labels=True, scale=0.05, freq=0.05, p_contamination=0.1),
#     cont_topo=lambda nm, nr, nc: get_contaminated_contour_ensemble_topological(nm, nr, nc, return_labels=True, p_contamination=0.1),
# )

datasets = dict(
    # no_cont=lambda nm,nr,nc: dataset_no_outliers(nm, nr, return_labels=True),
    # cont_mag_sym=lambda nm,nr,nc: dataset_sym_mag_outliers(nm, nr, return_labels=True),
    # cont_mag_peaks=lambda nm,nr,nc: dataset_peaks_mag_outliers(nm, nr, return_labels=True),
    # cont_shape_in=lambda nm,nr,nc: dataset_in_shape_outliers(nm, nr, return_labels=True),
    # cont_shape_out=lambda nm,nr,nc: dataset_out_shape_outliers(nm, nr, return_labels=True),
    cont_topo=lambda nm, nr, nc: dataset_topological_outliers(nm, nr, return_labels=True),
)

methods = dict(
    # cbd=lambda ensemble, td: band_depth.compute_depths(ensemble, modified=False, times_dict=td),
    # mcbd=lambda ensemble, td: band_depth.compute_depths(ensemble, modified=True, target_mean_depth=None, times_dict=td),
    # bod=lambda ensemble, td: boundary_depth.compute_depths(ensemble, modified=False, times_dict=td),
    mbod=lambda ensemble, td: inclusion_depth.compute_depths(ensemble, modified=True, times_dict=td)
)

for method_name, method_fn in methods.items():
    method_res_path = res_dir.joinpath(method_name)
    if not method_res_path.exists():
        print(f"Creating: {method_res_path}")
        method_res_path.mkdir()

    for dataset_name, dataset_fn in datasets.items():
        md_res_path = method_res_path.joinpath(dataset_name)
        if not md_res_path.exists():
            print(f"Creating: {md_res_path}")
            md_res_path.mkdir()

        for r in range(num_replications):  # replications per dataset
            mdr_res_path = md_res_path.joinpath(f"replication_{r}")
            if not mdr_res_path.exists():
                print(f"Creating: {mdr_res_path}")
                mdr_res_path.mkdir()

            entries_path = mdr_res_path.joinpath(f"{method_name}-{dataset_name}-{r}.pkl")

            if entries_path.exists():
                print(f"Result already exists at: {entries_path}")

            else:

                entries = []
                for s in sizes:
                    if not (s > size_limit_cbd and "cbd" in method_name):
                        dataset_path = ens_dir.joinpath(f"{dataset_name}-{s}-{r}.pkl")

                        if dataset_path.exists():
                            print(f"Loading: {dataset_path}")
                            with open(dataset_path, "rb") as f:
                                ensemble, labs = pickle.load(f)
                        else:
                            print(f"Creating: {dataset_path}")
                            ensemble, labs = dataset_fn(s, num_rows, num_cols)
                            with open(dataset_path, "wb") as f:
                                pickle.dump([ensemble, labs], f)

                        print(
                            f"Processing: dataset: {dataset_name} - size: {s} - method: {method_name} - replication: {r}")
                        entry = dict(dataset_name=dataset_name, size=s, replication_id=r)

                        depths = method_fn(ensemble, entry)  # each method logs its times

                        entry["method"] = method_name
                        entry["method_family"] = "cbd" if "cbd" in method_name else "bod"
                        entry[f"depths"] = depths

                        entries.append(entry)

                with open(entries_path, "wb") as f:
                    pickle.dump(entries, f)
