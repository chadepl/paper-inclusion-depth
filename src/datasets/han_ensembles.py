from pathlib import Path
import numpy as np
from skimage.measure import find_contours
from skimage.draw import polygon2mask
from skimage.transform import resize
import matplotlib.pyplot as plt


def get_han_slice_ensemble(num_rows, num_cols, path_to_ensembles, patient_id=0, structure_name="Parotid_R", slice_num=41):
    print(__file__)
    patient = ["HCAI-036", "HCAI-010"][patient_id]

    fn = Path(path_to_ensembles).joinpath(
        f"ensemble-{structure_name}-hptc/{patient}/ed_ensemble-v4_size-subsets-31-{structure_name}.npz").resolve()

    # fn = Path(f"/Users/chadepl/git/multimodal-contour-vis/backend/data/han_ensembles/hptc-{structure_name}-{patient}-{slice_num}.npz").resolve()
    print(fn)
    dataset = np.load(fn)
    img = dataset["img"][slice_num]
    gt = dataset["gt"][slice_num]  # TODO: should come formatted from file
    ensemble_probs = [v[slice_num] for k, v in dataset.items() if "ep" in k]

    shape_ensemble = ensemble_probs[0].shape

    # ensemble_probs = [ensemble[:, :, i] for i in range(ensemble.shape[-1])]
    # ensemble_contours = [find_contours(member, 0.6) for member in ensemble_probs]
    ensemble_contours = [find_contours(member, 0.8) for member in ensemble_probs]
    ensemble_masks = []
    for contour in ensemble_contours:
        mask = np.zeros(shape_ensemble)
        for c in contour:
            mask += polygon2mask(shape_ensemble, c)
        ensemble_masks.append(mask)

    img = resize(img, (num_rows, num_cols), order=1)
    gt = resize(gt, (num_rows, num_cols), order=0)
    ensemble_masks = [resize(mask, (num_rows, num_cols), order=0) for mask in ensemble_masks]

    return img, gt, ensemble_masks


if __name__ == "__main__":
    img, gt, ensemble = get_han_slice_ensemble(300, 300)
