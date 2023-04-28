"""
Here we explore how to leverage depths
and the orderings they induce to perform
localized positional analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import rectangle

from backend.src.utils import get_distance_transform


# We will simulate the user zooming in or selecting different parts
# We will recompute the ordering based on the selection
# We plot the spaghetti plot with the centrality based on the selection

def load_blobs(shape=(300, 300), output_mask=True):
    from pathlib import Path
    from skimage.io import imread
    from skimage.transform import resize
    from skimage.morphology import dilation
    from skimage.color import rgb2gray
    from skimage.draw import polygon

    outputs = []
    for p in Path("/Users/chadepl/git/multimodal-contour-vis/backend/data/hand-drawn/procreate-blobs/").glob("*.png"):
        img = imread(p)
        img = rgb2gray(img[:,:,0:3])
        img = dilation(img)
        img = resize(img, shape)
        img[img > 0] = 1

        if output_mask:
            rr, cc = np.where(img == 1)
            rr1, cc1 = polygon(rr, cc, shape=(300, 300))
            mask = np.zeros((300, 300))
            mask[rr1, cc1] = 1

            outputs.append(mask)
        else:
            outputs.append(img)

    return outputs


#num_members = 100
num_cols = num_rows = 300
#masks = get_circles_dataset(300, 300, num_members)
masks = load_blobs()
num_members = len(masks)

sdfs = [get_distance_transform(m, tf_type="signed") for m in masks]

sdfs_arr = np.array([sdf.flatten() for sdf in sdfs])
sdfs_std = sdfs_arr.std(axis=0)

masks_arr = np.array([mask.flatten() for mask in masks])
masks_std = masks_arr.std(axis=0)

composite = masks_arr.mean(axis=0).reshape(-1, num_cols)

plt.imshow(composite)
plt.show()

from matplotlib import colors
divnorm=colors.TwoSlopeNorm(vmin=sdfs[0].min(), vcenter=0., vmax=sdfs[0].max())
plt.imshow(sdfs[10], cmap="seismic", norm=divnorm)
plt.show()


rect_selects = [(100, 100, 150, 150)]

rs_idx = rect_selects[0]
rs_rr_cc = rectangle(rs_idx[0:2], rs_idx[2:4], shape=(300, 300))
rs = masks_std.reshape(-1, 300)[rs_rr_cc[0], rs_rr_cc[1]]
plt.imshow(rs)
plt.show()