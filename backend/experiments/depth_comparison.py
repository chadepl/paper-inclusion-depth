


"""
Depths can be obtained in different ways
But their results might vary
Here we try to ways to obtain depths
- A: based on distances
- B: based on the distance matrix

we also compute timings

We use different datasets to identify where one
depth might be better than others.
Note that the goal of computing depths is to stablish data orderings that make sense.

What can vary in an ensemble of shapes (from top to bottom):
- Shape globally
- Shape locally
- Reflections
- Rotation
- Scale
- Position
"""

from time import time

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform
from skimage.draw import disk, rectangle, line, polygon
from skimage.transform import rotate
import matplotlib.pyplot as plt

from backend.src.utils import get_distance_transform
from backend.src.contour_band_depth import get_depth_matrix, get_contour_band_depths
from backend.src.vis_utils import plot_contour_spaghetti
from backend.src.simple_shapes_datasets import get_circles_dataset, get_circles_with_shape_outliers, get_rectangles_dataset, get_lines_dataset, get_shapes_dataset



num_members = 100
num_cols = num_rows = 300
#masks = get_circles_dataset(num_cols, num_rows, num_members, variation="radii")
#masks = get_circles_with_shape_outliers(num_cols, num_rows, num_members, fraction_outliers=0.02)
#masks = get_rectangles_dataset(num_cols, num_rows, num_members, variation="rotation")
#masks = get_lines_dataset(num_cols, num_rows, num_members, variation="equispaced")
masks = get_shapes_dataset(num_cols, num_rows, num_members)

composite = np.zeros_like(masks[0])
for cm in masks:
    composite += cm/num_members

plt.imshow(composite, cmap="magma")
plt.show()

t_start = time()

depth_data = get_contour_band_depths(masks, 2)
dm = get_depth_matrix(depth_data)
th_dm = get_depth_matrix(depth_data, threshold=0.1)
plt.matshow(th_dm)
plt.show()

depths_cb = th_dm.mean(axis=1)

t_end = time()

print(f"Contour band depths took {t_end - t_start} seconds to compute")

t_start = time()

sdf_mat = [get_distance_transform(cm).flatten() for cm in masks]
sdf_mat = np.array(sdf_mat)
# sdf_dists = squareform(pdist(sdf_mat))

sdf_l1_depths = []
for i in range(sdf_mat.shape[0]):
    vecsum = np.zeros_like(sdf_mat[i, :])
    for j in range(sdf_mat.shape[0]):
        if i != j:
            diff = sdf_mat[j,:] - sdf_mat[i,:]
            uvec = diff/(norm(diff) + 1e-12)
            vecsum += uvec
    vecsum = vecsum / (sdf_mat.shape[0] - 1)
    sdf_l1_depths.append((i, 1 - norm(vecsum)))

t_end = time()

print(f"SDF-based l1 depths took {t_end - t_start} seconds to compute")


depths_sdf = [e[1] for e in sdf_l1_depths]
depths_sdf = np.array(depths_sdf)
#
# for i, cm in enumerate(circle_masks):
#     plt.plot()

desc = False
order = -1 if desc else 1

fig, ax = plt.subplots()
plot_contour_spaghetti(masks, memberships=depths_cb, highlight=np.argsort(depths_cb)[::order][:5], ax=ax)
ax.set_title(f"Contour band depths ({'top' if desc else 'bottom'})")
plt.show()

fig, ax = plt.subplots()
plot_contour_spaghetti(masks, memberships=depths_sdf, highlight=np.argsort(depths_sdf)[::order][:5], ax=ax)
ax.set_title(f"L1 depths ({'top' if desc else 'bottom'})")
plt.show()

import matplotlib.cm as cm
arg_sort_cb = np.argsort(depths_cb)
arg_sort_sdf = np.argsort(depths_sdf)

plt.scatter(np.arange(num_members), depths_cb[arg_sort_cb], s=5, c=[cm.magma(i/num_members) for i in arg_sort_cb], zorder=1)
plt.scatter(np.arange(num_members), depths_sdf[arg_sort_sdf], s=5, c=[cm.magma(i/num_members) for i in arg_sort_sdf], zorder=1)

for i, (i_dcb, dcb) in enumerate(zip(arg_sort_cb, depths_cb[arg_sort_cb])):
    for j, (i_dsdf, dsdf) in enumerate(zip(arg_sort_sdf, depths_sdf[arg_sort_sdf])):
        if i_dcb == i_dsdf:
            plt.plot([i, j], [dcb, dsdf], zorder=0, c=cm.magma(i_dcb/num_members))
            break


plt.show()