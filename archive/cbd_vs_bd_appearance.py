"""
 In this script we compare border depths vs band depths
 We show that the proposed depths:
 - Produce similar results to CBD
 - Are faster to compute than CBD
 - Change based on the distances between members (while order remains the same).
 - Are sensitive to local changes in the distribution.
"""
from time import time
import numpy as np
import pandas as pd
from skimage.draw import ellipse, rectangle
from skimage.measure import find_contours
from scipy.interpolate import RegularGridInterpolator
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

from backend.src.utils import get_distance_transform
from backend.src.contour_depths import band_depth, border_depth
from backend.src.datasets.circles import circles_with_outliers, circles_different_radii_distribution
from backend.src.datasets.bd_paper import get_contaminated_contour_ensemble_shape
from backend.src.datasets.papers import ensemble_ellipses_cbp
from backend.src.datasets.bd_paper import get_contaminated_contour_ensemble_magnitude
from backend.src.datasets.bd_paper import get_contaminated_contour_ensemble_shape
from backend.src.datasets.han_ensembles import get_han_slice_ensemble
from backend.src.vis_utils import plot_contour_spaghetti, plot_contour_boxplot

num_members = 50
num_rows = num_cols = 300
pos = []

# ensemble = circles_different_radii_distribution(50, 300, 300)
ensemble = get_contaminated_contour_ensemble_shape(50, 300, 300)
# ensemble = circles_with_outliers(num_members, num_rows, num_cols, num_outliers=4)
# ensemble = ensemble_ellipses_cbp(num_members, num_rows, num_cols)
# ensemble = get_contaminated_contour_ensemble_magnitude(num_members, num_rows, num_cols)
# ensemble = get_contaminated_contour_ensemble_shape(num_members, num_rows, num_cols)
# img, gt, ensemble = get_han_slice_ensemble(num_rows, num_cols)

modified = False
band_depths = band_depth.compute_depths(ensemble, modified=modified, target_mean_depth=None).tolist()
boundary_depths = border_depth.compute_depths_fast(ensemble, modified=modified)

labels = ["band_depth", "boundary_depths"]

fig, axs = plt.subplots(ncols=2)
for i, ax in enumerate(axs):
    ax.set_title(labels[i])
plot_contour_spaghetti(ensemble, arr=band_depths, is_arr_categorical=False, ax=axs[0])
plot_contour_spaghetti(ensemble, arr=boundary_depths, is_arr_categorical=False, ax=axs[1])
plt.show()

fig, axs = plt.subplots(ncols=2)
for i, ax in enumerate(axs):
    ax.set_title(labels[i])
plot_contour_boxplot(ensemble, band_depths, epsilon_out=0.05, ax=axs[0])
plot_contour_boxplot(ensemble, boundary_depths, epsilon_out=0.01, ax=axs[1])
plt.show()

depths = pd.DataFrame([band_depths, boundary_depths])
depths = depths.T
depths.columns = labels
sns.pairplot(depths)
plt.show()
