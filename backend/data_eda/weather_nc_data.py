"""
Processing and exploration of weather data
in nc format.
Dataset description: https://dataplatform.knmi.nl/dataset/qrf-rt-ssh-v2021
"""

from pathlib import Path
import netCDF4 as nc
from skimage.measure import find_contours
from skimage.transform import rescale
import matplotlib.pyplot as plt

ds = nc.Dataset("../data/weather/QRF-RT-SSh_2022041300_050_002.nc")

# Access metadata
print(ds)
print(ds.__dict__)  # allows accessing them by their key
print()

# Accessing variables and dimensions metadata
for dim in ds.dimensions.values():
    print(dim)

for var in ds.variables.values():
    print(var)


prcp = ds['Precipitation'][:]
print(prcp.shape)

def plot_ensemble_grid(ensemble):
    fig, axs = plt.subplots(nrows=10, ncols=5, figsize=(5,10))
    axs = axs.flatten()
    for i in range(50):
        pred = ensemble[0, i, :, :]
        print(pred.shape)
        pred = rescale(pred, 1, order=0, anti_aliasing=False)
        contours = find_contours(pred, level=0.0001)
        print(pred.min(), pred.mean(), pred.max())
        print(pred.shape)
        axs[i].imshow(pred)
        if len(contours) > 0:
            axs[i].plot(contours[0][:,1], contours[0][:,0], c="red")
        axs[i].set_axis_off()
    plt.show()



plot_ensemble_grid(prcp)


for p in Path("../data/weather/2022041212").rglob("*.nc"):
    print(p)
    ds = nc.Dataset(p)
    plot_ensemble_grid(ds["Precipitation"][:])
    break

