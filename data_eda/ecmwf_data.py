"""
ecmwf data
https://github.com/ecmwf/notebook-examples/blob/master/opencharts/medium-ens-tp-2t.ipynb
https://github.com/ecmwf/notebook-examples/blob/master/opencharts/medium-t500-mean-spread.ipynb
https://github.com/ecmwf/ecmwf-opendata
https://www.ecmwf.int/en/forecasts/datasets
https://www.ecmwf.int/en/about/media-centre/focus/2017/fact-sheet-ensemble-weather-forecasting
https://www.ecmwf.int/en/forecasts/dataset/operational-archive
https://events.ecmwf.int/event/296/contributions/3248/attachments/1883/3385/UEF2022_Vuckovic.pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import ecmwf.data as ecdata
from magpye import GeoMap
from ecmwf.opendata import Client

client = Client("ecmwf", beta=True)

parameters = ["2t", "tp"]
filename = Path('medium-ens-tp-2t.grib')

if not filename.exists():
    client.retrieve(
        date=0,
        time=0,
        step=[21, 24],
        stream="enfo",
        type=["cf", "pf"],
        levtype="sfc",
        # levelist=[500],
        param=parameters,
        target=str(filename)
    )

data = ecdata.read(str(filename))

print(data.describe())

t2m = data.select(shortName="2t", step=24)


# gh_em /= 10
#
# fig = GeoMap(area_name="europe")
# fig.save("/Users/chadepl/Downloads/test.png")

def get_field_set_arr(fso):
    latitudes = ecdata.latitudes(fso)  # y
    longitudes = ecdata.longitudes(fso)  # x
    data_arr = ecdata.values(fso)

    ys = ((latitudes - latitudes.min()) / (latitudes.max() - latitudes.min()) * 299).astype(int)
    xs = ((longitudes - longitudes.min()) / (longitudes.max() - longitudes.min()) * 299).astype(int)

    num_rows = (ys.max() - ys.min()) + 1
    num_cols = (xs.max() - xs.min()) + 1

    data_square = np.zeros((num_rows, num_cols))

    for i, (x, y) in enumerate(zip(xs, ys)):
        data_square[y, x] = data_arr[i]

    return data_square


# data_square = get_field_set_arr(t2m)
#
# contours = find_contours(data_square, 260)
#
# plt.imshow(data_square)
# for c in contours:
#     plt.plot(c[:,1], c[:,0], c="turquoise")
# plt.show()


ensemble_fs = data.select(shortName="2t", step=24)
data_squares = []
ds_contours = []

for member_id in range(51):
    ds = get_field_set_arr(ensemble_fs[member_id])
    data_squares.append(ds)
    contours = find_contours(ds, 260)
    ds_contours.append(contours)

# One member, different isovalues

# ds = data_squares[0]
# plt.imshow(ds)
# for i in np.linspace(ds.min(), ds.max(), 10):
#     contours = find_contours(ds, i)
#     for c in contours:
#         plt.plot(c[:, 1], c[:, 0], c="turquoise", linewidth=1)
# plt.show()


# All members, one isovalue

dsq_arr = np.concatenate([dsq[:, :, np.newaxis] for dsq in data_squares], axis=-1)
print(dsq_arr.mean())

plt.imshow(data_squares[0])
for i in range(len(data_squares)):
    ds = data_squares[i]
    contours = find_contours(ds, 280)
    for c in contours:
        plt.plot(c[:, 1], c[:, 0], linewidth=1, alpha=0.5)  # , c="turquoise")
plt.xlim(150, 190)
plt.ylim(180, 220)
plt.show()

plt.imshow(data_squares[0][180:220, 150:190])
contours = find_contours(data_squares[0][180:220, 150:190], 280)
for c in contours:
    plt.plot(c[:, 1], c[:, 0], linewidth=3, alpha=0.5)  # , c="turquoise")
plt.show()

from skimage.transform import resize
from backend.src.contour_depths import border_depth
from backend.src.vis_utils import plot_contour_boxplot

binary_masks = [resize((data_squares[i][180:220, 150:190] > 280).astype(float), (300, 300)) for i in
                range(len(data_squares))]
depths = border_depth.compute_depths(binary_masks, use_fast=True)
plot_contour_boxplot(binary_masks, depths)
plt.show()
# plt.imshow(binary_masks[0])
# plt.show()


# import geoplot as gplt
# import pandas as pd
# import geopandas as gpd
# import geoplot.crs as gcrs
#
#
# latitudes = ecdata.latitudes(data[0])  # y
# longitudes = ecdata.longitudes(data[0])  # x
# data_arr = ecdata.values(data[0])
#
# gpd_df = pd.DataFrame([latitudes, longitudes, data_arr]).T
# gpd_df.columns = ["latitudes", "longitudes", "values"]
# geometry = gpd.points_from_xy(gpd_df.longitudes, gpd_df.latitudes)
# gpd_df = gpd.GeoDataFrame(gpd_df, crs=gcrs.EqualEarth, geometry=geometry)
#
# gpd_df.plot()
# plt.show()
