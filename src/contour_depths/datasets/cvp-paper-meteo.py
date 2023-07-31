"""
Meteorological datasets from CVP paper
"""
import skimage.io

if __name__ == "__main__":
    from time import time
    from pathlib import Path
    import numpy as np
    from netCDF4 import Dataset
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours

    from src.contour_depths.depths import band_depth, boundary_depth
    from src.contour_depths.vis_utils import plot_contour_boxplot, plot_contour_spaghetti

    data_dir = Path("/Users/chadepl/git/multimodal-contour-vis/data/cvp-paper-meteo/")

    # Setting variables
    # Fig 1: 20121015; height: 500 hPa; 5600 m contour
    # Fig 2: 20121017; height: 925 hPa; 680 m contour

    config = [
        ["20121015", "120", 500, 5600],
        ["20121017", "108",]
    ][0]

    # Loading and preprocessing data

    f = data_dir.joinpath(f"{config[0]}_00_ecmwf_ensemble_forecast.PRESSURE_LEVELS.EUR_LL10.{config[1]}.pl.nc")
    rootgrp = Dataset(f, "r", "NETCDF4")
    print("Data model: ")
    print(rootgrp.data_model)
    print()
    print("Dimensions: ")
    for dimobj in rootgrp.dimensions.values():
        print(dimobj)
    # print()
    # print("Variables: ")
    # for varobj in rootgrp.variables.values():
    #     print(varobj)

    geopot = rootgrp["Geopotential_isobaric"][...]
    geopot = geopot / 9.81
    geopot = geopot.squeeze()

    lat = rootgrp["lat"][...]
    lon = rootgrp["lon"][...]
    isobaric = rootgrp["isobaric"][...]

    print()
    print(rootgrp["Geopotential_isobaric"])
    print()
    print(lat.shape, lat[0], lat[-1])  # latitude is y-axis/rows
    print(lon.shape, lon[0], lon[-1])  # longitude is x-axis/cols

    ##########################
    # Full ensemble analysis #
    ##########################

    height_level = np.where(isobaric == config[2])[0]  # slice height we are interested in

    geopot = geopot[:, height_level, :, :].squeeze()
    geopot = np.moveaxis(geopot, [0, 1, 2], [0, 1, 2])

    geopot = np.flip(geopot, axis=1)  # flip x axis
    lat = np.flip(lat)  # flip x axis

    bin_masks = []
    for gp in geopot:
        bin_masks.append(np.zeros_like(gp))
        bin_masks[-1][gp <= config[3]] = 1  # we extract the iso line

    #################
    # Data overview #
    #################

    def plot_contours(masks, color_arr, fname=None):
        img = skimage.io.imread(data_dir.joinpath("picking_background.png"), as_gray=True)
        flipped_img = img[::-1, :]

        if len(color_arr) == 1:
            color_arr = [color_arr[0] for i in range(len(masks))]
        else:
            color_arr = np.array(color_arr).reshape(len(masks), -1)
            min_col_val = color_arr.min()
            max_col_val = color_arr.max()
            if max_col_val - min_col_val != 0:
                color_arr = (color_arr - min_col_val)/(max_col_val - min_col_val)
            color_arr = [plt.cm.get_cmap("magma")(v)[0] for v in color_arr]

        from scipy.interpolate import splprep, splev

        fig, ax = plt.subplots(layout="tight")
        ax.imshow(flipped_img, cmap="gray")
        for i, bm in enumerate(masks):
            row_shape = 1292 - 145
            col_shape = 1914 - 72
            adjusted_mask = skimage.transform.resize(bm, (row_shape, col_shape), order=1)
            contour = skimage.measure.find_contours(adjusted_mask, 0.5)
            for c in contour:
                sampling_rate = 1
                x = (c[:, 1] + 72)[::sampling_rate].tolist()
                y = (c[:, 0] + 145)[::sampling_rate].tolist()
                m = len(x)
                s = m * 10
                tck, u = splprep([x, y], u=None, s=s)
                contour_perc_points = 1.0
                u_new = np.linspace(u.min(), u.max(), int(len(x) * contour_perc_points))
                x_new, y_new = splev(u_new, tck, der=0)
                ax.plot(x_new, y_new, linewidth=2, c=color_arr[i])

        ax.invert_yaxis()
        ax.set_axis_off()
        plt.show()
        if fname is not None:
            fig.savefig(fname)


    img = skimage.io.imread(data_dir.joinpath("picking_background.png"), as_gray=True)
    flipped_img = img[::-1, :]

    fig, ax = plt.subplots(layout="tight")
    plt.imshow(flipped_img, cmap="gray")
    for i in range(51):
        # lat is between 70 and 30
        # in the image this is between 145 and 1292 pixels
        # lon is between -60 and 40
        # in the image this is between 72 and 1914 pixels
        row_shape = 1292 - 145
        col_shape = 1914 - 72
        field = skimage.transform.resize(geopot[i], (row_shape, col_shape))
        ax.contour(field, levels=[config[3], ], colors=[(136/255,86/255,167/255, 0.1)], extent=(72, 1914, 145, 1292))
    ax.set_axis_off()
    ax.invert_yaxis()
    plt.show()

    plot_contours(bin_masks, [(136/255,86/255,167/255, 0.5),], fname="/Users/chadepl/Downloads/spa-meteo.png")

    ###################
    # Contour boxplot #
    ###################

    # depth computation

    #bin_masks = [bin_masks[i] for i in range(0, 20)]#np.random.choice(np.arange(len(bin_masks)), 10, replace=False)]

    t_tick = time()
    cbd_depths = band_depth.compute_depths(bin_masks, modified=True, target_mean_depth=None)
    print(f"cbd: {time() - t_tick}")
    t_tick = time()
    bod_depths = boundary_depth.compute_depths(bin_masks, modified=True)
    print(f"bod: {time() - t_tick}")
    print()
    print(cbd_depths)
    print(bod_depths)

    plot_contours(bin_masks, cbd_depths, fname="/Users/chadepl/Downloads/cbd-meteo.png")
    plot_contours(bin_masks, bod_depths, fname="/Users/chadepl/Downloads/bod-meteo.png")

    fig, axs = plt.subplots(nrows=2)
    plot_contour_boxplot(bin_masks, cbd_depths, epsilon_out=int(len(bin_masks)*0.2), axis_off=False, smooth_line=False, ax=axs[0])
    plot_contour_boxplot(bin_masks, bod_depths, epsilon_out=int(len(bin_masks)*0.2), axis_off=False, smooth_line=False, ax=axs[1])
    for ax in axs:
        ax.set_xlim(0, 101)
        ax.set_ylim(0, 41)
    plt.show()

    print(f"Num outliers: {int(len(bin_masks) * 0.2)}")

    fig, ax = plt.subplots(layout="tight")
    plot_contour_boxplot(bin_masks,
                         cbd_depths,
                         epsilon_out=int(len(bin_masks) * 0.2),
                         axis_off=True,
                         smooth_line=False, ax=ax)
    ax.invert_yaxis()
    fig.savefig("/Users/chadepl/Downloads/cbd-bp-meteo.png")

    fig, ax = plt.subplots(layout="tight")
    plot_contour_boxplot(bin_masks,
                         bod_depths,
                         epsilon_out=int(len(bin_masks) * 0.2),
                         axis_off=True,
                         smooth_line=False, ax=ax)
    ax.invert_yaxis()
    fig.savefig("/Users/chadepl/Downloads/bod-bp-meteo.png")


    plt.scatter(cbd_depths, bod_depths)
    plt.show()


    rootgrp.close()