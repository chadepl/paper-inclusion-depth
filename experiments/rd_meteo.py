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

    import sys
    sys.path.insert(0, "..")
    from src.depths import band_depth, inclusion_depth
    from src.vis_utils import plot_contour_boxplot, plot_contour_spaghetti

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

    t_tick = time()
    cbd_depths = band_depth.compute_depths(bin_masks, modified=True, target_mean_depth=None)
    print(f"cbd: {time() - t_tick}")
    t_tick = time()
    id_depths = inclusion_depth.compute_depths(bin_masks, modified=True)
    print(f"id: {time() - t_tick}")
    print()
    # print(cbd_depths)
    # print(id_depths)


    print("Outliers")
    out_mcbd = np.argsort(cbd_depths)[:10]
    out_mid = np.argsort(id_depths)[:10]
    print(f"{np.intersect1d(out_mid, out_mcbd).size}/12")
    print(out_mcbd)
    print(out_mid)

    print("Inliers")
    in_mcbd = np.argsort(cbd_depths)[::-1][:41]
    in_mid = np.argsort(id_depths)[::-1][:41]
    print(f"{np.intersect1d(in_mid, in_mcbd).size}/100")
    print(in_mcbd)
    print(in_mid)

    print("Score correlation")
    print(np.corrcoef(cbd_depths, id_depths))

    print("Masks comparison")
    med_cbd = bin_masks[in_mcbd[0]]
    med_id = bin_masks[in_mid[0]]
    mean_cbd = (np.array([bin_masks[e] for e in in_mcbd]).mean(axis=0)>0.5).astype(float)
    mean_id = (np.array([bin_masks[e] for e in in_mid]).mean(axis=0)>0.5).astype(float)

    print(f"MSE Med: {np.square(med_cbd - med_id).mean()}")
    print(f"MSE Mean: {np.square(mean_cbd - mean_id).mean()}")

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

    if False:
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
    if False:

        # depth computation

        #bin_masks = [bin_masks[i] for i in range(0, 20)]#np.random.choice(np.arange(len(bin_masks)), 10, replace=False)]

        plot_contours(bin_masks, cbd_depths, fname="/Users/chadepl/Downloads/cbd-meteo.png")
        plot_contours(bin_masks, id_depths, fname="/Users/chadepl/Downloads/id-meteo.png")

        fig, axs = plt.subplots(nrows=2)
        plot_contour_boxplot(bin_masks, cbd_depths, epsilon_out=int(len(bin_masks)*0.2), axis_off=False, smooth_line=False, ax=axs[0])
        plot_contour_boxplot(bin_masks, id_depths, epsilon_out=int(len(bin_masks)*0.2), axis_off=False, smooth_line=False, ax=axs[1])
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
                            id_depths,
                            epsilon_out=int(len(bin_masks) * 0.2),
                            axis_off=True,
                            smooth_line=False, ax=ax)
        ax.invert_yaxis()
        fig.savefig("/Users/chadepl/Downloads/id-bp-meteo.png")


        plt.scatter(cbd_depths, id_depths)
        plt.show()


        rootgrp.close()


    ##############
    # Plot lines #
    ##############

    def plot_masks(masks, colors, linestyles, linewidths, fname=None):
        img = skimage.io.imread(data_dir.joinpath("picking_background.png"), as_gray=True)
        flipped_img = img[::-1, :]

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
                ax.plot(x_new, y_new, c=colors[i], linestyle=linestyles[i], linewidth=linewidths[i])

        ax.invert_yaxis()
        ax.set_axis_off()
        plt.show()
        if fname is not None:
            fig.savefig(fname)

    plot_masks([med_cbd, med_id], ["gold", "gold"], ["solid", "dotted"], [2, 4], fname="/Users/chadepl/Downloads/meteo-cbd-vs-id-med.png")
    plot_masks([mean_cbd, mean_id], ["dodgerblue", "dodgerblue"], ["solid", "dotted"], [2, 4], fname="/Users/chadepl/Downloads/meteo-cbd-vs-id-mean.png")

