import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xarray as xr
from scipy.stats import pearsonr

from .. import config as cfg


def timeseries_skill(var, gt, eval_dir, name, start_year=1958, end_year=2020):

    var = np.nanmean(var, axis=(2, 3, 4))
    gt = np.nanmean(gt, axis=(2, 3, 4))
    var_ensmean = np.nanmean(var, axis=0)
    var_ensstd = np.nanstd(var, axis=0)
    gt_ensmean = np.nanmean(gt, axis=0)
    gt_ensstd = np.nanstd(gt, axis=0)

    correlation = np.corrcoef(var, gt)[0, 1]
    print(correlation)

    time = np.linspace(start_year, end_year, len(var_ensmean))

    # plot timeseries
    plt.figure(figsize=(10, 5))
    plt.title("Timeseries, Correlation: {:.2f}".format(correlation))
    plt.plot(time, var_ensmean, label="Cnn reconstruction", color="red")
    plt.plot(time, gt_ensmean, label="Assimilation Reanalysis", color="blue")
    # Shade the area of the ensemble standard deviation
    plt.fill_between(
        time,
        var_ensmean - var_ensstd,
        var_ensmean + var_ensstd,
        color="coral",
        alpha=0.3,
        label="CNN Ensemble Std Dev",
    )
    plt.fill_between(
        time,
        gt_ensmean - gt_ensstd,
        gt_ensmean + gt_ensstd,
        color="lightblue",
        alpha=0.3,
        label="Assimilation reanalysis Ensemble Std Dev",
    )
    plt.xlabel("Time in years")
    plt.ylabel("T in °C")
    plt.grid()
    plt.legend()
    plt.savefig(f"{eval_dir}/{name}_timeseries.pdf")
    plt.show()


def spatial_skill(var, gt, eval_dir, name):

    # Take mean over depth dimension if present
    var_ensmean = np.nanmean(var, axis=0)
    gt_ensmean = np.nanmean(gt, axis=0)
    if var_ensmean.ndim == 4:
        var = np.nanmean(var_ensmean, axis=1)
        gt = np.nanmean(gt_ensmean, axis=1)
    else:
        var = var_ensmean
        gt = gt_ensmean

    var = np.nan_to_num(var)
    gt = np.nan_to_num(gt)

    # Calculate spatial skill
    mse = np.zeros((var.shape[1], var.shape[2]))
    r2 = np.zeros((var.shape[1], var.shape[2]))

    for i in range(var.shape[1]):
        for j in range(var.shape[2]):
            var_ij = var[:, i, j]
            gt_ij = gt[:, i, j]

            mse[i, j] = mean_squared_error(gt_ij, var_ij)
            r2[i, j] = r2_score(gt_ij, var_ij)

    # plot spatial skill
    plt.figure(figsize=(16, 12))
    plt.subplot(2, 2, 1)
    plt.title("Mean Squared Error")
    plt.imshow(mse, vmin=0, vmax=1.5, cmap="Reds")
    plt.colorbar(label="mse in °C")
    plt.subplot(2, 2, 2)
    plt.title("Correlation")
    plt.imshow(r2, vmin=0, vmax=0.8, cmap="Reds")
    plt.colorbar(label="r-correlation coefficient")
    plt.subplot(2, 2, 3)
    plt.title("Mean T CNN Reconstruction")
    plt.imshow(
        np.where(np.mean(var, axis=0) == 0, np.nan), vmin=-1, vmax=1, cmap="coolwarm"
    )
    plt.colorbar(label="T in °C")
    plt.subplot(2, 2, 4)
    plt.title("Mean T Assimilation Reanalysis")
    plt.imshow(
        np.where(np.mean(gt, axis=0) == 0, np.nan), vmin=-1, vmax=1, cmap="coolwarm"
    )
    plt.colorbar(label="T in °C")
    plt.savefig(f"{eval_dir}/{name}_spatial.pdf")
    plt.show()
