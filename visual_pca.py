# Import modules
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from ipywidgets import interact, IntSlider, IntRangeSlider
import matplotlib.gridspec as gridspec

import cartopy.crs as ccrs
from functools import partial

plt.rc('font', size=14)


def setup() :
    # Data directory
    DATA_DIR = Path('data')

    # Filename to geopotential height at 500hPa from MERRA-2 reanalysis
    START_DATE = '19800101'
    END_DATE = '20220101'
    filename = 'merra2_analyze_height_500_month_{}-{}.nc'.format(START_DATE, END_DATE)
    z500_label = 'Geopotential height (m)'

    ds = xr.load_dataset(DATA_DIR / filename)
    da = ds['height_500']
    da_mean = da.mean(dim='time')

    da_dev = da - da_mean

    weights = np.cos(np.deg2rad(da_dev.lat))
    da_w = da_dev * weights

    da_untrend =  da_w.groupby("time.month") -  da_w.groupby("time.month").mean()


    st = da_untrend.stack(points = ['lat', 'lon'])
    X = st.values
    C = np.cov(X.T)


    eigenvalues, eigenvectors = np.linalg.eigh(C)

    eigenvectors = eigenvectors[:, ::-1]
    eigenvalues = eigenvalues[::-1]
    return X, eigenvectors, eigenvalues, st, da_untrend


# Interactive plotting function
def plot_interactive( X, eigenvectors, st, da_untrend, t=10, k_min=0, k_max=4):
    # Efficiently update the reconstruction for only the selected time slice `t`
    reconstruction = X[t] @ eigenvectors[:, k_min : k_max] @ eigenvectors[:,  k_min : k_max].T
    da_reconstruction = xr.DataArray(data=reconstruction, coords=[st.coords['points']]).unstack('points')

    # Set up gridspec with more space for main plots on the left and smaller eigenvector plots on the right
    fig = plt.figure(figsize=[30, 10])
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1], height_ratios=[2, 1])  # Adjusting ratios

    # Plot Original and Reconstruction (large plots on the left)
    for i, (label, data_array) in enumerate({'Original': da_untrend.isel(time=t), f'Reconstruction k={k_min}:{k_max}': da_reconstruction}.items()):
        ax = fig.add_subplot(gs[0, i*2:(i*2 +2)], projection=ccrs.PlateCarree())  # Use first two columns
        ax.coastlines()
        ax.gridlines(draw_labels=False)
        # Setting fixed color limits and colormap for red-blue
        data_array.plot(ax=ax, add_colorbar=False, vmin=-100, vmax=100, cmap="RdBu")
        ax.set_title(f'{label} at time {da_untrend.isel(time=t).time.dt.strftime("%Y-%m").data}')

    # Add common color bar at the bottom for the main plots
    # plt.subplots_adjust(bottom=0.2)
    # cbar_ax = fig.add_axes([0.2, 0.05, 0.5, 0.02])  # Adjust width and position for balance
    # fig.colorbar(data_array.plot(vmin=-100, vmax=100, cmap="RdBu", add_colorbar=False),
    #             cax=cbar_ax, orientation='horizontal')

    # Plot the first 4 eigenvectors as smaller maps on the right
    for j in range(4):
        ax = fig.add_subplot(gs[1, j], projection=ccrs.PlateCarree())  # Position in right column
        ax.coastlines()
        ax.gridlines(draw_labels=False)
        eigenvector_map = xr.DataArray(eigenvectors[:, j], coords=[st.coords['points']]).unstack('points')
        eigenvector_map.plot(ax=ax, cmap="RdBu", add_colorbar=False)
        ax.set_title(f'Eigenvector {j + 1}')

    plt.show()



if __name__ == '__main__' :
    X, eigenvectors, eigenvalues, st, da_untrend = setup()
    pi = lambda t, k_min, k_max: plot_interactive(X=X, eigenvectors=eigenvectors, st=st, da_untrend=da_untrend, t=t, k_min=k_min, k_max=k_max)

    # Use `interact` to make the plot interactive with sliders for `t` and `k`
    interact(pi,
            t=IntSlider(min=0, max=st.coords['time'].size - 1, step=1, value=10, layout={'width': '2000px'}),
            k_min = IntSlider(min=0, max=eigenvectors.shape[1]-1, step=1, value=0, layout={'width': '2000px'}),
            k_max = IntSlider(min=0, max=eigenvectors.shape[1]-1, step=1, value=4, layout={'width': '2000px'}))
