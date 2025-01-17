import rasterio
import numpy as np

from scipy.spatial import cKDTree
from rasterio.transform import from_origin

def idw_interpolation(x, y, z, grid_x, grid_y, threshold_down, threshold_up, power=2, epsilon=1e-8):
    """
    IDW (Inverse Distance Weighting) Interpolation
    :param x, y, z: Coordinates and values of the point cloud.
    :param grid_x, grid_y: Coordinates of the grid where interpolation will be performed.
    :param power: Weight factor (usually 2)
    :param threshold_down, threshold_up: Thresholds to consider valid distances.
    :return: Interpolated value for each point in the grid.
    """
    tree = cKDTree(np.column_stack((x, y)))
    
    z_interp = np.zeros(grid_x.shape)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            point = np.array([grid_x[i, j], grid_y[i, j]])
            distances, indices = tree.query(point, k=2)  # k nearest neighbors
            
            valid_mask = (distances > threshold_down) & (distances < threshold_up)
            valid_distances = distances[valid_mask]
            valid_indices = indices[valid_mask]
            
            if len(valid_distances) == 0:
                z_interp[i, j] = 0
            else:
                weights = 1 / ((valid_distances ** power) + epsilon)
                z_interp[i, j] = np.sum(weights * z[valid_indices]) / (np.sum(weights) + epsilon)
    
    return z_interp


def create_digital_elevation_model(x, y, z, size_img=512, threshold_down=-7, threshold_up=8, save_img=False):
    """
    Create a Digital Elevation Model (DEM)
    :param x, y, z: Coordinates and values of the point cloud.
    :param size_img: Size of the output image (default is 512).
    :param threshold_down, threshold_up: Thresholds to consider valid distances.
    :param save_img: Boolean flag to save the DEM as an image file (default is False).
    :return: Interpolated DEM as a numpy array.
    """
    grid_x, grid_y = np.meshgrid(np.linspace(np.min(x), np.max(x), size_img), 
                                np.linspace(np.min(y), np.max(y), size_img))

    z_interp = idw_interpolation(x, y, z, grid_x, grid_y, threshold_down, threshold_up)

    if save_img:
        # Define the transformation (define the origin and resolution of the DEM)
        transform = from_origin(np.min(x), np.max(y), (np.max(x) - np.min(x)) / size_img, (np.max(y) - np.min(y)) / size_img)

        # Save the DEM as a GeoTIFF file
        with rasterio.open('dem.tif', 'w', driver='GTiff', height=z_interp.shape[0], width=z_interp.shape[1],
                           count=1, dtype=z_interp.dtype, crs='EPSG:4326', transform=transform) as dst:
            dst.write(z_interp, 1)

    return z_interp