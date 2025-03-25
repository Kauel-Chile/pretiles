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
    mean_z = np.mean(z)
    tree = cKDTree(np.column_stack((x, y)))
    
    z_interp = np.zeros(grid_x.shape)
    mask = np.ones(grid_x.shape)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            point = np.array([grid_x[i, j], grid_y[i, j]])
            distances, indices = tree.query(point, k=2)  # k nearest neighbors
            
            valid_mask = (distances > threshold_down) & (distances < threshold_up)
            valid_distances = distances[valid_mask]
            valid_indices = indices[valid_mask]
            
            if len(valid_distances) == 0:
                z_interp[i, j] = mean_z
                mask[i, j] = 0
            else:
                weights = 1 / ((valid_distances ** power) + epsilon)
                z_interp[i, j] = np.sum(weights * z[valid_indices]) / (np.sum(weights) + epsilon)
    
    return z_interp, mask


def create_digital_elevation_model(x, y, z):
    z_mean = np.mean(z)
    
    # Calcular límites del grid en coordenadas centradas
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    # Crear grid en coordenadas centradas
    w = int(np.ceil((x_max - x_min) * 4))
    h = int(np.ceil((y_max - y_min) * 4))

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, w),
        np.linspace(y_min, y_max, h)
    )
    
    # Interpolación IDW (usando z centrado)
    z_interp, mask = idw_interpolation(
        x, y, z - z_mean,
        grid_x, grid_y,
        np.min(z - z_mean), np.max(z - z_mean)
    )
    
    return {
        'dem': z_interp,
        'mask': mask,
        'grid_x': grid_x,  # Grid en coordenadas originales
        'grid_y': grid_y,
    }