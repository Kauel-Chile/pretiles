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
    # Calcular medias para deshacer el centrado después
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)
    
    # Centrar coordenadas (mejora estabilidad numérica)
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    # Calcular límites del grid en coordenadas centradas
    x_min, x_max = np.min(x_centered), np.max(x_centered)
    y_min, y_max = np.min(y_centered), np.max(y_centered)
    
    # Crear grid en coordenadas centradas
    w = int(np.ceil((x_max - x_min) * 4))
    h = int(np.ceil((y_max - y_min) * 4))

    grid_x_centered, grid_y_centered = np.meshgrid(
        np.linspace(x_min, x_max, w),
        np.linspace(y_min, y_max, h)
    )
    
    # Interpolación IDW (usando z centrado)
    z_interp, mask = idw_interpolation(
        x_centered, y_centered, z - z_mean,
        grid_x_centered, grid_y_centered,
        np.min(z - z_mean), np.max(z - z_mean)
    )
    
    return {
        'dem': z_interp,
        'mask': mask,
        'grid_x': grid_x_centered + x_mean,  # Grid en coordenadas originales
        'grid_y': grid_y_centered + y_mean,
    }